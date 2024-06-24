""" mpc.py: Model Predictive Controller, Objectives and Constraints"""
import copy

from ddmpc.controller.conventional import Controller
from ddmpc.controller.model_predictive.nlp import NLP, NLPSolution
from ddmpc.utils.file_manager import file_manager
from ddmpc.utils.pickle_handler import read_pkl, write_pkl
from ddmpc.utils.plotting import *
import matlab.engine
from ddmpc.modeling.process_models.utils.adapters import StateSpace_ABCDE,par_vals2SSvectors
from ddmpc.modeling.features.features import Feature, Source, Constructed, Controlled, Control,Connection
from ddmpc.utils.modes import Economic, Steady
from ddmpc.controller.model_predictive.costs import Cost, AbsoluteLinear, Quadratic


class ModelPredictive(Controller):
    """ model predictive controller that can handel multiple Objective's, Constraint's and Predictor's """

    def __init__(
            self,
            nlp:                NLP,
            step_size:          int,
            forecast_callback:  Callable,
            solution_plotter:   Optional[Plotter] = None,
            show_solution_plot: bool = False,
            save_solution_plot: bool = True,
            save_solution_data: bool = True,
            state_spaces:       list[StateSpace_ABCDE] = None,
    ):
        """ Model Predictive Controller """

        super(ModelPredictive, self).__init__(step_size=step_size)

        self.nlp:                   NLP = nlp
        self._forecast_callback:    Callable = forecast_callback

        self._solution_plotter:     Plotter = solution_plotter

        self.show_solution_plot: bool = show_solution_plot
        self.save_solution_plot: bool = save_solution_plot
        self.save_solution_data: bool = save_solution_data
        self.par_vals = [0]
        self.par_ids = [0]
        self.eng = None
        self.state_spaces = state_spaces
        self.state_space_joined: StateSpace_ABCDE = StateSpace_ABCDE()

    def setupMatlab(self,state_spaces: list[StateSpace_ABCDE] = None):
        '''
        Set the parameters for the MPC in the workspace of the matlab engine. IMPORTANT:
        This function has to be called after the nlp.build function. And then send the
        reidentified state space models.
        '''
        if state_spaces is None:
            state_spaces = self.state_spaces
        self.state_spaces = state_spaces

        #OJO QUE ESTAMOS TENIENDO EN CUENTA QUE ES PARA LOS 
        #REGRESORES, DONDE SIEMPRE SE TIENE A COMO EYE Y B COMO ZEROS.

        self.state_space_joined = StateSpace_ABCDE()
        for (n,state_space) in enumerate(self.state_spaces):
            if n == 0:
                self.state_space_joined = copy.deepcopy(state_space)
            else:

                # Add y features
                self.state_space_joined.add_y(state_space.SS_y[0])
                self.state_space_joined.set_y_offset(state_space.y_offset[0],len(state_space.y_offset))

                #Add x features
                nx_added = 0
                for x in state_space.get_extended_vector(state_space.SS_x):
                    if x not in self.state_space_joined.get_extended_vector(self.state_space_joined.SS_x):
                        nx_added+=1
                        for (i,f) in enumerate(self.state_space_joined.SS_x):
                            if f.name == x[0]:
                                self.state_space_joined.SS_x[i].lag = max(self.state_space_joined.SS_x[i].lag, x[1])
                        A_new = np.eye(1+self.state_space_joined.A.shape[0])
                        self.state_space_joined.set_A(A_new)
                        B_new = np.zeros((self.state_space_joined.B.shape[0]+1,self.state_space_joined.B.shape[1]))
                        self.state_space_joined.set_B(B_new)

                C_new = np.zeros((self.state_space_joined.C.shape[0]+1,self.state_space_joined.C.shape[1]+nx_added))
                if nx_added>0:
                    C_new[:-1,:-nx_added] = self.state_space_joined.C
                else:
                    C_new[:-1,:] = self.state_space_joined.C
                for (i,x) in enumerate(state_space.get_extended_vector(state_space.SS_x)):
                    for (j,x_joined) in enumerate(self.state_space_joined.get_extended_vector(self.state_space_joined.SS_x)):
                        if x==x_joined:
                            C_new[-1,j] = state_space.C[0,i]
                self.state_space_joined.set_C(C_new)

                # Add u features
                nu_added = 0
                for u in state_space.get_extended_vector(state_space.SS_u):
                    if u not in self.state_space_joined.get_extended_vector(self.state_space_joined.SS_u):
                        nu_added+=1
                        for (i,f) in enumerate(self.state_space_joined.SS_u):
                            if f.name == u[0]:
                                self.state_space_joined.SS_u[i].lag = max(self.state_space_joined.SS_u[i].lag, u[1])
                        B_new = np.zeros((self.state_space_joined.B.shape[0],self.state_space_joined.B.shape[1]+1))
                        self.state_space_joined.set_B(B_new)

                D_new = np.zeros((self.state_space_joined.D.shape[0]+1,self.state_space_joined.D.shape[1]+nu_added))
                if nx_added>0:
                    D_new[:-1,:-nu_added] = self.state_space_joined.D
                else:
                    D_new[:-1,:] = self.state_space_joined.D
                for (i,u) in enumerate(state_space.get_extended_vector(state_space.SS_u)):
                    for (j,u_joined) in enumerate(self.state_space_joined.get_extended_vector(self.state_space_joined.SS_u)):
                        if u==u_joined:
                            D_new[-1,j] = state_space.D[0,i]
                self.state_space_joined.set_D(D_new)
                        
                # Add d features
                nd_added = 0
                for d in state_space.get_extended_vector(state_space.SS_d):
                    if d not in self.state_space_joined.get_extended_vector(self.state_space_joined.SS_d):
                        nd_added+=1
                        for (i,f) in enumerate(self.state_space_joined.SS_d):
                            if f.name == d[0]:
                                self.state_space_joined.SS_d[i].lag = max(self.state_space_joined.SS_d[i].lag, d[1])

                E_new = np.zeros((self.state_space_joined.E.shape[0]+1,self.state_space_joined.E.shape[1]+nd_added))
                if nd_added>0:
                    E_new[:-1,:-nd_added] = self.state_space_joined.E
                else:
                    E_new[:-1,:] = self.state_space_joined.E
                for (i,d) in enumerate(state_space.get_extended_vector(state_space.SS_d)):
                    for (j,d_joined) in enumerate(self.state_space_joined.get_extended_vector(self.state_space_joined.SS_d)):
                        if d==d_joined:
                            E_new[-1,j] = state_space.E[0,i]
                self.state_space_joined.set_E(E_new)
                


        # Set the objective functions in the workspace
        # We will assume just linear and quadratic cost functions, so there will be _quad and _lin weighting matrices.
        # nc = sum([1 for objective in self.nlp.objectives if objective.cost.__class__.__name__ == 'Quadratic'])
        
        for objective in self.nlp.objectives:
            if objective.feature.source in [f .source if hasattr(f, 'source') else f for f in self.state_space_joined.SS_y]:
                pass
            else:
                if isinstance(objective.feature, Control):
                    # Convertir una y que apunte a la variable de control
                    for (i,u) in enumerate(self.state_space_joined.get_extended_vector(self.state_space_joined.SS_u)):
                        if u == (objective.feature.source.read_name,0):
                            y_new = copy.deepcopy(self.state_space_joined.SS_u[i])
                            y_new.lag = 1
                            self.state_space_joined.add_y(y_new)
                            self.state_space_joined.set_y_offset(0.,len(self.state_space_joined.y_offset))
                            C_new = np.zeros((self.state_space_joined.C.shape[0]+1,self.state_space_joined.C.shape[1]))
                            C_new[:-1,:] = self.state_space_joined.C
                            self.state_space_joined.set_C(C_new)
                            D_new = np.zeros((self.state_space_joined.D.shape[0]+1,self.state_space_joined.D.shape[1]))
                            D_new[:-1,:] = self.state_space_joined.D
                            D_new[-1,i] = 1
                            self.state_space_joined.set_D(D_new)
                            E_new = np.zeros((self.state_space_joined.E.shape[0]+1,self.state_space_joined.E.shape[1]))
                            E_new[:-1,:] = self.state_space_joined.E
                            self.state_space_joined.set_E(E_new)
                elif  isinstance(objective.feature, Controlled):
                    # Convertir una y que apunte a la variable de estado
                    for (i,x) in enumerate(self.state_space_joined.get_extended_vector(self.state_space_joined.SS_x)):
                        if x == (objective.feature.source.read_name,0):
                            y_new = copy.deepcopy(self.state_space_joined.SS_x[i])
                            y_new.lag = 1
                            self.state_space_joined.add_y(y_new)
                            self.state_space_joined.set_y_offset(0.,len(self.state_space_joined.y_offset))
                            C_new = np.zeros((self.state_space_joined.C.shape[0]+1,self.state_space_joined.C.shape[1]))
                            C_new[:-1,:] = self.state_space_joined.C
                            C_new[-1,i] = 1
                            self.state_space_joined.set_C(C_new)
                            D_new = np.zeros((self.state_space_joined.D.shape[0]+1,self.state_space_joined.D.shape[1]))
                            D_new[:-1,:] = self.state_space_joined.D
                            self.state_space_joined.set_D(D_new)
                            E_new = np.zeros((self.state_space_joined.E.shape[0]+1,self.state_space_joined.E.shape[1]))
                            E_new[:-1,:] = self.state_space_joined.E
                            self.state_space_joined.set_E(E_new)

                elif isinstance(objective.feature, Connection):
                    y_new = copy.deepcopy(objective.feature.source)
                    self.state_space_joined.add_y(y_new)
                    self.state_space_joined.set_y_offset(0.,len(self.state_space_joined.y_offset))
                    # Convertir un y que apunte al incremento de la variable de estado o acción de control
                    for (i,x) in enumerate(self.state_space_joined.get_extended_vector(self.state_space_joined.SS_x)):
                        if x == (objective.feature.source.base.col_name,0):
                            C_new = np.zeros((self.state_space_joined.C.shape[0]+1,self.state_space_joined.C.shape[1]))
                            C_new[:-1,:] = self.state_space_joined.C
                            C_new[-1,i] = 1
                            C_new[-1,i+1] = -1
                            self.state_space_joined.set_C(C_new)
                            D_new = np.zeros((self.state_space_joined.D.shape[0]+1,self.state_space_joined.D.shape[1]))
                            D_new[:-1,:] = self.state_space_joined.D
                            self.state_space_joined.set_D(D_new)
                            E_new = np.zeros((self.state_space_joined.E.shape[0]+1,self.state_space_joined.E.shape[1]))
                            E_new[:-1,:] = self.state_space_joined.E
                            self.state_space_joined.set_E(E_new)
                    for (i,u) in enumerate(self.state_space_joined.get_extended_vector(self.state_space_joined.SS_u)):
                        if u == (objective.feature.source.base.col_name,0):
                            C_new = np.zeros((self.state_space_joined.C.shape[0]+1,self.state_space_joined.C.shape[1]))
                            C_new[:-1,:] = self.state_space_joined.C
                            self.state_space_joined.set_C(C_new)
                            D_new = np.zeros((self.state_space_joined.D.shape[0]+1,self.state_space_joined.D.shape[1]))
                            D_new[:-1,:] = self.state_space_joined.D
                            D_new[-1,i] = 1
                            D_new[-1,i+1] = -1
                            self.state_space_joined.set_D(D_new)
                            E_new = np.zeros((self.state_space_joined.E.shape[0]+1,self.state_space_joined.E.shape[1]))
                            E_new[:-1,:] = self.state_space_joined.E
                            self.state_space_joined.set_E(E_new)
                else:
                    raise ValueError(f"Objective {objective} feature is not Control, Controlled or Change")
                
                
        if self.eng is None:
            self.eng = matlab.engine.start_matlab()
        
        # Set the state space matrices in the workspace
        self.eng.workspace['A'] = self.state_space_joined.A
        self.eng.workspace['B'] = self.state_space_joined.B
        self.eng.workspace['C'] = self.state_space_joined.C
        self.eng.workspace['D'] = self.state_space_joined.D
        self.eng.workspace['E'] = self.state_space_joined.E
        print("y_offset",self.state_space_joined.y_offset)
        self.eng.workspace['y_offset'] = self.state_space_joined.y_offset




        # Set the contraints in the workspace
        constraints = {constraint.feature.source.col_name: (constraint.lb, constraint.ub) for constraint in self.nlp.constraints}

        x_lb = np.zeros((self.state_space_joined.get_nx(),1))
        x_ub = np.zeros((self.state_space_joined.get_nx(),1))
        x_extended = self.state_space_joined.get_extended_vector(self.state_space_joined.SS_x)
        for i in range(self.state_space_joined.get_nx()):
            if x_extended[i][0] in constraints.keys():
                x_lb[i] = constraints[x_extended[i][0]][0]
                x_ub[i] = constraints[x_extended[i][0]][1]
            else:
                x_lb[i] = -np.inf
                x_ub[i] = np.inf
        self.eng.workspace['x_lb'] = x_lb
        self.eng.workspace['x_ub'] = x_ub


        u_lb = np.zeros((self.state_space_joined.get_nu(),1))
        u_ub = np.zeros((self.state_space_joined.get_nu(),1))
        u_extended = self.state_space_joined.get_extended_vector(self.state_space_joined.SS_u)
        for i in range(self.state_space_joined.get_nu()):
            if u_extended[i][0] in constraints.keys():
                u_lb[i] = constraints[u_extended[i][0]][0]
                u_ub[i] = constraints[u_extended[i][0]][1]
            else:
                u_lb[i] = -np.inf
                u_ub[i] = np.inf
        self.eng.workspace['u_lb'] = u_lb
        self.eng.workspace['u_ub'] = u_ub

        d_lb = np.zeros((self.state_space_joined.get_nd(),1))
        d_ub = np.zeros((self.state_space_joined.get_nd(),1))
        d_extended = self.state_space_joined.get_extended_vector(self.state_space_joined.SS_d)
        for i in range(self.state_space_joined.get_nd()):
            if d_extended[i][0] in constraints.keys():
                d_lb[i] = constraints[d_extended[i][0]][0]
                d_ub[i] = constraints[d_extended[i][0]][1]
            else:
                d_lb[i] = -np.inf
                d_ub[i] = np.inf
        self.eng.workspace['d_lb'] = d_lb
        self.eng.workspace['d_ub'] = d_ub

        y_lb_day = np.zeros((self.state_space_joined.get_ny(),1))
        y_ub_day = np.zeros((self.state_space_joined.get_ny(),1))
        y_lb_night = np.zeros((self.state_space_joined.get_ny(),1))
        y_ub_night = np.zeros((self.state_space_joined.get_ny(),1))
        y_extended = self.state_space_joined.get_extended_vector(self.state_space_joined.SS_y)
        for i in range(self.state_space_joined.get_ny()):
            if y_extended[i][0] in constraints.keys():
                y_lb_day[i] = constraints[y_extended[i][0]][0]
                y_ub_day[i] = constraints[y_extended[i][0]][1]
                y_lb_night[i] = constraints[y_extended[i][0]][0]
                y_ub_night[i] = constraints[y_extended[i][0]][1]
            else:
                y_lb_day[i] = -np.inf
                y_ub_day[i] = np.inf
                y_lb_night[i] = -np.inf
                y_ub_night[i] = np.inf


        # Set the objectives in the workspace. Note that some objectives are translated into constraints
        ny = self.state_space_joined.get_ny()
        S_q = np.zeros((ny,ny))
        S_l = np.zeros((1,ny))
        eps_vars_Eco = np.zeros((1,ny))
        eps_weights_Eco = np.zeros((ny,ny))
        eps_vars_AbsLin = np.zeros((1,ny))
        eps_weights_AbsLin = np.zeros((ny,ny))
        day_hours = np.array([0,24])
        y_ref_day = np.zeros((1,ny))
        y_ref_night = np.zeros((1,ny))
        for objective in self.nlp.objectives:
            if isinstance(objective.feature, Controlled):
                if isinstance(objective.feature.mode, Economic):
                    for i,y in enumerate(self.state_space_joined.SS_y):
                        if objective.feature.source.col_name == (y.source.col_name if hasattr(y,'source') else y.name):
                            eps_vars_Eco[0,i] = 1
                            eps_weights_Eco[i,i] = objective.cost.weight
                            for controlled in self.nlp.model.controlled:
                                print(f"{controlled.source.col_name} == {objective.feature.source.col_name}")
                                if controlled.source.col_name == objective.feature.source.col_name:
                                    print(controlled.source.name)
                                    y_lb_day[i] = controlled.mode.day_lb
                                    y_ub_day[i] = controlled.mode.day_ub
                                    y_lb_night[i] = controlled.mode.night_lb
                                    y_ub_night[i] =controlled.mode.night_ub
                                    day_hours = np.array([controlled.mode.day_start, controlled.mode.day_end])
                elif isinstance(objective.feature.mode, Steady):
                    for i,y in enumerate(self.state_space_joined.SS_y):
                        if objective.feature.source.col_name == (y.source.col_name if hasattr(y,'source') else y.name):
                            if isinstance(objective.cost, Quadratic):
                                S_q[i,i] = objective.cost.weight
                            elif isinstance(objective.cost, AbsoluteLinear):
                                # S_l[0,i] = objective.cost.weight
                                eps_vars_AbsLin[0,i] = 1
                                eps_weights_AbsLin[i,i] = objective.cost.weight
                                y_ref_day[0,i] = objective.feature.mode.day_target
                                y_ref_night[0,i] = objective.feature.mode.night_target
                else:
                    raise NotImplementedError(f'Mode {objective.feature.mode} is not implemented yet '
                                                f'for Objective {objective}.')
            elif isinstance(objective.cost, AbsoluteLinear):
                for i,y in enumerate(self.state_space_joined.SS_y):
                    if objective.feature.source.col_name == (y.source.col_name if hasattr(y,'source') else y.name):
                        eps_vars_AbsLin[0,i] = 1
                        eps_weights_AbsLin[i,i] = objective.cost.weight
            elif isinstance(objective.cost, Quadratic):
                S_q[i,i] = objective.cost.weight
            else:
                raise NotImplementedError(f'Mode {objective.feature.mode} is not implemented yet '
                                            f'for Objective {objective}.')


        # Sent the variables to the workspace of the matlab engine
        self.eng.workspace['S_q'] = S_q
        self.eng.workspace['S_l'] = S_l
        self.eng.workspace['eps_vars_Eco'] = eps_vars_Eco
        self.eng.workspace['eps_weights_Eco'] = eps_weights_Eco
        self.eng.workspace['y_lb_day'] = y_lb_day
        self.eng.workspace['y_ub_day'] = y_ub_day
        self.eng.workspace['y_lb_night'] = y_lb_night
        self.eng.workspace['y_ub_night'] = y_ub_night
        self.eng.workspace['day_hours'] = day_hours



                

        


    def __str__(self):
        return f'ModelPredictive()'

    def __call__(self, past: pd.DataFrame) -> tuple[dict, dict]:

        if len(past) <= self.nlp.max_lag:
            return {}, {}

        current_time = past['time'].iloc[-1]

        # get the forecast and past data
        forecast = self._forecast_callback(horizon_in_seconds=int(self.nlp.N*self.step_size))

        # solve the nlp
        self.par_vals: list[float] = self._get_par_vals(past, forecast, current_time)
        
        self.par_ids = self._get_par_ids(past, forecast, current_time)
        
        
        # Here call the matlab function to solve the MPC. UPDATE TO STATE_SPACE_JOINED

        x_pre,u_pre,d_pre_fut = par_vals2SSvectors(par_vals = self.par_vals, par_ids = self.par_ids, state_space = self.state_spaces[0])
        
        self.eng.workspace['x_pre'] = x_pre
        self.eng.workspace['u_pre'] = u_pre
        self.eng.workspace['d_pre_fut'] = d_pre_fut
        self.eng.run('mpc_matlab.m', nargout=0)


        solution: NLPSolution = self.nlp.solve(self.par_vals)

        
        # retrieve the optimal controls
        controls: dict[str, float] = solution.optimal_controls

        additional_info: dict[str, float] = {'success': solution.success, 'runtime': solution.runtime}

        # append the solution to the solutions and save them to the disc
        self._save_solution(solution.df, current_time)

        # plot the solution
        self._plot_solution(solution.df, current_time)

        return controls, additional_info

    def _plot_solution(self, df: pd.DataFrame, current_time: int):

        if self._solution_plotter is None:
            return

        # add the time column to the DataFrame
        df['time'] = current_time + df.index * self.step_size

        self._solution_plotter.plot(
            df,
            save_plot=self.save_solution_plot,
            show_plot=self.show_solution_plot,
            current_time=current_time,
            filepath=file_manager.plot_filepath(name='mpc_solution', sub_folder='solutions', include_time=True)
        )

    def _get_par_vals(self, past: pd.DataFrame, forecast: pd.DataFrame, current_time: int) -> list[float]:
        """ calculates the input list for the nlp """

        par_vars = list()

        # iterate over all par vars
        for nlp_var in self.nlp._par_vars:

            t = current_time + self.step_size * nlp_var.k

            # if k <= 0 use the past DataFrame
            if nlp_var.k <= 0:
                value = past.loc[past['time'] == t, nlp_var.col_name].values

                if len(value) != 1:
                    print(f'Error occurred while getting par var {nlp_var}')
                    print('k =', nlp_var.k)
                    print('time =', int(t), datetime.datetime.fromtimestamp(t))
                    print('current_time=', int(current_time), datetime.datetime.fromtimestamp(current_time))
                    print(nlp_var.col_name)

                    past['t'] = past['time'].apply(func=datetime.datetime.fromtimestamp)
                    pd.set_option('display.float_format', lambda x: '%.2f' % x)

                    print(past.tail(n=self.nlp.max_lag).to_string())

                    raise ValueError('Error occurred, while getting par vars')

            # if k > 0 use the forecast DataFrame
            else:

                if nlp_var.col_name not in forecast.columns:
                    forecast = nlp_var.feature.source.process(forecast)

                try:
                    value = forecast.loc[forecast['time'] == t, nlp_var.col_name].values
                    assert len(value) == 1,\
                        f'{nlp_var} with col_name={nlp_var.col_name} at t={t} was not found in: \n {forecast.to_string()}'

                except KeyError:
                    raise KeyError(f'{nlp_var} with col_name={nlp_var.col_name} was not found in {forecast.columns}.')

            assert len(value) == 1
            assert value[0] is not None
            assert value[0] != np.nan, f'Detected nan for {nlp_var}'

            par_vars.append(float(value))

        return par_vars
    
    def _get_par_ids(self, past: pd.DataFrame, forecast: pd.DataFrame, current_time: int) -> list[float]:
        """ calculates the input list for the nlp """

        par_vars = list()
        par_ids = list()

        # iterate over all par vars
        for nlp_var in self.nlp._par_vars:

            t = current_time + self.step_size * nlp_var.k

            # if k <= 0 use the past DataFrame
            if nlp_var.k <= 0:
                value = past.loc[past['time'] == t, nlp_var.col_name].values
                par_id = (past.loc[past['time'] == t, nlp_var.col_name].name , nlp_var.k)

                if len(value) != 1:
                    print(f'Error occurred while getting par var {nlp_var}')
                    print('k =', nlp_var.k)
                    print('time =', int(t), datetime.datetime.fromtimestamp(t))
                    print('current_time=', int(current_time), datetime.datetime.fromtimestamp(current_time))
                    print(nlp_var.col_name)

                    past['t'] = past['time'].apply(func=datetime.datetime.fromtimestamp)
                    pd.set_option('display.float_format', lambda x: '%.2f' % x)

                    print(past.tail(n=self.nlp.max_lag).to_string())

                    raise ValueError('Error occurred, while getting par vars')

            # if k > 0 use the forecast DataFrame
            else:

                if nlp_var.col_name not in forecast.columns:
                    forecast = nlp_var.feature.source.process(forecast)

                try:
                    value = forecast.loc[forecast['time'] == t, nlp_var.col_name].values
                    par_id = (forecast.loc[forecast['time'] == t, nlp_var.col_name].name , nlp_var.k)
                    assert len(value) == 1,\
                        f'{nlp_var} with col_name={nlp_var.col_name} at t={t} was not found in: \n {forecast.to_string()}'

                except KeyError:
                    raise KeyError(f'{nlp_var} with col_name={nlp_var.col_name} was not found in {forecast.columns}.')

            assert len(value) == 1
            assert value[0] is not None
            assert value[0] != np.nan, f'Detected nan for {nlp_var}'

            par_vars.append(float(value))
            par_ids.append(par_id)

        return par_ids

    def _save_solution(self, df: pd.DataFrame, current_time: int):

        if not self.save_solution_data:
            return

        filename: str = 'solutions'
        directory: str = file_manager.data_dir()

        # read old solutions
        try:
            solutions = read_pkl(filename=filename, directory=directory)
        except (FileNotFoundError, EOFError):
            solutions = dict()

        # add the SimTime column to the DataFrame
        df['time'] = current_time + df.index * self.step_size
        solutions[current_time] = df

        # save solutions
        write_pkl(solutions, filename=filename, directory=directory, override=True)

