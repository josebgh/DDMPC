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
        self.state_spaces = state_spaces.copy()

        self.state_space_joined = StateSpace_ABCDE()
        self.state_space_joined = copy.deepcopy(self.state_spaces[0])
        self.state_spaces.pop(0)
        
        for (n,state_space) in enumerate(self.state_spaces):

            state_space = copy.deepcopy(state_space)
            state_space_joined_previous = copy.deepcopy(self.state_space_joined)

            # Obtain all x features
            for rm,f in zip(state_space.rm_1st_lag_SS_x,state_space.SS_x):
                not_in = True
                for i,f_joined in enumerate(self.state_space_joined.SS_x):
                    if f.name == f_joined.name:
                        self.state_space_joined.SS_x[i].lag = max(f_joined.lag, f.lag)
                        not_in = False
                if not_in:
                    self.state_space_joined.add_x(input=f,rm_1st_lag=rm)


            # Obtain all u features
            for f in state_space.SS_u:
                not_in = True
                for i,f_joined in enumerate(self.state_space_joined.SS_u):
                    if f.name == f_joined.name:
                        self.state_space_joined.SS_u[i].lag = max(f_joined.lag, f.lag)
                        not_in = False
                if not_in:
                    self.state_space_joined.add_u(input=f)

            # Obtain all d features
            for f in state_space.SS_d:
                not_in = True
                for i,f_joined in enumerate(self.state_space_joined.SS_d):
                    if f.name == f_joined.name:
                        self.state_space_joined.SS_d[i].lag = max(f_joined.lag, f.lag)
                        not_in = False
                if not_in:
                    self.state_space_joined.add_d(input=f)

            # Obtain all y features
            for f,y_offset in zip(state_space.SS_y,state_space.y_offset):
                not_in = True
                for i,f_joined in enumerate(self.state_space_joined.SS_y):
                    if f.name == f_joined.name:
                        self.state_space_joined.SS_y[i].lag = max(f_joined.lag, f.lag)
                        not_in = False
                if not_in:
                    self.state_space_joined.add_y(input=f)
                    self.state_space_joined.set_y_offset(y_offset=y_offset,pos=np.inf)
            
            # Recalculate the state space matrices
            extended_x_joined = self.state_space_joined.get_extended_vector(vector=self.state_space_joined.SS_x,rm_1st_lag=self.state_space_joined.rm_1st_lag_SS_x)
            extended_x_previous = state_space_joined_previous.get_extended_vector(vector=state_space_joined_previous.SS_x,rm_1st_lag=state_space_joined_previous.rm_1st_lag_SS_x)
            extended_x = state_space.get_extended_vector(vector=state_space.SS_x,rm_1st_lag=state_space.rm_1st_lag_SS_x)
            extended_y_joined = self.state_space_joined.get_extended_vector(vector=self.state_space_joined.SS_y)
            extended_y_previous = state_space_joined_previous.get_extended_vector(vector=state_space_joined_previous.SS_y)
            extended_y = state_space.get_extended_vector(vector=state_space.SS_y)
            extended_u_joined = self.state_space_joined.get_extended_vector(vector=self.state_space_joined.SS_u,isoutputs=True)
            extended_u_previous = state_space_joined_previous.get_extended_vector(vector=state_space_joined_previous.SS_u,isoutputs=True)
            extended_u = state_space.get_extended_vector(vector=state_space.SS_u,isoutputs=True)
            extended_d_joined = self.state_space_joined.get_extended_vector(vector=self.state_space_joined.SS_d)
            extended_d_previous = state_space_joined_previous.get_extended_vector(vector=state_space_joined_previous.SS_d)
            extended_d = state_space.get_extended_vector(vector=state_space.SS_d)

            nx_joined = len(extended_x_joined)
            ny_joined = len(extended_y_joined)
            nu_joined = len(extended_u_joined)
            nd_joined = len(extended_d_joined)
            A_new = np.zeros((nx_joined,nx_joined))
            C_new = np.zeros((ny_joined,nx_joined))
            B_new = np.zeros((nx_joined,nu_joined))
            D_new = np.zeros((ny_joined,nu_joined))
            Ex_new = np.zeros((nx_joined,nd_joined))
            Ey_new = np.zeros((ny_joined,nd_joined))
            x_offset_new = np.zeros([nx_joined])

            for i,x in enumerate(extended_x_previous):
                for m,x_joined in enumerate(extended_x_joined):
                    if x == x_joined:
                        for j,x_match in enumerate(extended_x_previous):
                            for n,x_joined_match in enumerate(extended_x_joined):
                                if x_match == x_joined_match:
                                    A_new[m,n] = state_space_joined_previous.A[i,j]
                                    x_offset_new[m] = state_space_joined_previous.x_offset[i]
                        for j,u_match in enumerate(extended_u_previous):
                            for n,u_joined_match in enumerate(extended_u_joined):
                                if u_match == u_joined_match:
                                    B_new[m,n] = state_space_joined_previous.B[i,j]
                        for j,d_match in enumerate(extended_d_previous):
                            for n,d_joined_match in enumerate(extended_d_joined):
                                if d_match == d_joined_match:
                                    Ex_new[m,n] = state_space_joined_previous.Ex[i,j]
            if state_space.model_SS_x:                        
                for i,x in enumerate(extended_x):
                    for m,x_joined in enumerate(extended_x_joined):
                        if x == x_joined:
                            for j,x_match in enumerate(extended_x):
                                for n,x_joined_match in enumerate(extended_x_joined):
                                    if x_match == x_joined_match:
                                        A_new[m,n] = state_space.A[i,j]
                                        x_offset_new[m] = state_space_joined_previous.x_offset[i]
                            for j,u_match in enumerate(extended_u):
                                for n,u_joined_match in enumerate(extended_u_joined):
                                    if u_match == u_joined_match:
                                        B_new[m,n] = state_space.B[i,j]
                            for j,d_match in enumerate(extended_d):
                                for n,d_joined_match in enumerate(extended_d_joined):
                                    if d_match == d_joined_match:
                                        Ex_new[m,n] = state_space.Ex[i,j]
            self.state_space_joined.set_A(A_new)
            self.state_space_joined.set_B(B_new)
            self.state_space_joined.set_Ex(Ex_new)
            self.state_space_joined.x_offset = x_offset_new

            for i,y in enumerate(extended_y_previous):
                for m,y_joined in enumerate(extended_y_joined):
                    if y == y_joined:
                        for j,x_match in enumerate(extended_x_previous):
                            for n,x_joined_match in enumerate(extended_x_joined):
                                if x_match == x_joined_match:
                                    C_new[m,n] = state_space_joined_previous.C[i,j]
                        for j,u_match in enumerate(extended_u_previous):
                            for n,u_joined_match in enumerate(extended_u_joined):
                                if u_match == u_joined_match:
                                    D_new[m,n] = state_space_joined_previous.D[i,j]
                        for j,d_match in enumerate(extended_d_previous):
                            for n,d_joined_match in enumerate(extended_d_joined):
                                if d_match == d_joined_match:
                                    Ey_new[m,n] = state_space_joined_previous.Ey[i,j]
            for i,y in enumerate(extended_y):
                for m,y_joined in enumerate(extended_y_joined):
                    if y == y_joined:
                        for j,x_match in enumerate(extended_x):
                            for n,x_joined_match in enumerate(extended_x_joined):
                                if x_match == x_joined_match:
                                    C_new[m,n] = state_space.C[i,j]
                        for j,u_match in enumerate(extended_u):
                            for n,u_joined_match in enumerate(extended_u_joined):
                                if u_match == u_joined_match:
                                    D_new[m,n] = state_space.D[i,j]
                        for j,d_match in enumerate(extended_d):
                            for n,d_joined_match in enumerate(extended_d_joined):
                                if d_match == d_joined_match:
                                    Ey_new[m,n] = state_space.Ey[i,j]
            self.state_space_joined.set_C(C_new)
            self.state_space_joined.set_D(D_new)
            self.state_space_joined.set_Ey(Ey_new)
            


        # Set the objective functions in the workspace
        # We will assume just linear and quadratic cost functions, so there will be _quad and _lin weighting matrices.
        # nc = sum([1 for objective in self.nlp.objectives if objective.cost.__class__.__name__ == 'Quadratic'])
        
        for objective in self.nlp.objectives:
            if objective.feature.source in [f .source if hasattr(f, 'source') else f for f in self.state_space_joined.SS_y]:
                pass
            else:
                if isinstance(objective.feature, Control):
                    # Convertir una y que apunte a la variable de control
                    for (i,u) in enumerate(self.state_space_joined.get_extended_vector(self.state_space_joined.SS_u,isoutputs=True)):
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
                            Ey_new = np.zeros((self.state_space_joined.Ey.shape[0]+1,self.state_space_joined.Ey.shape[1]))
                            Ey_new[:-1,:] = self.state_space_joined.Ey
                            self.state_space_joined.set_Ey(Ey_new)
                elif  isinstance(objective.feature, Controlled):
                    # Convertir una y que apunte a la variable de estado
                    for (i,x) in enumerate(self.state_space_joined.get_extended_vector(vector=self.state_space_joined.SS_x,rm_1st_lag=self.state_space_joined.rm_1st_lag_SS_x)):
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
                            Ey_new = np.zeros((self.state_space_joined.Ey.shape[0]+1,self.state_space_joined.Ey.shape[1]))
                            Ey_new[:-1,:] = self.state_space_joined.Ey
                            self.state_space_joined.set_Ey(Ey_new)

                elif isinstance(objective.feature, Connection):
                    y_new = copy.deepcopy(objective.feature.source)
                    self.state_space_joined.add_y(y_new)
                    self.state_space_joined.set_y_offset(0.,len(self.state_space_joined.y_offset))
                    # Convertir un y que apunte al incremento de la variable de estado o acción de control
                    for (i,x) in enumerate(self.state_space_joined.get_extended_vector(vector=self.state_space_joined.SS_x,rm_1st_lag=self.state_space_joined.rm_1st_lag_SS_x)):
                        if x == (objective.feature.source.base.col_name,0):
                            C_new = np.zeros((self.state_space_joined.C.shape[0]+1,self.state_space_joined.C.shape[1]))
                            C_new[:-1,:] = self.state_space_joined.C
                            C_new[-1,i] = 1
                            C_new[-1,i+1] = -1
                            self.state_space_joined.set_C(C_new)
                            D_new = np.zeros((self.state_space_joined.D.shape[0]+1,self.state_space_joined.D.shape[1]))
                            D_new[:-1,:] = self.state_space_joined.D
                            self.state_space_joined.set_D(D_new)
                            Ey_new = np.zeros((self.state_space_joined.Ey.shape[0]+1,self.state_space_joined.Ey.shape[1]))
                            Ey_new[:-1,:] = self.state_space_joined.Ey
                            self.state_space_joined.set_Ey(Ey_new)
                    for (i,u) in enumerate(self.state_space_joined.get_extended_vector(self.state_space_joined.SS_u,isoutputs=True)):
                        if u == (objective.feature.source.base.col_name,0):
                            C_new = np.zeros((self.state_space_joined.C.shape[0]+1,self.state_space_joined.C.shape[1]))
                            C_new[:-1,:] = self.state_space_joined.C
                            self.state_space_joined.set_C(C_new)
                            D_new = np.zeros((self.state_space_joined.D.shape[0]+1,self.state_space_joined.D.shape[1]))
                            D_new[:-1,:] = self.state_space_joined.D
                            D_new[-1,i] = 1
                            u_prev = (u[0],1)
                            pos_prev_u = self.state_space_joined.get_extended_vector(self.state_space_joined.SS_x,rm_1st_lag=self.state_space_joined.rm_1st_lag_SS_x).index(u_prev)
                            C_new[-1,pos_prev_u] = -1
                            self.state_space_joined.set_D(D_new)
                            Ey_new = np.zeros((self.state_space_joined.Ey.shape[0]+1,self.state_space_joined.Ey.shape[1]))
                            Ey_new[:-1,:] = self.state_space_joined.Ey
                            self.state_space_joined.set_Ey(Ey_new)
                else:
                    raise ValueError(f"Objective {objective} feature is not Control, Controlled or Change")
                
                # auxEy = self.state_space_joined.Ey[1][0].copy()
                # auxD = self.state_space_joined.D[1][1].copy()
                # self.state_space_joined.Ey[1][0] = auxD
                # self.state_space_joined.D[1][1] = auxEy
                
        if self.eng is None:
            self.eng = matlab.engine.start_matlab()
        
        # Set the state space matrices in the workspace
        self.eng.workspace['A'] = self.state_space_joined.A
        self.eng.workspace['B'] = self.state_space_joined.B
        self.eng.workspace['C'] = self.state_space_joined.C
        self.eng.workspace['D'] = self.state_space_joined.D
        self.eng.workspace['Ex'] = self.state_space_joined.Ex
        self.eng.workspace['Ey'] = self.state_space_joined.Ey
        self.eng.workspace['y_offset'] = self.state_space_joined.y_offset
        self.eng.workspace['x_offset'] = self.state_space_joined.x_offset




        # Set the contraints in the workspace
        constraints = {constraint.feature.source.col_name: (constraint.lb, constraint.ub) for constraint in self.nlp.constraints}

        x_lb = np.zeros((self.state_space_joined.get_nx(),1))
        x_ub = np.zeros((self.state_space_joined.get_nx(),1))
        x_extended = self.state_space_joined.get_extended_vector(vector=self.state_space_joined.SS_x,rm_1st_lag=self.state_space_joined.rm_1st_lag_SS_x)
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
        u_extended = self.state_space_joined.get_extended_vector(self.state_space_joined.SS_u,isoutputs=True)
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
        eps_weights_AbsLin = np.zeros((1,ny))
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
                                if controlled.source.col_name == objective.feature.source.col_name:
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
                                eps_weights_AbsLin[0,i] = objective.cost.weight
                                y_ref_day[0,i] = objective.feature.mode.day_target
                                y_ref_night[0,i] = objective.feature.mode.night_target
                else:
                    raise NotImplementedError(f'Mode {objective.feature.mode} is not implemented yet '
                                                f'for Objective {objective}.')
            elif isinstance(objective.cost, AbsoluteLinear):
                for i,y in enumerate(self.state_space_joined.SS_y):
                    if objective.feature.source.col_name == (y.source.col_name if hasattr(y,'source') else y.name):
                        eps_vars_AbsLin[0,i] = 1
                        eps_weights_AbsLin[0,i] = objective.cost.weight
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
        self.eng.workspace['eps_vars_AbsLin'] = eps_vars_AbsLin
        self.eng.workspace['eps_weights_AbsLin'] = eps_weights_AbsLin
        self.eng.workspace['y_ref_day'] = y_ref_day
        self.eng.workspace['y_ref_night'] = y_ref_night
        self.eng.workspace['N'] = self.nlp.N

        self.flag = True
        self.eng.run('mpc_matlab_setup.m', nargout=0)


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

        # x0,u_pre,d_full = par_vals2SSvectors(par_vals = self.par_vals, par_ids = self.par_ids, state_space = self.state_space_joined)
        x0,d_full = par_vals2SSvectors(par_vals = self.par_vals, par_ids = self.par_ids, state_space = self.state_space_joined)
        self.eng.workspace['x0'] = x0
        # self.eng.workspace['u_pre'] = u_pre
        self.eng.workspace['d_full'] = d_full
        # if self.flag:
        #     self.eng.run('save_workspace.m', nargout=0)
        #     self.flag = False
        self.eng.run('mpc_matlab.m', nargout=0)
        u0 = self.eng.workspace['u0']
        print(u0)
        # mpcsolve = self.eng.workspace['mpcsolve']
        # print("mpcsolve",mpcsolve)
        # self.eng.run('save_workspace.m', nargout=0)

        solution: NLPSolution = self.nlp.solve(self.par_vals)

        
        # retrieve the optimal controls
        # controls: dict[str, float] = solution.optimal_controls
        controls: dict[str,float] = {self.state_space_joined.SS_u[0].name: u0[0][0], self.state_space_joined.SS_u[1].name: u0[1][0]}

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

