from typing import Union, Callable, Optional
import copy

import numpy as np
import casadi as ca

from sklearn import linear_model

# from ddmpc.data_handling.processing_data import TrainingData
from ddmpc.modeling.process_models.machine_learning.regression.polynomial import LinearRegression
from ddmpc.modeling.modeling import Model
from ddmpc.modeling.predicting import Input,Output
from ddmpc.modeling.features import Change

class StateSpace_ABCDE:
    """StateSpace(A, B, C, D, Ex, Ey)

        Construct a state space object such that:
        x = A*x + B*u + E*d + y_offset ***set dims of y_offset to x ***
        y = C*x + D*u + E*d + y_offset

        THIS FUNCTION HAS BEING PATHED BY ADDING E*d and y_offset to the state space model equation for x. This is not
        correct and should be fixed in the future.

        It also contains the list of inputs (.SS_u), outputs (.SS_y), states (.SS_x) and disturbances (.SS_d) atrributes.
    """
    def __init__(self):
        self.A = np.zeros((0,0))
        self.B = np.zeros((0,0))
        self.C = np.zeros((0,0))
        self.D = np.zeros((0,0))
        self.Ex = np.zeros((0,0))
        self.Ey = np.zeros((0,0))
        self.y_offset = np.array([0.])
        self.x_offset = np.array([0.])
        self.SS_x = list() # list of states (Input objects)
        self.rm_1st_lag_SS_x = list() # list of bools, which indicates if the first lag of the state should be removed. (Used when the state is a control variable)
        self.SS_d = list() # list of disturbances (Input objects)
        self.SS_u = list() # list of inputs (Input objects)
        self.SS_y = list() # list of outputs (Output objects)
        self.model_SS_x = True
        

    def set_A(self, A: np.matrix):
        self.A = A

    def set_B(self, B: np.matrix):
        self.B = B

    def set_C(self, C: np.matrix):
        self.C = C

    def set_D(self, D: np.matrix):
        self.D = D

    def set_Ex(self, Ex: np.matrix):
        self.Ex = Ex

    def set_Ey(self, Ey: np.matrix):
        self.Ey = Ey

    def set_y_offset(self, y_offset: float, pos:int=0):
        if pos > len(self.y_offset)-1:
            self.y_offset = np.append(self.y_offset, y_offset)
        else:
            self.y_offset[pos] = y_offset

    def set_x_offset(self, x_offset: np.ndarray, pos:int=0):
        for i in range(len(x_offset)):
            if pos+i > len(self.x_offset)-1:
                self.x_offset = np.append(self.x_offset, x_offset[i])
            else:
                self.x_offset[pos+i] = x_offset[i]

    def add_x(self, input : Input, rm_1st_lag:bool=False):
        self.SS_x.append(input)
        self.rm_1st_lag_SS_x.append(rm_1st_lag)
        
    def add_d(self, input : Input):
        self.SS_d.append(input)
        
    def add_u(self, input : Input):
        self.SS_u.append(input)
        
    def add_y(self, input : Input):
        self.SS_y.append(input)

    def get_nx(self):
        return self.A.shape[0]
    
    def get_nu(self):
        return self.B.shape[1]
    
    def get_nd(self):
        return self.Ey.shape[1]
    
    def get_ny(self):
        return self.C.shape[0]
    
    def get_extended_vector(self, vector: list(), rm_1st_lag: list() = None, isoutputs: bool = False) -> list:
        """
        This function returns the extended vector of the variable var. Receive the own vector SS_x, SS_u, SS_d or SS_y.
        """
        # DIFERENCIAR ENTRE SI SE TRATA DE UN ESTADO (HAY QUE ELIMINAR EL LAG=1 DE LOS CONTROLS)
        if (rm_1st_lag is None) and (isoutputs is False):
            vector = [(var.name,(k)) for var in vector for k in range(var.lag if hasattr(var, 'lag') else 1)]
        elif rm_1st_lag is not None:
            vector = [(var.name,(k)) for var,rm in zip(vector,rm_1st_lag) for k in range(var.lag if hasattr(var, 'lag') else 1) if not (rm and k==0)]
        elif isoutputs:
            vector = [(var.name,(0)) for var in vector]
        return vector

def lr2ss(linear_regression: LinearRegression, model: Model) -> StateSpace_ABCDE:
    """
    This function returns a class containing the matrices of the state space form for the linear regression model, such that x=A*x+B*u+E*d, y=C*x+D*u.
    """

    SS_output = StateSpace_ABCDE()

    # if not isinstance(linear_regressions, list):
    #     linear_regressions = [linear_regressions]

    # assert len(model.controlled)==len(linear_regressions), f'Error: {model.controlled} controlled variable, but just {len(linear_regressions)} linear regression models are passed (a linear regression model should be used for each controlled variable).'
    # ny = len(model.controlled)
    
    # for linear_regression in linear_regressions:
    # DE MOMENTO LO HAGO SOLO PARA 1 TRAINING DATA, HAY QUE ACTUALIZAR.
    ny = 1
    nu = sum(1 for x in linear_regression.inputs if x.source in model.controls)
    nx = sum(x.lag for x in linear_regression.inputs if x.source in model.controlled)
    nx += sum(x.lag for x in linear_regression.inputs if x.source in model.controls) - nu
    nd = sum(x.lag for x in linear_regression.inputs) - nx - nu
    A = np.zeros((nx, nx))
    A[1:,0:-1] = np.eye(nx-1)
    # A = np.zeros((nx + nu))
    B = np.zeros((nx, nu))
    C = np.zeros((ny, nx))
    D = np.zeros((ny, nu))
    Ex = np.zeros((nx, nd))
    Ey = np.zeros((ny, nd))

    total_i = 0
    C_i = 0
    D_i = 0
    E_i = 0
    SS_output.add_y(linear_regression.output)
    SS_output.model_SS_x = ( linear_regression.output.source.col_name in [ input.source.name for input in linear_regression.inputs ] ) or ( isinstance(linear_regression.output.source,Change) and ( linear_regression.output.source.base.col_name in [ input.source.col_name for input in linear_regression.inputs ] ))
    # C AND D SHOULDN'T BE CALCULATED IN THIS WAY, BUT USING EYE MATRICES TO STATES AND OUTPUTS.
    for f in linear_regression.inputs:
        if f.source in model.controlled:
            for i in range(0, f.lag):
                coef = linear_regression.linear_model.coef_[0][total_i]
                C[0][C_i] = coef
                if SS_output.model_SS_x:
                    if isinstance(linear_regression.output.source,Change):
                        A[0][C_i] = coef + 1
                    else:
                        A[0][C_i] = coef
                C_i += 1
                total_i += 1
            SS_output.add_x(input=f,rm_1st_lag=False)
    # FALTA INCLUIR METODO PAR ASEGURAR QUE LAS VARIABLES CONTROLLED SON LAS PRIMERAS EN EL SS.
    # for f in linear_regression.inputs:
        if f.source in model.controlled:
            pass # we ensure that the controlled variables are the first ones in the state space model
        elif f.source in model.controls:
            for i in range(0, f.lag):
                coef = linear_regression.linear_model.coef_[0][total_i]
                if i==0:
                    D[0][D_i] = coef
                    if SS_output.model_SS_x:
                        B[0][D_i] = coef
                    if f.lag>1:
                        A[C_i][C_i-1] = 0 # is not 1...
                        if SS_output.model_SS_x:
                            B[C_i][D_i] = 1 # ...first state of the output states is the output, but not a previous state.
                    D_i += 1
                else:
                    C[0][C_i] = coef
                    if SS_output.model_SS_x:
                        A[0][C_i] = coef
                    C_i += 1
                total_i += 1
            SS_output.add_u(f)
            if f.lag>1:
                SS_output.add_x(input=f,rm_1st_lag=True)
        else:
            for _ in range(0, f.lag):
                coef = linear_regression.linear_model.coef_[0][total_i]
                if SS_output.model_SS_x:
                    Ex[0][E_i] = coef
                Ey[0][E_i] = coef
                E_i += 1
                total_i += 1
            SS_output.add_d(f)
    SS_output.set_A(A)
    SS_output.set_B(B)
    SS_output.set_C(C)
    SS_output.set_D(D)
    SS_output.set_Ex(Ex)
    SS_output.set_Ey(Ey)
    

    SS_output.set_y_offset(linear_regression.linear_model.intercept_)
    x_offset = np.zeros([SS_output.get_nx()])
    if SS_output.model_SS_x:
        x_offset[0] = linear_regression.linear_model.intercept_
    SS_output.set_x_offset(x_offset)

    return SS_output


def LRinputs2SSvectors(input_values: Union[list, ca.MX, ca.DM, np.ndarray], state_space: StateSpace_ABCDE, linear_regression: LinearRegression) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function receive the inputs vector and returns the state space vectors x, u and d.
    """
    nx = state_space.get_nx()
    nu = state_space.get_nu()
    ny = state_space.get_ny()
    nd = state_space.get_nd()

    if not isinstance(input_values, np.ndarray) and not isinstance(input_values, list):
        input_values = [input_values]

    if isinstance(input_values, list):
    # if True:

        x = list()
        u = list()
        d = list()

        # Could be more efficient if we iterate over the inputs of the linear regression model and the 
        # compare with the state space model, but this way is ensured that the order of the state space
        # vectors are correct in comparison with the matrices.
        # In addition, we are assuming the order of linear regression inputs is the same as the inputs values
        # something like this could be done to compare the names of the variables and get the correct order of the inputs values:
        for k,f in enumerate(state_space.SS_x):
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, f.lag):
                        if state_space.rm_1st_lag_SS_x[k] and j==0:
                            pass
                        else:
                            x.append(input_values[real_i+j])
        for f in state_space.SS_u:
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, 1):
                        u.append(input_values[real_i+j])
        for f in state_space.SS_d:
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, f.lag):
                        d.append(input_values[real_i+j])

    elif isinstance(input_values, np.ndarray):
        x = np.zeros((nx, 1))
        u = np.zeros((nu, 1))
        d = np.zeros((nd, 1))


        # Could be more efficient if we iterate over the inputs of the linear regression model and the 
        # compare with the state space model, but this way is ensured that the order of the state space
        # vectors are correct in comparison with the matrices.
        x_i = 0 # index to point x
        for f in state_space.SS_x:
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, f.lag):
                        x[x_i] = input_values[real_i+j]
                        x_i += 1
        u_i = 0 # index to point u
        for f in state_space.SS_u:
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, f.lag):
                        u[u_i] = input_values[real_i+j]
                        u_i += 1
        d_i = 0 # index to point d
        for f in state_space.SS_d:
            for i,f_aux in enumerate(linear_regression.inputs):
                if f.source == f_aux.source:
                    real_i = sum([input.lag for input in linear_regression.inputs[0:i]])
                    for j in range(0, f.lag):
                        d[d_i] = input_values[real_i+j]
                        d_i += 1

    else:
        raise ValueError("input_values has to be either a list, np.ndarray or ca.MX")

    return x, u, d

def par_vals2SSvectors(par_vals: list, par_ids: list, state_space: StateSpace_ABCDE) -> tuple[list, list, list]:
    """
    This function receive the parameters vector and returns the state space vectors x, u and d.
    """
    par_dict = dict(zip(par_ids, par_vals))
    nx = state_space.get_nx()
    nu = state_space.get_nu()
    ny = state_space.get_ny()
    nd = state_space.get_nd()

    x0 = list()
    for x in  state_space.get_extended_vector(vector=state_space.SS_x,rm_1st_lag=state_space.rm_1st_lag_SS_x):
        key = (x[0],-x[1])
        x0.append(par_dict[key])

    # u_pre = list()
    # for u in state_space.get_extended_vector(vector=state_space.SS_u,isoutputs=True):
    #     key = (u[0],-u[1])
    #     u_pre.append(par_dict[key])
        
    d_full = [0] # first element is 0 in order to first append include a list in the list and not the values.
    j=0
    f = state_space.SS_d[0]
    while (f.source.col_name, j+1) in par_dict:
        d_pre = list()
        for d in state_space.get_extended_vector(vector=state_space.SS_d):
            key = (d[0],-d[1]+j)
            d_pre.append(par_dict[key])
        j+=1
        d_full.append(d_pre)
    d_full.pop(0) # remove the first element
            
    # for f in state_space.SS_x:
    #     for i in range(-f.lag,0):
    #         key = (f.source.col_name, i+1)
    #         x0.append(par_dict[key])
            
    # u_pre = list()
    # for f in state_space.SS_u:
    #     for i in range(-f.lag,-1):
    #         key = (f.source.col_name, i+1)
    #         u_pre.append(par_dict[key])
    
    # d_full = [0] # first element is 0 in order to first append include a list in the list and not the values.
    # j=0
    # f = state_space.SS_d[0]
    # while (f.source.col_name, j+1) in par_dict:
    #     d = list()
    #     for f in state_space.SS_d:
    #         for i in range(-f.lag,0):
    #             # print("f.lag",f.lag)
    #             key = (f.source.col_name, i+1+j)
    #             d.append(par_dict[key])
    #     j+=1
    #     d_full.append(d)
    # d_full.pop(0) # remove the first element
    
    # return x0, u_pre, d_full
    return x0, d_full