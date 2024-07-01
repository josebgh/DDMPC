%% MPC YALMIP BUILDING

yalmip('clear')

% auxiliar variables
full_day = 24*60*60;
day_seconds = day_hours*60*60;
% states dimensions
nx = size(A,1);
nu = size(B,2);
nd = size(Ex,2);
ny = size(C,1);

% dynamic variables
x0 = sdpvar(nx,1,'full');
d_full = sdpvar(nd*ones(1,N),ones(1,N),'full');

x = sdpvar(nx*ones(1,N+1),ones(1,N+1),'full');
u = sdpvar(nu*ones(1,N),ones(1,N),'full');
d = sdpvar(nd*ones(1,N),ones(1,N),'full');
y = sdpvar(ny*ones(1,N),ones(1,N),'full');

% vars for objectives
eps_Eco = sdpvar(ny*ones(1,N),ones(1,N),'full');
eps_AbsLin_1 = sdpvar(ny*ones(1,N),ones(1,N),'full');
eps_AbsLin_2 = sdpvar(ny*ones(1,N),ones(1,N),'full');

y_ref = sdpvar(ny*ones(1,N),ones(1,N),'full');
y_lb = sdpvar(ny*ones(1,N),ones(1,N),'full');
y_ub = sdpvar(ny*ones(1,N),ones(1,N),'full');

% artificial reference variables
xss = sdpvar(nx*ones(1,N+1),ones(1,N+1),'full');
uss = sdpvar(nu*ones(1,N),ones(1,N),'full');
d = sdpvar(nd*ones(1,N),ones(1,N),'full');
yss = sdpvar(ny*ones(1,N),ones(1,N),'full');

% tracking cost weighting matrices
Q = 100*eye(nx);
R = eye(nu);

cost = 0;
constr = [];
for k=1:N

    constr = constr + [ x{1} == x0 ];
    
    % Controlled Economic Cost
    for i=1:length(yss{k})
        if eps_vars_Eco(i)==1
            cost = cost + [ eps_Eco{k}(i)' * eps_weights_Eco(i,i) * eps_Eco{k}(i) ];
            constr = constr + [ y_lb{k}(i) - eps_Eco{k}(i) <= yss{k}(i) ];
            constr = constr + [ yss{k}(i) <= y_ub{k}(i) + eps_Eco{k}(i) ];
        end
    end
    
    % Quadratic Cost
    cost = cost + [ ( yss{k} - y_ref{k} )' * S_q * ( yss{k} - y_ref{k} ) ];

    % Absolute linear Cost
    for i=1:length(yss{k})
        if eps_vars_AbsLin(i)==1
            cost = cost + [ eps_weights_AbsLin(i) * eps_AbsLin_1{k}(i) ];
            cost = cost + [ eps_weights_AbsLin(i) * eps_AbsLin_2{k}(i) ];
            constr = constr + [ eps_AbsLin_1{k}(i) - eps_AbsLin_2{k}(i) == ( yss{k}(i) - y_ref{k}(i) ) ];
            constr = constr + [ eps_AbsLin_1{k}(i) >= 0 ];
            constr = constr + [ eps_AbsLin_2{k}(i) >= 0 ];
        end
    end

    % Tracking cost
    cost = cost + [ ( x{k} - xss{k} )' * Q * ( x{k} - xss{k} ) ];
    cost = cost + [ ( u{k} - uss{k} )' * R * ( u{k} - uss{k} ) ];

    % Constraints (y constraints alredy in Controlled Economic cost)
    for i=1:length(x{k})
        if k>1
            constr = constr + [ x_lb(i) <= x{k}(i) ];
            constr = constr + [ x{k}(i) <= x_ub(i) ];
        end

        constr = constr + [ x_lb(i) <= xss{k}(i) ];
        constr = constr + [ xss{k}(i) <= x_ub(i) ];
    end

    for i=1:length(u{k})
        constr = constr + [ u_lb(i) <= u{k}(i) ];
        constr = constr + [ u{k}(i) <= u_ub(i) ];

        constr = constr + [ u_lb(i) <= uss{k}(i) ];
        constr = constr + [ uss{k}(i) <= u_ub(i) ];
    end
    
    % System
    constr = constr + [ x{k+1} == A*x{k} + B*u{k} + Ex*d{k} + x_offset' ];
    constr = constr + [ y{k} == C*x{k} + D*u{k} + Ey*d{k} + y_offset'];

    constr = constr + [ xss{k+1} == A*xss{k} + B*uss{k} + Ex*d{k} + x_offset' ];
    constr = constr + [ yss{k} == C*xss{k} + D*uss{k} + Ey*d{k} + y_offset'];

end
    constr = constr + [ x{N+1} == xss{N+1} ];
    
    constr = constr + [ xss{1} == A*xss{N} + B*uss{N} + Ex*d{N} + x_offset' ];
    
% Create MPC object
% options = sdpsettings('solver','quadprog','verbose',1);
options = sdpsettings('solver','gurobi','verbose',1);
% options = sdpsettings('solver','fmincon','verbose',1);
% options = sdpsettings('solver','','verbose',1);

MPCcontrol = optimizer(constr,cost,options,...
      {x0,[d{:}],[y_ref{:}],[y_lb{:}],[y_ub{:}]},...
      {[x{:}],[u{:}],cost});