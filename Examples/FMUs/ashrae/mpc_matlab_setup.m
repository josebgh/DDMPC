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

% yds = sdpvar(ny*ones(1,N+1),ones(1,N+1),'full');
% xds = sdpvar(nx*ones(1,N+1),ones(1,N+1),'full');
% uds = sdpvar(nu*ones(1,N),ones(1,N),'full');

cost = 0;
constr = [];
for k=1:N

    constr = constr + [ x{1} == x0 ];
    
    % Controlled Economic Cost
    for i=1:length(y{k})
        if eps_vars_Eco(i)==1
            cost = cost + [ eps_Eco{k}(i)' * eps_weights_Eco(i,i) * eps_Eco{k}(i) ];
            constr = constr + [ y_lb{k}(i) - eps_Eco{k}(i) <= y{k}(i) ];
            constr = constr + [ y{k}(i) <= y_ub{k}(i) + eps_Eco{k}(i) ];
        end
    end
    
    % Quadratic Cost
    cost = cost + [ ( y{k} - y_ref{k} )' * S_q * ( y{k} - y_ref{k} ) ];

    % Absolute linear Cost
    for i=1:length(y{k})
        if eps_vars_AbsLin(i)==1
            cost = cost + [ eps_weights_AbsLin(i) * eps_AbsLin_1{k}(i) ];
            cost = cost + [ eps_weights_AbsLin(i) * eps_AbsLin_2{k}(i) ];
            constr = constr + [ eps_AbsLin_1{k}(i) - eps_AbsLin_2{k}(i) == ( y{k}(i) - y_ref{k}(i) ) ];
            constr = constr + [ eps_AbsLin_1{k}(i) >= 0 ];
            constr = constr + [ eps_AbsLin_2{k}(i) >= 0 ];
        end
    end

    % Constraints (y constraints alredy in Controlled Economic cost)
    if k>1
        for i=1:length(x{k})
            constr = constr + [ x_lb(i) <= x{k}(i) ];
            % constr = constr + [ x_lb(i) <= x{k}(i) ];
            constr = constr + [ x{k}(i) <= x_ub(i) ];
        end
    end

    for i=1:length(u{k})
        constr = constr + [ u_lb(i) <= u{k}(i) ];
        constr = constr + [ u{k}(i) <= u_ub(i) ];
    end
    
    % System
    constr = constr + [ x{k+1} == A*x{k} + B*u{k} + Ex*d{k} + x_offset' ];
    constr = constr + [ y{k} == C*x{k} + D*u{k} + Ey*d{k} + y_offset'];

end
    
% Create MPC object
% options = sdpsettings('solver','quadprog','verbose',1);
options = sdpsettings('solver','gurobi','verbose',1);
% options = sdpsettings('solver','fmincon','verbose',1);
% options = sdpsettings('solver','','verbose',1);

MPCcontrol = optimizer(constr,cost,options,...
      {x0,[d{:}],[y_ref{:}],[y_lb{:}],[y_ub{:}]},...
      {[x{:}],[u{:}],cost});