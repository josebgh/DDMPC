
x0 = cell2mat(x0)';
d_full_aux = zeros(nd,N);
for i=1:N
    d_full_aux(:,i)=cell2mat(d_full{i})';
end
d_full = d_full_aux;
y_ref = repmat(y_ref_night',[1,N]);
y_lb = repmat(y_lb_night,[1,N]);
y_ub = repmat(y_ub_night,[1,N]);

mpcsolve = MPCcontrol({x0,d_full,y_ref,y_lb,y_ub});
u0 = mpcsolve{2}(:,1);