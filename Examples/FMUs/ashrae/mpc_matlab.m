
x0 = cell2mat(x0)';
d_full_aux = zeros(nd,N);
for i=1:N
    d_full_aux(:,i)=cell2mat(d_full{i})';
end
d_full = d_full_aux;

% current_time and T is in seconds.
time_samples = (current_time:T:current_time+(N-1)*T) + 1*60*60 - 24*60*60;
time_samples_daily = mod(time_samples,full_day);
time_samples_weekly = mod(time_samples,7*full_day);
isDay = time_samples_daily>=day_seconds(1) & time_samples_daily<=day_seconds(2);
isWeekday = time_samples_weekly>=0 & time_samples_weekly<=5*full_day;

isDay_ref = (repmat(y_ref_day,[1,N])==Inf | repmat(y_ref_day,[1,N])==-Inf) | (isDay & isWeekday);
isNight_ref = (repmat(y_ref_night,[1,N])==Inf | repmat(y_ref_night,[1,N])==-Inf) | (~isDay | ~isWeekday);
y_ref = repmat(y_ref_day,[1,N]).*isDay_ref + repmat(y_ref_night,[1,N]).*isDay_ref;
isDay_lb = (repmat(y_lb_day,[1,N])==Inf | repmat(y_lb_day,[1,N])==-Inf) | (isDay & isWeekday);
isNight_lb = (repmat(y_lb_day,[1,N])==Inf | repmat(y_lb_day,[1,N])==-Inf) | (~isDay | ~isWeekday);
y_lb = repmat(y_lb_day,[1,N]).*isDay_lb + repmat(y_lb_night,[1,N]).*isNight_lb;
isDay_ub = (repmat(y_ub_day,[1,N])==Inf | repmat(y_ub_day,[1,N])==-Inf) | (isDay & isWeekday);
isNight_ub = (repmat(y_ub_day,[1,N])==Inf | repmat(y_ub_day,[1,N])==-Inf) | (~isDay | ~isWeekday);
y_ub = repmat(y_ub_day,[1,N]).*isDay_ub + repmat(y_ub_night,[1,N]).*isNight_ub;

mpcsolve = MPCcontrol({x0,d_full,y_ref,y_lb,y_ub});
u0 = mpcsolve{2}(:,1);