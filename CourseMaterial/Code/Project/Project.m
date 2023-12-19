%% Clean up and add path
clear; clc;
close all;
addpath('../functions', '../data')
load proj23.mat
%% Plot
figure()
plot(ElGeneina.rain_org_t, ElGeneina.rain_org)
hold on
plot(ElGeneina.rain_org_t, ElGeneina.rain_org, '*', 'Color', 'r')
%% Add missing values and scale down by 3
rain = ElGeneina.rain_org;
rain = rain./3;
rain_with_missing = zeros(length(rain) + 2.*(length(rain) - 1), 1); % Might be a bit pedantic, but should be this
                                
counter = 1;
for i=1:3:length(rain_with_missing)
    rain_with_missing(i) = rain(counter);
    if counter < length(rain)
        rain_with_missing(i+1) = NaN; 
        rain_with_missing(i+2) = NaN;
    end
    counter = counter + 1;
end
noVal = find(isnan(rain_with_missing));

%% Lets form the one-step prediction using the Kalman filter.
% Construct a Kalman filter that assumes the model parameters to be known.
% Note how this differs from the setup in code20, where the parameters
% where instead treated as unknown.
y0 = rain_with_missing;
y = rain_with_missing;
y1 = y;
N = length(rain_with_missing);
tt = ElGeneina.rain_t(1:end-2);
figure(); plot(tt, y1, '*')
p0 = 1;                                         % Number of unknowns in the A polynomial.
q0 = 0;                                         % Number of unknowns in the C polynomial.

A     = eye(p0+q0);
Rw    = 1;                                      % Measurement noise covariance matrix, R_w. Note that Rw has the same dimension as Ry.
Re    = 1e-6*eye(p0+q0);                        % System noise covariance matrix, R_e. Note that Re has the same dimension as Rx_t1.
Rx_t1 = eye(p0+q0);                             % Initial covariance matrix, R_{1|0}^{x,x}
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = zeros(p0+q0,N);                         % Estimated states. Intial state, x_{1|0} = 0.
yhat  = zeros(N,1);                             % Estimated output.
xStd  = zeros(p0+q0,N);                         % Stores one std for the one-step prediction.
for t=2:N                                       % We use t-3, so start at t=4.
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    C    = [ -y1(t-1)];    % Use earlier prediction errors as estimate of e_t.
    
    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    yhat(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.

    % If a sample is missing, just retain the earlier state.
    if isnan( y(t) )
        xt(:,t) = x_t1;                         % x_{t|t} = x_{t|t-1} 
        Rx_t    = Rx_t1;                        % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} 
        y1(t)   = yhat(t);                      % Replace the missing sample with the estimated value. 
    else
        h_et(t) = y(t)-yhat(t);                 % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
        xt(:,t) = x_t1 + Kt*( h_et(t) );        % x_{t|t} = x_{t|t-1} + K_t ( y_t -  \hat{y}_{t|t-1} ) 
        Rx_t    = Rx_t1 - Kt*Ry*Kt';            % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    end
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rx_t) );
end


% Show the one-step prediction. We compare with y, not y1, as the former
% retains the missing samples (this was the reason to use the y1 vector);
% this is just to illustrate the missing samples in the figure.
figure
plot(tt, [y yhat] )
xlabel('Days')
if sum(isnan(y))
    hold on
    plot( tt(noVal), y0(noVal), 'b*')
    hold off
    legend('Realisation', 'Kalman estimate', 'Missing sample', 'Location','SW')
    title('One-step prediction using the Kalman filter with missing samples')
else 
    legend('Realisation', 'Kalman estimate', 'Location','SW')
    title('One-step prediction using the Kalman filter')
end
%%
counter = 1;
sums = zeros(480,1);
sums(1) = y1(1);
counter = 2;
for i=4:3:length(y1)
    sums(counter) = y1(i) + y1(i-1) + y1(i-2);
    counter = counter + 1;
end
