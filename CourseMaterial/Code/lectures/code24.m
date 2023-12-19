%
% Time series analysis
% Lund University
%
% Example code 24: Form a k-step prediction of an ARMA process with unknown
% parameters using the Kalman filter (see also example 8.12). 
% 
% Note: This is an IMPORTANT example!
%
% Reference: 
%   "An Introduction to Time Series Modeling", 4th ed, by Andreas Jakobsson
%   Studentlitteratur, 2021
%
clear; clc;
close all;
addpath('../functions', '../data')              % Add this line to update the path

% Simulate a process.
rng(1)                                          % Set the seed (just done for the lecture!)
extraN = 100;
N  = 1000;
A0 = conv([1 zeros(1,5) -1],[ 1 -0.7 ]);        % Try changing the polynomials - but note that you need to change the C vector then!
C0 = [1 0.5 0 -0.2];                            
e  = randn( N+extraN, 1 );
y  = filter( C0, A0, e );   y = y(extraN+1:end);    e = e(extraN+1:end);

% Plot realisation.
figure
plot(y)
title( 'Realisation of an ARMA-process' )
ylabel('Amplitude')
xlabel('Time')
xlim([1 N])


%% Estimate the unknown parameters using a Kalman filter and form the k-step prediction.
k  = 3;                                         % k-step prediction.
p0 = 3;                                         % Number of unknowns in the A polynomial (note: this is only the non-zero parameters!).
q0 = 2;                                         % Number of unknowns in the C polynomial (note: this is only the non-zero parameters!).

A     = eye(p0+q0);
Rw    = 1;                                      % Measurement noise covariance matrix, R_w. Note that Rw has the same dimension as Ry.
Re    = 1e-6*eye(p0+q0);                        % System noise covariance matrix, R_e. Note that Re has the same dimension as Rx_t1.
Rx_t1 = eye(p0+q0);                             % Initial covariance matrix, R_{1|0}^{x,x}
Rx_k  = Rx_t1;
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = zeros(p0+q0,N-k);                       % Estimated states. Intial state, x_{1|0} = 0.
yhat  = zeros(N-k,1);                           % Estimated output.
yhatk = zeros(N-k,1);                           % Estimated k-step prediction.
xStd  = zeros(p0+q0,N-k);                       % Stores one std for the one-step prediction.
xStdk = zeros(p0+q0,N-k);                       % Stores one std for the k-step prediction.
for t=8:N-k                                     % We use t-7, so start at t=8. As we form a k-step prediction, end the loop at N-k.
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    C    = [ -y(t-1) -y(t-6) -y(t-7) h_et(t-1) h_et(t-3) ];     % C_{t|t-1}
    
    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    yhat(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
    h_et(t) = y(t)-yhat(t);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Form the k-step prediction by first constructing the future C vector
    % and the one-step prediction. Note that this is not yhat(t) above, as
    % this is \hat{y}_{t|t-1}.
    Ck = [ -y(t) -y(t-5) -y(t-6) h_et(t) h_et(t-2) ];           % C_{t+1|t}
    yk = Ck*xt(:,t);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}

    % Note that the k-step predictions is formed using the k-1, k-2, ...
    % predictions, with the predicted future noises being set to zero. If
    % the ARMA has a higher order AR part, one needs to keep track of each
    % of the earlier predicted values.
    Rx_k = Rx_t1;
    for k0=2:k
        Ck = [ -yk -y(t-6+k0) -y(t-7+k0) h_et(t+k0-1) h_et(t+k0-3) ]; % C_{t+k|t}
        yk = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        Rx_k = A*Rx_k*A' + Re;                  % R_{t+k+1|t}^{x,x} = A R_{t+k|t}^{x,x} A^T + Re  
    end
    yhatk(t+k) = yk;                            % Note that this should be stored at t+k.

    % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rx_t) );             % This is one std for each of the parameters for the one-step prediction.
    xStdk(:,t) = sqrt( diag(Rx_k) );            % This is one std for each of the parameters for the k-step prediction.
end


%% Examine the estimated parameters.
% If you examine the confidence interval for the k-step prediction, you
% will find that this is very close to the one-step prediction for these
% settings (use xStdk). Why is that?
trueParams = [A0(2) A0(7) A0(8) C0(2) C0(4)];   % These are the true parameters we seek.
figure
plotWithConf( (1:N-k), xt', xStd', trueParams );
title(sprintf('Estimated parameters, with Re = %7.6f and Rw = %4.3f', Re(1,1), Rw(1,1)))
xlabel('Time')
ylim([-1.5 1.5])
fprintf('The final values of the estimated parameters are:\n')
for k0=1:length(trueParams)
    fprintf('  True value: %5.2f, estimated value: %5.2f (+/- %5.4f).\n', trueParams(k0), xt(k0,end), xStd(k0,end) )
end 


%% Show the one-step prediction. 
figure
plot( [y(1:N-k) yhat] )
title('One-step prediction using the Kalman filter')
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
e1 = y(200:N-k)-yhat(200:end);                   % Ignore the initial values to let the filter converge first.
plotACFnPACF( e1, 40, 'One-step prediction using the Kalman filter');
fprintf('Examining the one-step residual.\n')
checkIfWhite( e1 );


%% Show the k-step prediction. 
figure
plot( [y yhatk] )
title( sprintf('%i-step prediction using the Kalman filter', k) )
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Location','SW')
xlim([1 N-k])


%% Examine k-step prediction residual.
ek = y(N-200:N-k)-yhatk(N-200:N-k);             % Ignore the initial values to let the filter converge first.
plotACFnPACF( ek, 40, sprintf('%i-step prediction using the Kalman filter', k)  );

fprintf('  The variance of original signal is                %5.2f.\n', var(y)')
fprintf('  The variance of the 1-step prediction residual is %5.2f.\n', var(e1)')
fprintf('  The variance of the %i-step prediction residual is %5.2f.\n', k, var(ek)')
