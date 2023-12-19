%
% Time series analysis
% Lund University
%
% Example code 27: Predicting a BJ model using the Kalman filter (see also
% code15). 
% 
% Note: This is an IMPORTANT example!
%
% Note that Matlab uses a slightly different notation for the BJ model as
% compared to that used in the course. In Matlab's notation:
%
%   A(z) y(t) = [B(z)/F(z)] u(t) + [C(z)/D(z)] e(t)
%
% This means that:
%   A(z) = 1,       B(z) = B(z),    F(z) = A2(z)
%   C(z) = C1(z),   D(z) = A1(z)
%
% Reference: 
%   "An Introduction to Time Series Modeling", 4th ed, by Andreas Jakobsson
%   Studentlitteratur, 2021
%
clear; clc;
close all;
addpath('../functions', '../data')              % Add this line to update the path
rng(1)                                          % Set the seed (just done for the lecture!)

k = 1;                                          % This is a one-step prediction.
extraN   = 1000;                                % Lets check again why we do this; try removing this.
N        = 1300;
noLags   = 30;
modelLim = 1000;                                % Determine where the validation data starts.

% Simulate some process.
sX = 24;
A1 = [ 1 -0.9 0.82 ];
C1 = [ 1 0 -0.8 ];
A3 = conv( [ 1 zeros(1,sX-1) -1 ], [ 1 -0.4 ] );
C3 = [ 1 0.8 2.1 ];   
B  = [ 1.2 ];
A2 = 1;

z = filter( C1, A1, randn(N+extraN, 1) );       % This is the noise model.
x = filter( C3, A3, randn(N+extraN, 1) );       % This is the input signal.
y = filter( B, A2, x ) + z;
y = y(extraN+1:end);
x = x(extraN+1:end);
xM = x(1:modelLim);                             % Extract model data.
yM = y(1:modelLim);

% Examine the data.
figure; 
subplot(211); 
plot( x );
line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' )
axis([1 N min(x)*1.8 max(x)*1.8])
ylabel('Input signal')
title('Measured signals')
legend('Input signal', 'Prediction starts','Location','SW')
subplot(212); 
plot( y ); 
line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' )
axis([1 N min(y)*1.5 max(y)*1.5])
ylabel('Output signal')
xlabel('Time')
legend('Output signal', 'Prediction starts','Location','SW')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create the BJ model.
% For simplicity, we use the true model orders (see also code15).
inputModel = estimateARMA( xM, A3, C3, 'Input model', noLags, 1 );
foundModel = estimateBJ( yM, xM, C1, A1, B, A2, 'BJ model', noLags, 1 );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Predict the output using the found model.
[Fx, Gx] = polydiv( inputModel.C, inputModel.A, k );
xhatk = filter(Gx, inputModel.C, x);
 
KA = conv( foundModel.D, foundModel.F );
KB = conv( foundModel.D, foundModel.B );
KC = conv( foundModel.F, foundModel.C );
[Fy, Gy]   = polydiv( foundModel.C, foundModel.D, k );
[Fhh, Ghh] = polydiv( conv(Fy, KB), KC, k );
yhatP = filter(Fhh, 1, xhatk) + filter(Ghh, KC, x) + filter(Gy, KC, y);
eP    = y(modelLim:end)-yhatP(modelLim:end);    % Form the prediction residuals for the validation data.


%% Estimate the unknown parameters using a Kalman filter and form the one-step prediction.
% The ARMAX model is  
%
%   KA y(t) = KB x(t) + KC e(t)
%
% This means, for our example, that the one-step prediction is formed as
%
% y(t+1) = -KA(2)y(t) - KA(3)y(t) + KB(1)x(t+1) + KB(2)x(t) + KB(3)x(t-1) + e(t+1) + KC(3)e(t-1)
%        =  [ y(t) y(t-1) x(t+1) x(t) x(t-1) e(t-1) ] [ -KA(2) -KA(3) KB(1) KB(2) KB(3) KC(3) ]^T + e(t+1)
%
% Note that Matlab vectors starts at index 1, therefore the first index in
% the A vector, A(1), is the same as we normally denote a_{0}. Furthermore,
% note that both x(t) are x(t-1) known, whereas x(t+1) needs to be
% predicted. For simplicity, we here use the polynomial prediction of the
% input, but this should of course also be predicted using a Kalman filter.
% 
% For illustration purposes, we consider three different cases; in the first
% version, we estimate the parameters of the input; in the second, we 
% assume these to be fixed. In the third case, we modify the second case
% and examine if we can remove the KC parameter without losing too much
% performance. 
%
codeVersion = 1;
switch codeVersion
    case 1
        noPar   = 6;                            % The vector of unknowns is [ -KA(2) -KA(3) KB(1) KB(2) KB(3) KC(3) ]
        xt      = zeros(noPar,N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,2) = [ -KA(2) -KA(3) KB(1) KB(2) KB(3) KC(3) ];
    case 2
        noPar   = 3;                            % The vector of unknowns is [ -KA(2) -KA(3) KC(3) ]
        xt      = zeros(noPar,N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,2) = [ -KA(2) -KA(3) KC(3) ];
    case 3
        noPar   = 2;                            % The vector of unknowns is [ -KA(2) -KA(3)  ]
        xt      = zeros(noPar,N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,2) = [ -KA(2) -KA(3) ];
end

A     = eye(noPar);
Rw    = std(eP);                                % Measurement noise covariance matrix, R_w. Try using the noise estimate from the polynomial prediction.
Re    = 1e-6*eye(noPar);                        % System noise covariance matrix, R_e.
Rx_t1 = 1e-4*eye(noPar);                        % Initial covariance matrix, R_{1|0}^{x,x}
Rx_k  = Rx_t1;
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
yhatK = zeros(N,1);                             % Estimated output.
xStd  = zeros(noPar,N);                         % Stores one std for the one-step prediction.
startInd = 3;                                   % We use t-2, so start at t=3.
for t=startInd:N
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    switch codeVersion
        case 1                                  % Estimate all parameters.
            C = [ y(t-1) y(t-2) xhatk(t) x(t-1) x(t-2) h_et(t-2) ];
            yhatK(t) = C*x_t1;
        case 2                                  % Note that KB does not vary in this case.
            C = [ y(t-1) y(t-2) h_et(t-2) ];
            yhatK(t) = C*x_t1 + KB * [xhatk(t) x(t-1) x(t-2)]';
        case 3
            C = [ y(t-1) y(t-2) ];              % Ignore one component.
            yhatK(t) = C*x_t1 + KB * [xhatk(t) x(t-1) x(t-2)]';
    end

    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    h_et(t) = y(t)-yhatK(t);                    % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rx_t) );             % This is one std for each of the parameters for the one-step prediction.
end



%% Examine the estimated parameters.
% Compute the true parameters for the KA, KB, and KC polynomials.
KA0 = conv( A1, A2 );
KB0 = conv( A1, B );
KC0 = conv( A2, C1 );
switch codeVersion
    case 1
        trueParams = [ -KA0(2) -KA0(3) KB0(1) KB0(2) KB0(3) KC0(3) ];
    case 2
        trueParams = [ -KA0(2) -KA0(3) KC0(3) ];
    case 3
        trueParams = [ -KA0(2) -KA0(3) ];
end

figure
plotWithConf( 1:N, xt', xStd', trueParams );
line( [modelLim modelLim], [-2 2], 'Color','red','LineStyle',':' )
axis([startInd-1 N -1.5 1.5])
title(sprintf('Estimated parameters, with Re = %7.6f and Rw = %4.3f', Re(1,1), Rw(1,1)))
xlabel('Time')
fprintf('Using code version %i:\n', codeVersion);
fprintf('The final values of the Kalman estimated parameters are:\n')
for k0=1:length(trueParams)
    fprintf('  True value: %5.2f, estimated value: %5.2f (+/- %5.4f).\n', trueParams(k0), xt(k0,end), xStd(k0,end) )
end 


%% Show the one-step prediction. 
figure
plot( [y yhatK yhatP ] )
title('One-step prediction of the validation data')
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Polynomial estimate', 'Location','SW')
xlim([modelLim N])

% Form the prediction residuals for the validation data.
eK = y(modelLim:end)-yhatK(modelLim:end);
plotACFnPACF( eP, 40, 'One-step prediction using the polynomial estimate');
plotACFnPACF( eK, 40, 'One-step prediction using the Kalman filter');
fprintf('The variance of the validation data is               %7.2f.\n', var(y(modelLim:end)))
fprintf('The estimated variance of the polynomial estimate is %7.2f.\n', var(eP))
fprintf('The estimated variance of the Kalman estimate is     %7.2f.\n', var(eK))
