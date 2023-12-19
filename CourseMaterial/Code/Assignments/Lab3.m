%% Clean up and add path
clear; clc;
close all;
addpath('../functions', '../data')
%% 2.1
load tar2.dat
load thx.dat

% Plot Data and true AR coefficients 
figure()
subplot(311)
plot(tar2)
legend('Data')

subplot(312)
plot(thx(:,1))
legend('a_1')

subplot(313)
plot(thx(:,2))
legend('a_2')

% Estimate Params us RLS with forgetting factor

lambda = [1 0.95 0.9 0.9424]; % 0.9424 From code below
for i=1:length(lambda)
    X = recursiveAR(2);
    X.ForgettingFactor = lambda(i);
    X.InitialA = [1 0 0] ;
    for kk=1:length(tar2)
        [Aest(kk, :), yhat(kk)] = step(X, tar2(kk));
    end
    
    figure()
    subplot(211)
    plot(thx(:,1))
    hold on
    plot(Aest(:,2))
    legend('True a_1', 'Estimated a_1')
    title(append('Forgetting Factor = ', string(lambda(i))))
    subplot(212)
    plot(thx(:,2))
    hold on
    plot(Aest(:,3))
    legend('True a_2', 'Estimated a_2')
end
%% Estimate best lambda
n=100;
lambda_line = linspace(0.85, 1 , n) ;
ls2 = zeros(n, 1);
yhat = zeros(n, 1);
for i = 1:length(lambda_line)
    reset(X);
    X.ForgettingFactor = lambda_line(i) ;
    X.InitialA = [1 0 0];
    for kk=1:length(tar2)
        [~, yhat(kk)] = step(X, tar2(kk)) ;
    end
    ls2(i) = sum((tar2 - yhat).^2);
end
[min_val, idx] = min(ls2);
best_lambda = lambda_line(idx);
figure()
plot(lambda_line, ls2)
hold on
plot(best_lambda, min_val, '*', 'Color', 'r')

%% Kalman
% Example of Kalman filter
% Simulate N samples of a process to test your code.
y = ?;     % Simulated data

A = [? ?; ? ?];
Re = [0.004 0; 0 0];          % State covariance matrix
Rw = 1.25;                   % Observation variance

% Set some initial values
Rxx_1 = 10 * eye(2);      %Initial state variance
xtt_1 = [0 0]';          %Initial state values

% Vectors to store values in
Xsave = zeros ( 2 ,N ) ; % Stored s t a t e s
ehat = zeros ( 1 ,N ) ; % Prediction r e s i d u a l
yt1 = zeros ( 1 ,N ) ; % One step prediction
yt2 = zeros ( 1 ,N ) ; % Two step prediction

% The filter use data up to time t−1 to predict value at t ,
% then update using the prediction error. Why do we start
% from t =3? Why stop at N−2?

for t=3:N-2
    Ct = [? ?];  %C_{t|t-1}
    yhat(t) = ?;  %y_{t|t-1}
    ehat(t) = ?;  % e_t = y_t - y_{t|t-1}

    % Update 
    Ryy = ?;      %R^{yy}_{t|t-1}
    Kt = ?;         %K_t
    xt_t = ?;       %x_t{t|t}
    Rxx = ?;        %R{xx}_{t|t}

    % Predict the next state
    xt_t1 = ?;      % x_{t+1|t}
    Rxx_1 = ?;      % R^{xx}_{t+1|t}

    % Form 2-step prediction. Ignore this part at first.
    Ct1 = [? ?];    %C_{t+1|t}
    yt1(t+1) = ?;   %y_{t+1|t} = C_{t+1|t} x_{t|t}

    Ct2 = [? ?];    %C_{t+2|t}
    yt2(t+2) = ?;   %y_{t+2|t} = C_{t+2|t} x_{t|t}
    
    % Store the state vector
    Xsave(:,t) = xt_t;

end
