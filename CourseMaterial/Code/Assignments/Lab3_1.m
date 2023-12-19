%% Clean up and add path
clear; clc;
close all;
addpath('../functions', '../data')
%% Section 2.1
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

%% 2.3 Using the Kalman filter for prediction

clear; clc;
rng(0)
N = 10000;
ee = 0.1 * randn(N,1);
A0 = [1 -.8 .2];
y = filter(1, A0, ee);
Re = [1e-6 0; 0 1e-6]; 
Rw = 0.1;

k = 2; %k-step pred

A = eye(2);

Rxx_1 = eye(2);                             % Initial covariance matrix, R_{1|0}^{x,x}
Rx_k  = Rxx_1;
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = zeros(2,N-k);                       % Estimated states. Intial state, x_{1|0} = 0.
yhat  = zeros(N-k,1);                           % Estimated output.
yhatk = zeros(N-k,1);  
ehat = zeros ( 1 ,N ) ; % Prediction residual

for t=3:N-k
    x_t1 = A*xt(:,t-1);          % x_{t|t-1} = A x_{t-1|t-1}
    Ct = [ -y(t-1) -y(t-2) ];     % C_{t|t-1}

    yhat(t) = Ct*x_t1;  %y_{t|t-1} (8.103)
    ehat(t) = y(t) - yhat(t);  % e_t = y_t - y_{t|t-1}

    % Update the parameter estimates.
    Ryy = Ct*Rxx_1*Ct' + Rw;      %R^{yy}_{t|t-1} (8.116 i boken)
    Kt = Rxx_1*Ct'/(Ryy);       %K_t (8.111)
 
    
    xt(:,t) = x_t1 + Kt*ehat(t);
    
    %Update the covariance matrix
    Rxx = (eye(2)-Kt*Ct)*Rxx_1;  %R{xx}_{t|t} (8.114)
    Rxx_1 = A*Rxx*A' + Re;      % R^{xx}_{t+1|t}, (8.115)
    
    % Form the k-step prediction by first constructing the future C vector
    % and the one-step prediction. Note that this is not yhat(t) above, as
    % this is \hat{y}_{t|t-1}.
    Ck = [ -y(t) -y(t-1) ];           % C_{t+1|t}
    yk = Ck*xt(:,t);                  % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}
    
    % Note that the k-step predictions is formed using the k-1, k-2, ...
    % predictions, with the predicted future noises being set to zero. If
    % the ARMA has a higher order AR part, one needs to keep track of each
    % of the earlier predicted values.
    Rx_k = Rxx_1;
    for k0 = 2:k
        Ck = [ -yk -y(t-2+k0) ]; % C_{t+k|t}
        yk = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        Rx_k = A*Rx_k*A' + Re;                  % R_{t+k+1|t}^{x,x} = A R_{t+k|t}^{x,x} A^T + Re  
    end
    yhatk(t+k) = yk;  
    

end

figure
plot(xt(2,:))
legend('a2 (true value 0.2)')

% Plotting the 1 and 2-step Kalman filter predictions
figure
plot(y(end-100-2:end-2))
hold on
plot( yhat(end-100:end),'g' ) 
plot( yhatk(end-100:end),'r')
hold off
legend ( 'y' , 'k=1' , 'k=2' )

% Sum of 200 last squared residuals
sum(ehat(end-200:end).^2)

%% 2.4 Quality control of a process
clc; clear;
N = 1000;
rng(0)
e = randn(N,1);
v = randn(N,1);
x = zeros(N,1);
y = zeros(N,1);
b = 20;
%P = [7/8, 1/8; 1/8, 7/8];
%mc = dtmc(P);

P = 1/8;
u  = zeros(1,1000);
u(1) = [round(rand(1))];

for i =2:1000
    random = rand(1);
    if random < P
        u(i) = abs(u(i-1) - 1);
    end
end
figure()
plot(u)

y(1) = x(1) + b*u(1) + v(1);
for i=2:N
    x(i) = x(i-1) + e(i);
    y(i) = x(i) + b*u(i) + v(i);
end


%% Estimating parameters with Kalman Filter

A     = eye(2);                                 % A matrix should be I when estimating params
Rw    = 10e-3;                                      % Measurement noise covariance matrix, R_w. Note that Rw has the same dimension as Ry.
Re    = eye(2)*10e-1;                        % System noise covariance matrix, R_e. Note that Re has the same dimension as Rx_t1.
Rx_t1 = 1;                             % Initial covariance matrix, V0 = R_{1|0}^{x,x}
e_hat  = zeros(N,1);                             % Estimated one-step prediction error.
xt    = zeros(2,N);                         % Estimated states. Intial state, x_{1|0} = 0.
yhat  = zeros(N,1);                             % Estimated output.
xStd  = zeros(2,N);                         % Stores one std for the one-step prediction.
for t=2:N                                       
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    C    = [ 1 u(t) ];     % Use earlier prediction errors as estimate of e_t.
    
    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    yhat(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
    e_hat(t) = y(t)-yhat(t);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( e_hat(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Estimate a one std confidence interval of the estimated parameters.
    % This is only for the plots.
    xStd(:,t) = sqrt( diag(Rx_t) );            % This is one std for each of the parameters for the one-step prediction.
end

%% Looking at estimated parameters
figure()
hold on
plot(xt(1,:))
plot(x)
plot(xt(2,:))
plot(b*ones(N,1))
legend('Estimated process', 'True Process', 'Estimated b', 'True b')

%% 2.5 Recursive temperature modeling
clc; clear;
load svedala94.mat

AS = [1 zeros(1, 5) -1];
ydiff = filter(AS, 1, svedala94); 
ydiff = ydiff(length(AS):end);

figure
plot(ydiff)
title('Svedala temp season differentiated')

T = linspace(datenum(1994, 1, 1), datenum(1994, 12, 31), length(svedala94));
figure
plot(T, svedala94);
xlabel('Svedala temp.')
datetick('x');

% Using armax to estimate an ARMA(2,2)-process

th = armax(ydiff, [2,2]);
th_winter = armax(ydiff(1:540), [2,2]);
th_summer = armax (ydiff(907:1458 ) , [2,2] ) ;

% Use the recursiveARMA object to estimate A(z) and C(z)

X_winter = recursiveARMA([2,2]);
X_winter.InitialA = [1 th_winter.A(2:end)];
X_winter.InitialC = [1 th_winter.C(2:end)];

X_winter.ForgettingFactor = 0.95;

for k= 1:length(ydiff)   
    [Aest(k,:), Cest(k,:), yhat(k)] = step(X_winter, ydiff(k));
end
figure()
subplot 311
plot(T, svedala94 );
datetick('x')

subplot 312
plot(Aest (:, 2:3))
% hold on
% plot(repmat(th_winter.A(2:end),[length(ydiff) 1]),'g:'); 
% hold on
% plot(repmat(th_summer.A(2:end),[length(ydiff) 1]),'r:');
title('Recursive estimates for A')
% axis tight
% hold off


subplot 313
hold on
plot(Cest(:,2:3))
% hold on
% plot(repmat(th_winter.C(2:end),[length(ydiff) 1]),'g:'); 
% hold on
% plot(repmat(th_summer.C(2:end),[length(ydiff) 1]),'r:');
% xlim([1 2184])
title('Recursive estimates For C')
% hold off

%% 2.6 Recursive temperature modeling, again
clc; clear;
load svedala94

y = svedala94(850:1100);

newY = y - mean(y);

t = (1:length(newY))';
U = [sin(2*pi*t/6) cos(2*pi*t/6)];
Z = iddata(y,U);
model = [3 [1 1] 4 [0 0]];
    % [na [nb_1 nb_2] nc [nk_1 nk_2]];

thx = armax(Z,model);
figure()
plot(U *cell2mat(thx.b)')
hold on
plot(newY)
legend('Seasonal function', 'Zero mean-Data')


%% 2.6.3

U = [sin(2*pi*t/6) cos(2*pi*t/6) ones(size(t))];
Z = iddata(y,U);
m0 = [thx.A(2:end) thx.B 0 thx.C(2:end)];
m0 = cell2mat(m0);
Re = diag([0 0 0 0 0 1 0 0 0 0]); %??
model = [3 [1 1 1] 4 0 [0 0 0] [1 1 1]];
[thr, yhat] = rpem(Z,model, 'kf', Re, m0);
figure()
plot(thr)

%% 2.6.4
m = thr(:,6);
a = thr(end,4);
b = thr(end,5);
y_mean = m + a.*U(:,1)+b.*U(:,2);
y_mean = [0;y_mean(1:end-1)];
figure()
plot(newY, 'r')
hold on
plot(y_mean, 'b')
plot(m)
legend('y','y_mean','m')
%% 2.6.5
y = svedala94;
y = y-y(1);
t = (1:length(y))';

U = [sin(2*pi*t/6) cos(2*pi*t/6) ones(size(t))];
Z = iddata(y,U);
m0 = [thx.A(2:end) thx.B 0 thx.C(2:end)];
m0 = cell2mat(m0);
Re = diag([0 0 0 1 1 1 0 0 0 0]); %??
model = [3 [1 1 1] 4 0 [0 0 0] [1 1 1]];
[thr, yhat] = rpem(Z,model, 'kf', Re, m0);

m = thr(:,6);
a = thr(end,4);
b = thr(end,5);
y_mean = m + a.*U(:,1)+b.*U(:,2);
y_mean = [0;y_mean(1:end-1)];
plot(y, 'r')
hold on
plot(y_mean, 'b')
plot(m)
legend('y','y_mean','m')