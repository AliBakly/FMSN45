%% Clean up and add path
clear; clc;
close all;
addpath('CourseMaterial/Code/functions', 'CourseMaterial/Code/data')
%% Lab3 prep exercise 2
P = [7/8 1/8; 1/8 7/8];
n = 1000;
states = [3 8];

x = zeros(1,n);
x(1) = states(1);
state = 1;

for i=2:n
    random = rand(1);
    j = 1;
    while(random > 0)
        random = random - P(state,j);
        j = j + 1;
    end
    j = j - 1;
    state = j;
    x(i) = states(j);
end

sum(x>5)
%%
rng(0)
P = 7/8;
u  = zeros(1,1000);
u(1) = [round(rand(1))];

for i =2:1000
    random = rand(1);
    if random < 1/8
        u(i) = abs(u(i-1) - 1);
    end
end
figure()
plot(u)
%% Lab3 prep exercise 1
load thx.dat
load tar2.dat
y = tar2;

N = length(y);
A = eye(2);

Re = [0.004 0; 0 0];
Rw = 1.25;

% Set some initial values
Rxx_1 = 10 * eye(2);      %Initial state variance
xtt_1 = [0 0]';          %Initial state values

% Vectors to store values in
Xsave = zeros ( 2 ,N ) ; % Stored states
ehat = zeros ( 1 ,N ) ; % Prediction residual
yt1 = zeros ( 1 ,N ) ; % One step prediction
yt2 = zeros ( 1 ,N ) ; % Two step prediction
yhat = zeros ( 1,N);

% The filter use data up to time t−1 to predict value at t ,
% then update using the prediction error. start from t = 3
% since we do y(t-2), which is only defined when t-2 > 0.

for t=3:N
    
    x_t1 = A*Xsave(:,t-1); 
    Ct = [ -y(t-1) -y(t-2) ];  %C_{t|t-1}

    yhat(t) = Ct*x_t1;  %y_{t|t-1} (8.103)
    ehat(t) = y(t) - yhat(t);  % e_t = y_t - y_{t|t-1}
   

    % Update 
    Ryy = Ct*Rxx_1*Ct' + Rw;      %R^{yy}_{t|t-1} (8.116 i boken)
    Kt = Rxx_1*Ct'/(Ryy);       %K_t (8.111)
 
    Rxx = (eye(2)-Kt*Ct)*Rxx_1;  %R{xx}_{t|t} (8.114)
    Xsave(:,t) = x_t1 + Kt*ehat(t);
    % xt_t = x_t1 + (Kt*(y(t) - Ct*xtt_1));  %x_t{t|t}  (8.109)


    % Predict the next state
  %  xt_t1 = A*xtt_1;      % x_{t+1|t}, 2x1 A*xtt + B*u_t (8.102)
    Rxx_1 = A*Rxx*A' + Re;      % R^{xx}_{t+1|t}, (8.115)

    
    % Store the state vector
   % Xsave(:,t) = xt_t;
end

figure
hold on
plot(thx(:,1)) %true a1
plot(thx(:,2)) %true a2
plot(Xsave(1,:)')
plot(Xsave(2,:)')
legend('True a1', 'True a2', 'Estimated a1', 'Estimated a2')
sum(ehat.^2)
%% Plotting prediction
figure
hold on
plot(yhat)
plot(y)
legend('1-step pred. y-hat', 'tar2 data')

%acfNpacfnorm(yhat-tar2, 20, '1-step pred. error residual')

%% 2.3 Using the Kalman filter for prediction

clear; clc;
rng(0)
N = 10000;
ee = 0.1 * randn(N,1);
A0 = [1 -.8 .2];
y = filter(1, A0, ee);
Re = [1e-6 0; 0 1e-6]; 
Rw = 0.1;


