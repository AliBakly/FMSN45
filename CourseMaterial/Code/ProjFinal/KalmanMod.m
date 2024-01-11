%% Estimate the unknown parameters using a Kalman filter and form the predictions.
load BJSIMDAT.mat
N = length(t_predict_model_val_test);
y = y_trend_model_val_test;
x = x_model_val_test;

nnz_KB  = find(KB ~= 0);
noPar   = length(nnz_KB) + 3 ;                            % The vector of unknowns is [ -KA(2) -KA(3) KB(1) KB(2) KB(3) KC(3) ]
xt      = zeros(noPar, N);               % Estimated states. Set the initial state to the estimated parameters.
xt(:,36) = [-KA(2) -0.3 KB(nnz_KB) 0.3];

%%
% For the output
A     = eye(noPar);
Rw    = 5e-4;%std(ehat_model_val);              % Measurement noise covariance matrix, R_w. Try using the noise estimate from the polynomial prediction.
Re    = 1e-6*eye(noPar);                        % System noise covariance matrix, R_e.
Re(2,2) = 1e-4; Re(15,15) = 1e-4;% Re(13,13) = 7e-5;
Rx_t1 = 1e-6*eye(noPar);                        % Initial covariance matrix, R_{1|0}^{x,x}
Rx_t1(2,2) = 1e-4; Rx_t1(15,15) = 1e-4;%
Rx_k  = Rx_t1;
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
yhat = zeros(N,1);                             % Estimated output.
xStd  = zeros(noPar,N);                         % Stores one std for the one-step prediction.
yhatk_1 = zeros(N,1);  
yhatk_7 = zeros(N,1);  

% We need to predict inputs also
noPar_inp = length(inputModel.A) - 1;
A_inp     = zeros(noPar_inp);
A_inp(1, :) = - inputModel.A(2:end);
A_inp(2: noPar_inp+1:end) = 1;

xt_inp      = zeros(noPar_inp, N);
Rw_inp    = 1;%std(ehat_model_val_inp);                                % Measurement noise covariance matrix, R_w. Try using the noise estimate from the polynomial prediction.
Re_inp    = 1e0*eye(noPar_inp);                        % System noise covariance matrix, R_e.
Rx_t1_inp = 1e0*eye(noPar_inp);                        % Initial covariance matrix, R_{1|0}^{x,x}
Rx_k_inp  = Rx_t1_inp;
h_et_inp  = zeros(N,1);                             % Estimated one-step prediction error.
xhat_inp = zeros(N,1);                             % Estimated output.
xhatK_inp_4 = zeros(N,1); % testing
xStd_inp  = zeros(noPar_inp,N);                         % Stores one std for the one-step prediction.

startInd = 37;                                   % We use t-2, so start at t=3.

for t=startInd:N
    % Input
    % Update the predicted state and the time-varying state vector.
    x_t1_inp = A_inp*xt_inp(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    
    C_inp = [inputModel.C zeros(1, noPar_inp - length(inputModel.C))];
    xhat_inp(t) = C_inp*x_t1_inp;

    % Update the parameter estimates.
    Ry_inp = C_inp*Rx_t1_inp*C_inp' + Rw_inp;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt_inp = Rx_t1_inp*C_inp'/Ry_inp;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    h_et_inp(t) = x(t)-xhat_inp(t);                    % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt_inp(:,t) = x_t1_inp + Kt_inp*( h_et_inp(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t_inp  = Rx_t1_inp - Kt_inp*Ry_inp*Kt_inp';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1_inp = A_inp*Rx_t_inp*A_inp' + Re_inp;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Estimate a one std confidence interval of the estimated parameters.
    xStd_inp(:,t) = sqrt( diag(Rx_t_inp) );             % This is one std for each of the parameters for the one-step prediction.
    Ck = C_inp;  % Not time varying
    xk = Ck*A_inp*xt_inp(:,t);            % \hat{x}_{t+1|t} = C_{t+1|t} A x_{t|t}
    % 2 - 4 step predictions
    k= 4;
    Rx_k_inp = Rx_t1_inp;
    ksteps_inp = [xk]; %[\hat{x}_{t+1|t} \hat{x}_{t+2|t} \hat{x}_{t+3|t} \hat{x}_{t+4|t}]

    for k0=2:k
        xk = Ck*A_inp^k*xt_inp(:,t);                    % \hat{x}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        Rx_k_inp = A_inp*Rx_k_inp*A_inp' + Re_inp;                  % R_{t+k+1|t}^{x,x} = A R_{t+k|t}^{x,x} A^T + Re  
        ksteps_inp(end + 1) = xk;
    end

    xhatK_inp_4(t + k) = xk;

    % Output
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}

    C = [ y(t-1) y(t-36) x(t - nnz_KB + 1)' h_et(t-36)];
    yhat(t) = C*x_t1;

    % Update the parameter estimates.
    Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    h_et(t) = y(t)-yhat(t);                    % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rx_t) );             % This is one std for each of the parameters for the one-step prediction.
    
    Ck = [y(t-1 + 1) y(t- 36 +1) x(t - nnz_KB + 1 + 1)' h_et(t-36 +1)];   % C_{t+1|t}
    yk = Ck*A*xt(:,t);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}
    k = 1;
    yhatk_1(t + k) = yk;
    
    k= 7;
    Rx_k = Rx_t1;
    for k0=2:k
        if (k0 > 3) % when k >= 4 we have future inputs 
            Ck = [yk y(t- 36 +k0) ksteps_inp(1:k0 - 3) x(t - nnz_KB(k0 - 3 +1:end) + k0)' h_et(t-36 +k0)]; % C_{t+k|t}
        else
            Ck = [yk y(t- 36 +k0) x(t - nnz_KB + k0 + 1)' h_et(t-36 +k0)]; % C_{t+k|t}
        end
        yk = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        Rx_k = A*Rx_k*A' + Re;                  % R_{t+k+1|t}^{x,x} = A R_{t+k|t}^{x,x} A^T + Re  
    end

    yhatk_7(t + k) = yk;  
end

yhatk_1 = yhatk_1(1:end-1);
yhatk_1 = yhatk_1 + (slope.*t_predict_model_val_test + intercept);
yhatk_7 = yhatk_7(1:end - 7);
yhatk_7 = yhatk_7 + (slope.*t_predict_model_val_test + intercept);
xhatK_inp_4 = xhatK_inp_4(1:end-4);

%% Examine the estimated parameters.

trueParams = [ -KA(2) -0.3 KB(nnz_KB) 0.3];

figure
plotWithConf( 1:N, xt', xStd', trueParams );
axis([startInd-1 N -1 1])
title(sprintf('Estimated parameters, with Re = %7.6f and Rw = %4.3f', Re(1,1), Rw(1,1)))
xlabel('Time')
fprintf('The final values of the Kalman estimated parameters are:\n')
max_5 = zeros(7,2);
for k0=1:length(trueParams)
    %fprintf('  True value: %5.2f, estimated value: %5.2f (+/- %5.4f).\n', trueParams(k0), xt(k0,end), xStd(k0,end) )
    abs(trueParams(k0)- xt(k0,end))
    if abs(trueParams(k0)- xt(k0,end)) > min(max_5(:,2))
        index = find(min(max_5(:,2)) == max_5(:,2));
        max_5(index(1), 1) = k0;
        max_5(index(1), 2) = abs(trueParams(k0)- xt(k0,end));
    end
    
end 
max_5
%xhatK_inp_4 = xhatK_inp_4(1:end-(k-1));
%% 1- step
ehat_1 = y_model_val_test - yhatk_1;
ehat_7 = y_model_val_test - yhatk_7;
counter = 1;
for k = [1 7]
    if k ==1
        yhatk = yhatk_1;
        ehat = ehat_1;
    else
        yhatk = yhatk_7;
        ehat = ehat_7;
    end

    naive = naive_vals{k};
    % Val
    f  = figure;
    subplot(211)
    plot(validation_t,y_model_val_test(idx_validation_nvdi))
    hold on;
    plot(validation_t, yhatk(idx_validation_nvdi))
    hold on;
    plot(validation_t, naive{1}(:,1))
    xlabel('Time')
    legend('True', 'Kalman estimate', 'naive')
    title(append(int2str(k),'-step prediction of the validation data'))
    
    subplot(212)
    plot(validation_t, ehat(idx_validation_nvdi))
    hold on;
    plot(validation_t, naive{1}(:,2))
    legend('ehat', 'ehat naive')
    xlabel('Time')

    
    % Test
    figure
    subplot(211)
    plot(test_t,y_model_val_test(idx_test_nvdi))
    hold on;
    plot(test_t, yhatk(idx_test_nvdi))
    hold on;
    plot(test_t, naive{2}(:,1))
    xlabel('Time')
    legend('True', 'Kalman estimate', 'naive')
    title(append(int2str(k),'-step prediction of the test data'))
    
    subplot(212)
    plot(test_t, ehat(idx_test_nvdi))
    hold on;
    plot(test_t, naive{2}(:,2))
    
    legend('ehat', 'ehat naive')
    xlabel('Time')

end

%%
% Form the prediction residuals for the validation/test data.
var_ehat_validation_Kalman_norm = var(ehat_1(idx_validation_nvdi))/var(y_model_val_test(idx_validation_nvdi))
var_ehat_test_Kalman_norm = var(ehat_1(idx_test_nvdi))/var(y_model_val_test(idx_test_nvdi))

var_ehat_validation_Kalman_norm = var(ehat_7(idx_validation_nvdi))/var(y_model_val_test(idx_validation_nvdi))
var_ehat_test_Kalman_norm = var(ehat_7(idx_test_nvdi))/var(y_model_val_test(idx_test_nvdi))

%plotACFnPACF( eP, 40, 'One-step prediction using the polynomial estimate');
plotACFnPACF( ehat_1(150: end), 40, '1-step prediction using the Kalman filter'); 
plotACFnPACF( ehat_7(150: end), 40, '7-step prediction using the Kalman filter');
%%
figure();
plot(t_predict_model_val_test, x)
hold on;
plot(t_predict_model_val_test, xhatK_inp_4);

title("Input prediction using Kalman filter with stationary model.")
legend( 'True', '4-step')
e_inp = x- xhatK_inp_4;
[acfest, pacfest] = plotACFnPACF( e_inp(150: end), 40, '7-step prediction using the Kalman filter');

checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(e_inp(150: end));