%% Generate data from a BJ model
load BJSIMDAT.mat
rng(0)
n = 1000; % Number of samples
A3 = A;
C3 = C;
w = randn(n + 100, 1) ;
x = filter(C3, A3, w) ; % Create the input
A1 = MboxJ.D;
A2 = MboxJ.F;
C = MboxJ.C;
B = MboxJ.B ;
e = randn(n + 100, 1) ;
y = filter(C, A1, e) + filter(B, A2, x) ; % Create the output
x = x(101 : end);  y = y(101 : end); % Omit initial samples

rng(0)
n = 1000; % Number of samples
A3 = A;
C3 = C;
w_2 = randn(n + 100, 1) ;
x_2 = filter(C3, A3, w) ; % Create the input
A1_2 = [MboxJ.D(1) MboxJ.D(2:end)-1];
A2_2 = [MboxJ.F(1) MboxJ.F(2:end)];
C_2 = [MboxJ.C(1) MboxJ.C(2:end)];
B_2 = [MboxJ.B(1) MboxJ.B(2:end).*3 ] ;
e_2 = randn(n + 100, 1) ;
y_2 = filter(C_2, A1_2, e_2) + filter(B_2, A2_2, x_2) ; % Create the output
x_2 = x_2(101 : end);  y_2 = y_2(101 : end); % Omit initial samples

y_full = [y;y_2];
x_full = [x;x_2];

%clear A1 A2 C B e w A3 C3
%%
KA = conv( A1, A2 );
KB = conv( A1, B);
KC = conv( A2, C );
KA_2 = conv( A1_2, A2_2 );
KB_2 = conv( A1_2, B_2);
KC_ = conv( A2_2, C_2 );
%% Estimate the unknown parameters using a Kalman filter and form the one-step prediction.
N = 2000;
y = y_full;
x = x_full;

%%%% Estimate the unknown parameters using a Kalman filter and form the predictions.

nnz_KB  = find(KB ~= 0);
noPar   = length(nnz_KB) + 1 ;                            % The vector of unknowns is [ -KA(2) -KA(3) KB(1) KB(2) KB(3) KC(3) ]
xt      = zeros(noPar, N);               % Estimated states. Set the initial state to the estimated parameters.
xt(:,max(nnz_KB)-1) = [-KA(2) KB(nnz_KB)];

%%
% For the output
A     = eye(noPar);
Rw    = 1;%std(ehat_model_val);              % Measurement noise covariance matrix, R_w. Try using the noise estimate from the polynomial prediction.
Re    = 5e-5*eye(noPar);                        % System noise covariance matrix, R_e.
%Re(3,3) = 1e-3;Re(4,4) = 1e-3;Re(9,9) = 1e-3;Re(10,10) = 1e-3;
Rx_t1 = 1e-6*eye(noPar);                        % Initial covariance matrix, R_{1|0}^{x,x}
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
Rw_inp    = 1e-3;%std(ehat_model_val_inp);                                % Measurement noise covariance matrix, R_w. Try using the noise estimate from the polynomial prediction.
Re_inp    = 1e0*eye(noPar_inp);                        % System noise covariance matrix, R_e.
Rx_t1_inp = 1e0*eye(noPar_inp);                        % Initial covariance matrix, R_{1|0}^{x,x}
Rx_k_inp  = Rx_t1_inp;
h_et_inp  = zeros(N,1);                             % Estimated one-step prediction error.
xhat_inp = zeros(N,1);                             % Estimated output.
xhatK_inp_4 = zeros(N,1); % testing
xStd_inp  = zeros(noPar_inp,N);                         % Stores one std for the one-step prediction.

startInd = max(nnz_KB);                                   % We use t-2, so start at t=3.

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

    C = [ y(t-1) x(t - nnz_KB + 1)'];
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
    
    Ck = [y(t-1 + 1) x(t - nnz_KB + 1 + 1)'];   % C_{t+1|t}
    yk = Ck*A*xt(:,t);                            % \hat{y}_{t+1|t} = C_{t+1|t} A x_{t|t}
    k = 1;
    yhatk_1(t + k) = yk;
    
    k= 7;
    Rx_k = Rx_t1;
    for k0=2:k
        if (k0 > 3) % when k >= 4 we have future inputs 
            Ck = [yk ksteps_inp(1:k0 - 3) x(t - nnz_KB(k0 - 3 +1:end) + k0)']; % C_{t+k|t}
        else
            Ck = [yk x(t - nnz_KB + k0 + 1)']; % C_{t+k|t}
        end
        yk = Ck*A^k*xt(:,t);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        Rx_k = A*Rx_k*A' + Re;                  % R_{t+k+1|t}^{x,x} = A R_{t+k|t}^{x,x} A^T + Re  
    end

    yhatk_7(t + k) = yk;  
end

yhatk_1 = yhatk_1(1:end-1);
%yhatk_1 = yhatk_1 + (slope.*t_predict_model_val_test + intercept);
yhatk_7 = yhatk_7(1:end - 7);
%yhatk_7 = yhatk_7 + (slope.*t_predict_model_val_test + intercept);


%% Examine the estimated parameters.

trueParams_1 = [ -KA(2) KB(nnz_KB)];
trueParams_2 = [ -KA_2(2) KB_2(nnz_KB)];

figure
plotWithConf( 1:1000, xt(:,1:1000)', xStd(:,1:1000)', trueParams_1 );
hold on
plotWithConf( 1001:2000, xt(:,1001:2000)', xStd(:,1001:2000)', trueParams_2 );
axis([startInd-1 N -1.5 1.5])
title(sprintf('Estimated parameters, with Re = %7.6f and Rw = %4.3f', Re(1,1), Rw(1,1)))
xlabel('Time')
fprintf('The final values of the Kalman estimated parameters are:\n')
for k0=1:length(trueParams)
    fprintf('  True value: %5.2f, estimated value: %5.2f (+/- %5.4f).\n', trueParams(k0), xt(k0,end), xStd(k0,end) )
end 


% Show the one-step prediction. 
%xhatK_inp_4 = xhatK_inp_4(1:end-(k-1));
figure
plot( [y_model_val_test yhatk_1 ] )
title('1-step prediction of the validation data')
xlabel('Time')
legend('Realisation', 'Kalman estimate')
xlim([0 N])

figure
plot( [y_model_val_test yhatk_7 ] )
title('7-step prediction of the validation data')
xlabel('Time')
legend('Realisation', 'Kalman estimate')
xlim([0 N])

ehat_1 = y_model_val_test - yhatk_1;
ehat_7 = y_model_val_test - yhatk_7;
% Form the prediction residuals for the validation/test data.
var_ehat_validation_Kalman_norm = var(ehat_1(idx_validation_nvdi))/var(y_model_val_test(idx_validation_nvdi))
var_ehat_test_Kalman_norm = var(ehat_1(idx_test_nvdi))/var(y_model_val_test(idx_test_nvdi))

var_ehat_validation_Kalman_norm = var(ehat_7(idx_validation_nvdi))/var(y_model_val_test(idx_validation_nvdi))
var_ehat_test_Kalman_norm = var(ehat_7(idx_test_nvdi))/var(y_model_val_test(idx_test_nvdi))

%plotACFnPACF( eP, 40, 'One-step prediction using the polynomial estimate');
plotACFnPACF( ehat_1(150: end), 40, '1-step prediction using the Kalman filter'); 
plotACFnPACF( ehat_7(150: end), 40, '7-step prediction using the Kalman filter');