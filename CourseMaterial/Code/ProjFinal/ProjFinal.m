%% FMSN45 PROJECT 2023, By Ali Bakly & Davy Than
%% Clean up and add path
clear; clc;
close all;
addpath('../functions', '../data')
load proj23.mat
%% Initial Plot
city = ElGeneina; % If you want to reestimate params with Kassala, set to Kassala
figure()
area(city.rain_org_t, city.rain_org, 'FaceColor','#4DBEEE')
hold on
pl1 = plot(city.rain_t, city.rain,'*', 'color', 'r','LineWidth',1);
hold on
pl2 = plot(city.rain_org_t, city.rain_org,'*', 'color', '#0072BD','LineWidth',1);
lgd = legend([pl2 pl1], {'Meassurements', 'Interpolation'});
fontsize(lgd,12,'points')
xlabel('Time')
ylabel('Rain')
title('Rain plot')
rain = city.rain_org;

%% Find idxs for splits and set initial states
len = length(city.nvdi);
idx_model_nvdi = 1:round(0.75*len);
idx_validation_nvdi = round(0.75*len) + 1 : round(0.9*len);
idx_test_nvdi = round(0.9*len + 1 : len);
tt = city.nvdi_t;

last_t = tt(length(idx_model_nvdi)); % Use only model data for reconstruction
last_rain_rec = find(last_t - city.rain_org_t < 0 & last_t - city.rain_org_t > -0.1);

% Only same time as modelling for these params
y = city.rain_org(1:last_rain_rec);
N = length(y);

%% ----------A: Estimating parameters with Kalman Filter-------------
a1 = 0.5;%0.999999; % Intial AR param estimate
converged = false;
converged_2 = false;
convergence_counter = 0;
while converged_2 == false %
    if(converged) % If on last iteration, use all available data to reconstruct
        converged_2 = true;
        y = city.rain_org(1:end);
        N = length(y);
    end
    A =[-a1.^3 0 0;a1.^2 0 0; -a1 0 0];
    Z = [1 -a1 a1^2; 0 1 -a1; 0 0 1];
    at = zeros(N,1);
    Rw    = 1e-2;                                      % Measurement noise covariance matrix, R_w. Note that Rw has the same dimension as Ry.
    Re    = eye(3)*1e2;%10e-1                        % System noise covariance matrix, R_e. Note that Re has the same dimension as Rx_t1.
    Re = Z*Re*transpose(Z);
    Rx_t1 = eye(3)*1e2; %correct?                             % Initial covariance matrix, V0 = R_{1|0}^{x,x}
    e_hat  = zeros(N,1);                             % Estimated one-step prediction error.
    xt    = zeros(3,N);                         % Estimated states. Intial state, x_{1|0} = 0.
    yhat  = zeros(N,1);                             % Estimated output.
    xStd  = zeros(3,N);                         % Stores one std for the one-step prediction.
    for t=2 :N                                       
        % Update the predicted state and the time-varying state vector.
        x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
        C    = [1 1 1];     % Use earlier prediction errors as estimate of e_t.
        
        % Update the parameter estimates.
        Ry = C*Rx_t1*C' + Rw;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
        Kt = Rx_t1*C'/Ry;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
        yhat(t) = C*x_t1;                           % One-step prediction, \hat{y}_{t|t-1}.
        e_hat(t) = y(t)-yhat(t);                     % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
        xt(:,t) = x_t1 + Kt*( e_hat(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 
        
        if (any(xt(:,t) < 0))
            target_sum = sum(xt(:,t)); % The sum that needs to be maintained
            new_data = max(xt(:,t), 0);

            % Scale the new data to ensure the sum matches the target sum
            scale_factor = target_sum / sum(new_data);
            xt(:,t) = new_data * scale_factor;
        end

        % Update the covariance matrix estimates.
        Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
        Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re
    
        % Estimate a one std confidence interval of the estimated parameters.
        % This is only for the plots.
        xStd(:,t) = sqrt( diag(Rx_t) );            % This is one std for each of the parameters for the one-step prediction.    
    end
    
    if converged_2 == false
        xtemp = flip(xt,1);
        data = iddata(xtemp(:)); % All x data, should be AR
        model_init = idpoly([1 a1], [], []) ;
        model_armax_1 = pem(data , model_init);
        present(model_armax_1)
        if a1 == model_armax_1.A(2)
            converged = true; % We have converged
            final_a1 = a1
        else
            a1 = model_armax_1.A(2) % New Ar parameter, update and loop
            convergence_counter = convergence_counter +1;
        end
    end
end

%% Reconstruct 
rain_rec = flip(xt,1);
rain_reconstructed = rain_rec(:);
rain_reconstructed = [rain_reconstructed(2:end); 0];
% With Provided time vector
tt = city.rain_t;

std_rec = flip(xStd,1); 
std_reconstructed = std_rec(:);
std_reconstructed = [std_reconstructed(2:end); std_reconstructed(end)];

%% Viz for Project
figure()
area(city.rain_org_t, city.rain_org, 'FaceColor','#9eed34')
hold on
area(tt, rain_reconstructed, 'FaceColor','#4DBEEE')
hold on
plot(tt, rain_reconstructed,'*', 'color', '#0072BD','LineWidth',1)
hold on
plot(city.rain_org_t, city.rain_org,'*', 'color', '#395c0a','LineWidth',1)
lgd = legend('Original Rain', 'Reconstructed Rain');
fontsize(lgd,11,'points')
xlabel('Time')
ylabel('Rain')
%xlim([tt(14),tt(31)])
%ylim([0, 250])
title('Reconstructed Rain')
%plotWithConf( 1:3*N, rain_reconstructed, std_reconstructed);
%% Compare sums
sums = zeros(480, 1);
sums(1) = 0;
counter = 2;
for i=3:3:length(rain_reconstructed)-2
    sums(counter) = rain_reconstructed(i) + rain_reconstructed(i+1) + rain_reconstructed(i+2);
    counter = counter +1;
end
diff = sums - city.rain_org;
sum(abs(diff)) % Should be close to 0.
%% ---------- B ----------------- 
% Plot
nvdi = 2.*(city.nvdi./255) -1; %log?
nvdi_K = 2.*(Kassala.nvdi./255) -1;
nvdi_t = city.nvdi_t;
figure();
plot(nvdi_t, nvdi)
xlabel('Time')
ylabel('NVDI')
title('NVDI Data and Split')

plotACFnPACF(nvdi, 100, 'data');

%%
model_t = nvdi_t(idx_model_nvdi);
validation_t = nvdi_t(idx_validation_nvdi);
test_t = nvdi_t(idx_test_nvdi);

model_nvdi = nvdi(idx_model_nvdi);
validation_nvdi = nvdi(idx_validation_nvdi);
test_nvdi = nvdi(idx_test_nvdi);

model_nvdi_K = nvdi_K(idx_model_nvdi);
validation_nvdi_K = nvdi_K(idx_validation_nvdi);
test_nvdi_K = nvdi_K(idx_test_nvdi);

figure();
plot(model_t, model_nvdi, 'LineWidth',1.2)
hold on
plot([model_t(end); validation_t], [model_nvdi(end); validation_nvdi], 'LineWidth',1.2)
hold on;
plot([validation_t(end); test_t], [validation_nvdi(end); test_nvdi], 'LineWidth',1.2)
lgd = legend('Model', 'Validation', 'Test');
fontsize(lgd,12,'points')
xlabel('Time')
ylabel('NVDI')
title('NVDI Data and Split')

figure; 
lambda_max = bcNormPlot(model_nvdi,1)
%% Remove deterministic trend
mdl = fitlm(model_t, model_nvdi)
intercept = mdl.Coefficients(1,1).Estimate;
slope = mdl.Coefficients(2,1).Estimate;
nvdi_no_trend = model_nvdi- (model_t.*slope + intercept);
nvdi_no_trend_copy = nvdi_no_trend;
figure();
plot(model_t, nvdi_no_trend)
plotACFnPACF(nvdi_no_trend, 40, 'data');
figure; 
lambda_max = bcNormPlot(nvdi_no_trend,1);
figure();
plot(model_t, nvdi_no_trend)
 
%% Ar(1) with (1+a36 z^-36) and C = 36
nvdi_no_trend_copy(90:100) = nvdi_no_trend(90-36:100-36); % Outlier
figure();
plot(model_t, nvdi_no_trend_copy,'LineWidth',1.2)
xlabel('Time')
title('No trend data + no outlier')
legend('Model')
plotACFnPACF(nvdi_no_trend_copy, 100, 'No trend data without outlier');
figure; 
lambda_max = bcNormPlot(nvdi_no_trend_copy + 0.2,1)
title("Box-Cox normality plot")

data = iddata(nvdi_no_trend_copy);
Am = conv([1 0.1 0.1], [1 zeros(1, 35), 0.1]);
Cm = [1 zeros(1, 35),0.1];
model_init = idpoly(Am, [], Cm);
model_init.Structure.a.Free = Am;
model_init.Structure.c.Free = Cm;
model_armax_1 = pem(data , model_init);

e_hat = filter(model_armax_1.A , model_armax_1.C , nvdi_no_trend_copy) ;
e_hat = e_hat(length(model_armax_1.A ) : end ) ;

present(model_armax_1)
[acfEst, pacfEst] = plotACFnPACF(e_hat, 70, 'NVDI Model' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(e_hat);

%% PREDICTION
ks = [1 7];
naive_vals = {};
Kassala = false; % Predict on Kassala without reestimating --> set to true !
for i=1:2
    t_predict = cat(1,model_t, validation_t, test_t);
    if Kassala
        y = cat(1, model_nvdi_K, validation_nvdi_K, test_nvdi_K);
    else
        y = cat(1, model_nvdi, validation_nvdi, test_nvdi);
    end
    y_trend = y - (t_predict.*slope + intercept);
    A = model_armax_1.A;
    C = model_armax_1.C;

    k = ks(i);
    % Naive:
    if(i == 1) % 1 - Step: Use previous value
        y_naive = zeros(1, length(y) +1 );
        y_naive(2: end) = y;
        y_naive = y_naive(1:end - 1);
        ehat_naive = y-y_naive';

        y_naive_val = y_naive(idx_validation_nvdi);
        ehat_naive_val = ehat_naive(idx_validation_nvdi);

        y_naive_test = y_naive(idx_test_nvdi);
        ehat_naive_test = ehat_naive(idx_test_nvdi);

        naive_vals{1}{1} = [y_naive_val' ehat_naive_val];
        naive_vals{1}{2} = [y_naive_test' ehat_naive_test];
    else % 7 - step: Use last years value
        y_naive = zeros(1, length(y) + 36 );
        y_naive(37: end) = y;
        y_naive = y_naive(1:end - 36);
        ehat_naive = y-y_naive';

        y_naive_val = y_naive(idx_validation_nvdi);
        ehat_naive_val = ehat_naive(idx_validation_nvdi);

        y_naive_test = y_naive(idx_test_nvdi);
        ehat_naive_test = ehat_naive(idx_test_nvdi);

        naive_vals{7}{1} = [y_naive_val' ehat_naive_val];
        naive_vals{7}{2} = [y_naive_test' ehat_naive_test];
    end
    
    % Predict
    [Fk, Gk] = polydiv(C, A, k) ;
    yhatk = filter(Gk, C, y_trend) ;
    yhatk = yhatk + (t_predict.*slope + intercept); % Add trend
    ehat = y-yhatk;
    [acfEst, pacfEst] = plotACFnPACF(ehat, 70, append(int2str(k), '-step prediction'));
    checkIfNormal(acfEst(2:end), 'ACF' );
    checkIfNormal(pacfEst(2:end), 'PACF' );
    checkIfWhite(e_hat);

    yhatk_val = yhatk(idx_validation_nvdi);
    ehat_val = ehat(idx_validation_nvdi);
    
    yhatk_test = yhatk(idx_test_nvdi);
    ehat_test = ehat(idx_test_nvdi);

    t_predict_val = t_predict(idx_validation_nvdi);
    y_val = y(idx_validation_nvdi);

    t_predict_test = t_predict(idx_test_nvdi);
    y_test = y(idx_test_nvdi);
    
    % Validation plots
    figure()
    subplot(211)
    plot(t_predict_val,y_val)
    hold on
    plot(t_predict_val, yhatk_val)
    hold on
    plot(t_predict_val, y_naive_val)
    title('NVDI: Validation')
    legend('True data', append(int2str(k), '-step prediction'), 'Naive')

    subplot(212)
    plot(t_predict_val, ehat_val)
    var_ehat_val_norm = var(ehat_val)./var(y_val)
    hold on
    plot(t_predict_val, ehat_naive_val)
    var_ehat_naive__val_norm = var(ehat_naive_val)./var(y_val)
    legend('ehat = y-yhatk', 'naive')

    % Test plots
    figure()
    subplot(211)
    plot(t_predict_test,y_test)
    hold on
    plot(t_predict_test, yhatk_test)
    hold on
    plot(t_predict_test, y_naive_test)
    title('NVDI: Test')
    legend('True data', append(int2str(k), '-step prediction'), 'Naive')

    subplot(212)
    plot(t_predict_test, ehat_test)
    var_ehat_test_norm = var(ehat_test)./var(y_test)
    hold on
    plot(t_predict_test, ehat_naive_test)
    var_ehat_naive_test_norm = var(ehat_naive_test)./var(y_test)
    legend('ehat = y-yhatk', 'naive')
end
%% BOX-Jenkins Examine the data.
% Extract rain for modelling
idx_rain_nvdimodel_dates = find(tt==model_t(1)): find(tt==model_t(1)) + length(nvdi_no_trend) - 1;
x = rain_reconstructed(idx_rain_nvdimodel_dates);
figure(); plot(x)
figure; 
lambda_max = bcNormPlot(x,1)
title("Box-Cox normality plot")
y = nvdi_no_trend; 
% Transform
x = log(x+1);
mean_x = mean(x);
x = (x - mean_x)./20;

% manual inspection
f =figure(); 
subplot(211);
plot(model_t,x); 
ylabel('Input signal')
title('Measured signals')
subplot(212); 
plot(model_t, y ); 
ylabel('Output signal')
xlabel('Time')
figure
[Cxy,lags] = xcorr( y, x, 100, 'coeff' );
stem( lags, Cxy )
hold on
condInt = 2*ones(1,length(lags))./sqrt( length(y) );
plot( lags, condInt,'r--' )
plot( lags, -condInt,'r--' )
hold off
xlabel('Lag')
ylabel('Amplitude')
title('Crosscorrelation between in- and output')
%% Modeling x
figure();

plot(x);
title('Input data');
plotACFnPACF(x, 100, 'Input Data' );

% Model as ARMA(1,2) component
data = iddata(x);

idx = [1 2 12 36 37 38];
arr = zeros(1, idx(end) +1);
arr(idx +1 ) = 1;
Am = arr.*0.3;
Am(1) =1;

idx = [3 36];
arr = zeros(1, idx(end) +1);
arr(idx +1 ) = 1;
Cm = arr.*0.01;
Cm(1) =1;
model_init = idpoly(Am, [], Cm) ;
model_init.Structure.a.Free = Am;
model_init.Structure.c.Free = Cm;

inputModel = pem(data , model_init);

w_t = filter(inputModel.A , inputModel.C , x) ;
w_t = w_t(length(inputModel.A ): end ) ;
var_ex = var(w_t);

eps_t = filter(inputModel.A , inputModel.C , y) ;
eps_t = eps_t(length(inputModel.A ): end ) ;

present(inputModel)
[acfEst, pacfEst] = plotACFnPACF(w_t, 100, 'Input Model' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(w_t);

A = inputModel.A;
C = inputModel.C;
%% Cross- corr
n = length(x);
M = 100;
figure()
[Cxy,lags] = xcorr(eps_t, w_t, M, 'coeff' ); %(d,r,s) = (4,2,0)
stem(lags, Cxy)
title ('Crosscorrelation function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off

%% Model orders
d=3; r=0; s=0; % Not really using this
A2 = [1 0.3.*ones(1, r)];
% B = [zeros(1,d) 0.1.*ones(1, s+1)];

idx = [3 5 7 17 31 34];
arr = zeros(1, idx(end) +1);
arr(idx +1 ) = 0.3;
B = arr;

Mi = idpoly([1], [B], [], [], [A2]) ;
Mi.Structure.B.Free = B;

z = iddata(y, x) ;
Mba2 = pem(z, Mi ) ;present(Mba2)
etilde = resid(Mba2, z);
etilde = etilde.y;
etilde = etilde(length(Mba2.B): end);

figure()
[Cxy,lags] = xcorr(etilde, x(length(Mba2.B): end), M, 'coeff' );
stem(lags, Cxy)
title ('Crosscorrelation function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off

%% Model etilde
figure();
plot(etilde);
title('etilde data');
plotACFnPACF(etilde, M, 'Input Data' );

model_init = idpoly([1 1], [], []) ;
model_input = pem(iddata(etilde), model_init);

e_hat = filter(model_input.A , model_input.C , etilde) ;
e_hat = e_hat(length(model_input.A ) : end ) ;

present(model_input)
[acfEst, pacfEst] = plotACFnPACF(e_hat, M, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(e_hat);

A1 = model_input.A;
C2 = model_input.C;

%% Full BJ Model
Mi = idpoly(1, B, C2, A1, A2) ;
Mi.Structure.D.Free = A1;
Mi.Structure.B.Free = B;

z = iddata(y, x);
MboxJ = pem (z, Mi);
present (MboxJ );

ehat = resid(MboxJ, z).y;
ehat = ehat(length(MboxJ.B): end);

[acfEst, pacfEst] = plotACFnPACF(ehat, M, 'Box-Jenkins');
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(ehat);

figure()
[Cxy,lags] = xcorr(ehat, x(length(MboxJ.B): end), M, 'coeff' );
stem(lags, Cxy)
title ('Cross-correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off
%% Compare variances
ut = filter( MboxJ.B, MboxJ.F, x);
ut = ut(length(MboxJ.B): end);
var(ut)
var(y(length(MboxJ.B): end))

%% Predicting using BJ
k = 7;
idx_rain_nvdi_val = find(tt==validation_t(1)): find(tt==validation_t(end)) ;
idx_rain_nvdi_test = find(tt==test_t(1)): find(tt==test_t(end)) ;

x_val = (log(rain_reconstructed(idx_rain_nvdi_val) + 1) - mean_x)./20;
x_test = (log(rain_reconstructed(idx_rain_nvdi_test) + 1) - mean_x)./20;

x_model_val = cat(1, x, x_val);
x_model_val_test = cat(1, x, x_val, x_test);

t_predict_model_val = cat(1,model_t, validation_t);
t_predict_model_val_test = cat(1,model_t, validation_t, test_t);

y_model_val = cat(1, model_nvdi, validation_nvdi);
y_model_val_test = cat(1, model_nvdi, validation_nvdi, test_nvdi);

y_val = y_model_val(idx_validation_nvdi);
y_test = y_model_val_test(idx_test_nvdi);

y_trend_model_val = y_model_val - (t_predict_model_val.*slope + intercept);
y_trend_model_val_test = y_model_val_test - (t_predict_model_val_test.*slope + intercept);

% Predict input
[Fx, Gx] = polydiv( inputModel.C, inputModel.A, k );
xhatk_model_val = filter(Gx, inputModel.C, x_model_val);
xhatk_model_val_test = filter(Gx, inputModel.C, x_model_val_test);

xhatk_val = xhatk_model_val_test(idx_validation_nvdi);
xhatk_test = xhatk_model_val_test(idx_test_nvdi);

ehat_val = x_val - xhatk_val;
ehat_test = x_test - xhatk_test;
ehat_model_val_inp = x_model_val - xhatk_model_val; % need to throw here

t_predict_val = t_predict_model_val_test(idx_validation_nvdi);
t_predict_test = t_predict_model_val_test(idx_test_nvdi);

%modelLim = 400
f = figure();
subplot(211)
plot(t_predict_val, xhatk_val)
title(append(int2str(k), "-step predcition of input on vlaidation data."))
hold on
plot(t_predict_val, x_val)
legend('xhatk', 'x')
subplot(212)
plot(t_predict_val, ehat_val);
legend('ehat')

f = figure();
subplot(211)
plot(t_predict_test, xhatk_test)
hold on
plot(t_predict_test, x_test)
title(append(int2str(k), "-step predcition of input on test data."))
legend('xhatk', 'x')
subplot(212)
plot(t_predict_test, ehat_test);
legend('ehat')

std_xk = sqrt( sum( Fx.^2 )*var_ex ); %NEED TO CHECK THIS
fprintf( 'The theoretical std of the %i-step prediction error is %4.2f.\n', k, std_xk)

%% Form the residual. Input
%ehat = ehat(30:end);
[acfEst, pacfEst] = plotACFnPACF(ehat_model_val_inp(20:end), 75, 'Input model prediction');
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(ehat_model_val_inp(20:end));


%% Proceed to predict the data using the predicted input.
% Form the BJ prediction polynomials. In our notation, these are
%   A1 = foundModel.D
%   C1 = foundModel.C
%   A2 = foundModel.F
% 
% The KA, KB, and KC polynomials are formed as:
%   KA = conv( A1, A2 );
%   KB = conv( A1, B );
%   KC = conv( A2, C1 );
%
KA = conv( MboxJ.D, MboxJ.F );
KB = conv( MboxJ.D, MboxJ.B );
KC = conv( MboxJ.F, MboxJ.C );

% Form the ARMA prediction for y_t (note that this is not the same G
% polynomial as we computed above (that was for x_t, this is for y_t).
[Fy, Gy] = polydiv( MboxJ.C, MboxJ.D, k );

% Compute the \hat\hat{F} and \hat\hat{G} polynomials.
[Fhh, Ghh] = polydiv( conv(Fy, KB), KC, k );

% Form the predicted output signal using the predicted input signal.
yhatk_model_val  = filter(Fhh, 1, xhatk_model_val) + filter(Ghh, KC, x_model_val) + filter(Gy, KC, y_trend_model_val);
yhatk_model_val_test  = filter(Fhh, 1, xhatk_model_val_test) + filter(Ghh, KC, x_model_val_test) + filter(Gy, KC, y_trend_model_val_test);

yhatk_val = yhatk_model_val_test(idx_validation_nvdi) + (slope.*t_predict_val + intercept);
yhatk_test = yhatk_model_val_test(idx_test_nvdi) + (slope.*t_predict_test + intercept);

ehat_val = y_val - yhatk_val;
ehat_test = y_test - yhatk_test;
ehat_model_val = y_model_val - (yhatk_model_val + (slope.*t_predict_model_val + intercept));% Need to throw SHould not use test here, but forget now

var_ehat_val = var(ehat_val);
var_ehat_test = var(ehat_test);
var_ehat_val_norm = var_ehat_val/var(y_val)
var_ehat_test_norm = var_ehat_test/var(y_test)

[acfEst, pacfEst] = plotACFnPACF(ehat_model_val(100:end), 75, append(int2str(k), '-step prediction'));
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(ehat_model_val(50:end) - + (slope.*t_predict_model_val(50:end) + intercept));


naive = naive_vals{k}; 
f = figure();
subplot(211)
plot(t_predict_val, y_val)
hold on
plot(t_predict_val, yhatk_val)
hold on
plot(t_predict_val, naive{1}(:,1))
legend('y', 'yhatk', 'naive')
title(append(int2str(k), "-step predcition of NVDI on validation data."))
subplot(212)
plot(t_predict_val, ehat_val);
hold on
plot(t_predict_val, naive{1}(:,2))
legend('ehat', 'ehat naive')
%title('NVDI: Validation')


f = figure();
subplot(211)
plot(t_predict_test, y_test)
title(append(int2str(k), "-step predcition of NVDI on test data."))
hold on
plot(t_predict_test, yhatk_test)
hold on
plot(t_predict_test, naive{2}(:,1))
legend('y', 'yhatk', 'naive')
subplot(212)
plot(t_predict_test, ehat_test);
hold on
plot(t_predict_test, naive{2}(:,2))
legend('ehat', 'ehat naive')
%title('NVDI: Test')

%% ----------------- C --------------------------
% Estimate the unknown parameters using a Kalman filter and form the predictions.
N = length(t_predict_model_val_test);
y = y_trend_model_val_test;
x = x_model_val_test;

nnz_KB  = find(KB ~= 0);
noPar   = length(nnz_KB) + 1 ;                            % The vector of unknowns is [ -KA(2) -KA(3) KB(1) KB(2) KB(3) KC(3) ]
xt      = zeros(noPar, N);               % Estimated states. Set the initial state to the estimated parameters.
xt(:,max(nnz_KB)-1) = [-KA(2) KB(nnz_KB)];

%%
% For the output
A     = eye(noPar);
Rw    = 5e-4;%std(ehat_model_val);              % Measurement noise covariance matrix, R_w. Try using the noise estimate from the polynomial prediction.
Re    = 7e-7*eye(noPar);                        % System noise covariance matrix, R_e.
Re(1,1) = 7e-5; Re(3,3) = 7e-5; Re(4,4) = 7e-5; Re(13,13) = 5e-5;
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
    % --------- INPUT ----------
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

    % ------------- OUTPUT ----------------
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
yhatk_1 = yhatk_1 + (slope.*t_predict_model_val_test + intercept);
yhatk_7 = yhatk_7(1:end - 7);
yhatk_7 = yhatk_7 + (slope.*t_predict_model_val_test + intercept);
xhatK_inp_4 = xhatK_inp_4(1:end-4);


%% Examine the estimated parameters.
trueParams = [ -KA(2) KB(nnz_KB)];

figure
plotWithConf( 1:N, xt', xStd', trueParams );
axis([startInd-1 N -1 1])
title(sprintf('Estimated parameters, with Re = %7.6f and Rw = %4.3f', Re(1,1), Rw(1,1)))
xlabel('Time')
fprintf('The final values of the Kalman estimated parameters are:\n')
max_5 = zeros(7,2);
for k0=1:length(trueParams)
    %fprintf('  True value: %5.2f, estimated value: %5.2f (+/- %5.4f).\n', trueParams(k0), xt(k0,end), xStd(k0,end) )
    %abs(trueParams(k0)- xt(k0,end))
    if abs(trueParams(k0)- xt(k0,end)) > min(max_5(:,2))
        index = find(min(max_5(:,2)) == max_5(:,2));
        max_5(index(1), 1) = k0;
        max_5(index(1), 2) = abs(trueParams(k0)- xt(k0,end));
    end
    
end 
max_5
%% 1- step
ehat_1 = y_model_val_test - yhatk_1;
ehat_7 = y_model_val_test - yhatk_7;
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
    f = figure;
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
[acf_est, pacf_est] = plotACFnPACF( ehat_1(150: end), 70, '1-step prediction using the Kalman filter');
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(ehat_1(150:end));
plotACFnPACF( ehat_7(150: end), 70, '7-step prediction using the Kalman filter');
%% Plotting input pred 4-step
figure();
plot(t_predict_model_val_test, x)
hold on;
plot(t_predict_model_val_test, xhatK_inp_4);

title("Input prediction using Kalman filter with stationary model.")
legend( 'True', '4-step')
e_inp = x- xhatK_inp_4;
[acfest, pacfest] = plotACFnPACF( e_inp(150: end), 40, '4-step prediction using the Kalman filter');

checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(e_inp(150: end));
%%
% SEE OTHER FILES FOR THE OTHER KALMAN TESTING
% PROBLEM D IS DONE BY RERUNNING RELEVANT SECTION WITH 
% City = Kassala to reestimate params, or with City = ElGeneina
% but Kassala = true without reestimation of params. 