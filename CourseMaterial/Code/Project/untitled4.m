%% Clean up and add path
clear; clc;
close all;
addpath('../functions', '../data')
load proj23.mat
%% Initial Plot
city = ElGeneina;
figure()
area(city.rain_org_t, city.rain_org, 'FaceColor','#4DBEEE')
hold on
pl1 = plot(city.rain_t, city.rain,'*', 'color', 'r','LineWidth',1);
hold on
pl2 = plot(city.rain_org_t, city.rain_org,'*', 'color', '#0072BD','LineWidth',1);
legend([pl2 pl1], {'Meassurements', 'Interpolation'})
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
rain_init = zeros(3,N);
rain_init(1,:) = y./3;  
rain_init(2,:) = y./3;
rain_init(3,:) = y./3;

%% Estimating parameters with Kalman Filter
windowsize = 20;
a1 = -0.5;%0.999999; % Intial AR param estimate

for i = 1:21 % Update Ar 20 time
    if(i == 21) % If on last iteration, use all available data to reconstruct
        y = city.rain_org(1:end);
        N = length(y);
        rain_init = zeros(3,N);
        rain_init(1,:) = y./3;  
        rain_init(2,:) = y./3;
        rain_init(3,:) = y./3;
    end
    A =[-a1.^3 0 0;a1.^2 0 0; -a1 0 0];
    Z = [1 -a1 a1^2; 0 1 -a1; 0 0 1];
    at = zeros(N,1);
    Rw    = 1e-3;                                      % Measurement noise covariance matrix, R_w. Note that Rw has the same dimension as Ry.
    Re    = eye(3)*1e1;%10e-1                        % System noise covariance matrix, R_e. Note that Re has the same dimension as Rx_t1.
    Re = Z*Re*transpose(Z);
    Rx_t1 = eye(3)*1e1; %correct?                             % Initial covariance matrix, V0 = R_{1|0}^{x,x}
    e_hat  = zeros(N,1);                             % Estimated one-step prediction error.
    xt    = rain_init;                         % Estimated states. Intial state, x_{1|0} = 0.
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
            
            % Solve the NNLS problem
            new_data = lsqnonneg(eye(3), xt(:,t));
            
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
    
    xtemp = flip(xt,1);
    data = iddata(xtemp(:)); % All x data, should be AR
    model_init = idpoly([1 a1], [], []) ;
    model_armax_1 = pem(data , model_init);
    present(model_armax_1)
    if a1 == model_armax_1.A(2)
        break % We have converged
    else
    a1 = model_armax_1.A(2) % New Ar parameter, update and loop
    end
end

%% Reconstruct 
rain_rec = flip(xt,1); % DISCUSS THIS
rain_reconstructed = rain_rec(:);
rain_reconstructed = [rain_reconstructed(2:end); 0];
% With Provided time vector
tt = city.rain_t;

std_rec = flip(xStd,1); % DISCUSS THIS
std_reconstructed = std_rec(:);
std_reconstructed = [std_reconstructed(2:end); 0];

%% Plot reconstruction
figure(); 
hold on;
pl1 = plot(tt, rain_reconstructed); 

pl2 = plot(tt(1), rain_reconstructed(1),'*', 'color', 'r');
hold on
pl2 = plot(tt(2), rain_reconstructed(2),'*', 'color', 'r');
hold on
pl3 = plot(tt(2), rain(2),'*', 'color', 'k');
hold on
counter = 2;
for i=3:3:length(rain_reconstructed)-2
    plot(tt(i), rain_reconstructed(i),'*', 'color', 'r')
    hold on 
    plot(tt(i+1), rain_reconstructed(i+1),'*', 'color', 'r')
    hold on
    pl2 = plot(tt(i+2), rain_reconstructed(i+2),'*', 'color', 'r');
    hold on
    pl3 = plot(tt(i+2), rain(counter),'*', 'color', 'k');
    hold on
    counter = counter + 1;
end

legend([pl1 pl2 pl3],{'Reconstructed Line ','Reconstructed Points', 'Original'})
xlabel('Time')
ylabel('Rain')
title('Reconstructed Rain')
%% Other viz for Project
figure()
area(ElGeneina.rain_org_t, ElGeneina.rain_org, 'FaceColor','#9eed34')
hold on
area(tt, rain_reconstructed, 'FaceColor','#4DBEEE')
hold on
plot(tt, rain_reconstructed,'*', 'color', '#0072BD','LineWidth',1)
hold on
plot(ElGeneina.rain_org_t, ElGeneina.rain_org,'*', 'color', '#395c0a','LineWidth',1)
legend('Original Rain', 'Reconstructed Rain')
xlabel('Time')
ylabel('Rain')
xlim([tt(13),tt(30)])
title('Reconstructed Rain')
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
%% Plot
nvdi = 2.*(city.nvdi./255) -1; %log?

nvdi_t = city.nvdi_t;
figure();
plot(nvdi_t, nvdi)
xlabel('Time')
ylabel('NVDI')
title('NVDI Data and Split')

plotACFnPACF(nvdi, 100, 'data');

%%
%idx_model_nvdi = idx_model_nvdi(115:end);
model_t = nvdi_t(idx_model_nvdi);
%model_t = model_t(125:end);
validation_t = nvdi_t(idx_validation_nvdi);
test_t = nvdi_t(idx_test_nvdi);

model_nvdi = nvdi(idx_model_nvdi);
%model_nvdi = model_nvdi(125:end);
validation_nvdi = nvdi(idx_validation_nvdi);
test_nvdi = nvdi(idx_test_nvdi);

figure();
plot(model_t, model_nvdi)
hold on
plot([model_t(end); validation_t], [model_nvdi(end); validation_nvdi])
hold on;
plot([validation_t(end); test_t], [validation_nvdi(end); test_nvdi])
legend('Model', 'Validation', 'Test')
xlabel('Time')
ylabel('NVDI')
title('NVDI Data and Split')

figure; 
lambda_max = bcNormPlot(model_nvdi,1)
%% Remove deterministic trend WRONG ONLY ON MODEL DATA
figure();
plot(model_t, model_nvdi)

mdl = fitlm(model_t, model_nvdi)
intercept = mdl.Coefficients(1,1).Estimate;
slope = mdl.Coefficients(2,1).Estimate;
nvdi_trend = model_nvdi- (model_t.*slope + intercept);
figure();
plot(model_t, nvdi_trend)
plotACFnPACF(nvdi_trend, 40, 'data');
figure; 
lambda_max = bcNormPlot(nvdi_trend,1);

outlier = [0.0802; %KALMANNNNN
    0.0986;
    0.1280;
    0.1342;
    0.1628;
    0.1701;
    0.1914;
    0.1959;
    0.1999;
    0.1943]; %CHEATing????
model_nvdi(93:102) = outlier;
nvdi_trend(93:102) = outlier - (model_t(93:102).*slope + intercept); 
nvdi(93:102) = outlier;

%% Remove season of 36
% A36 = [1 zeros(1, 35) -1]; 
% nvdi_trend = filter(A36, 1, nvdi_trend);
% nvdi_trend = nvdi_trend(length(A36) : end );
% nvdi_t = nvdi_t(length(A36) : end);
% figure();
% plot(nvdi_t, nvdi_season)
% plotACFnPACF(nvdi_season, 40, 'data');
% figure; 
% lambda_max = bcNormPlot(nvdi_season,1);
%% Ar(1) 
plotACFnPACF(nvdi_trend, 100, 'AR(1)' );
data = iddata(nvdi_trend);
Am = [1 1];

model_init = idpoly(Am, [], []) ;
model_armax_1 = pem(data , model_init);

e_hat = filter(model_armax_1.A , model_armax_1.C , nvdi_trend) ;
e_hat = e_hat(length(model_armax_1.A ) : end ) ;

present(model_armax_1)
[acfEst, pacfEst] = plotACFnPACF(e_hat, 100, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(e_hat);
%% a1 + a36 
data = iddata(nvdi_trend);
Am = [1 1 zeros(1,34) 1];
model_init = idpoly(Am, [], []) ;
model_init.Structure.a.Free = Am;
model_armax_1 = pem(data , model_init);

e_hat = filter(model_armax_1.A , model_armax_1.C , nvdi_trend) ;
e_hat = e_hat(length(model_armax_1.A ) : end ) ;

present(model_armax_1)
[acfEst, pacfEst] = plotACFnPACF(e_hat, 100, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(e_hat);
 
%% Ar(1) with (1+a36 z^-36)
data = iddata(nvdi_trend);
Am = conv([1 1], [1 zeros(1, 35), -1]);
Am = [Am zeros(1, 34) 1]; % ehhh
%Cm = [1 0 1];%[1 0 1 1];
Cm = [];
model_init = idpoly(Am, [], Cm);
model_init.Structure.a.Free = Am;
%model_init.Structure.c.Free = Cm;
model_armax_1 = pem(data , model_init);

e_hat = filter(model_armax_1.A , model_armax_1.C , nvdi_trend) ;
e_hat = e_hat(length(model_armax_1.A ) : end ) ;

present(model_armax_1)
[acfEst, pacfEst] = plotACFnPACF(e_hat, 100, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(e_hat);

%% PREDICTION: DON'T FORGET TO TRANSFORM TO ORG DOMAIN
% Should take extracare of throwing here!
ks = [1 7];
naive_vals = {};
for i=1:2
    t_predict = cat(1,model_t, validation_t);
    y = cat(1, model_nvdi, validation_nvdi);
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

        y_naive = y_naive(length(model_t)+1:end);
        ehat_naive = ehat_naive(length(model_t)+1:length(t_predict));
        naive_vals{1} = [y_naive' ehat_naive];
    else % 7 - step: Use last years value
        y_naive = zeros(1, length(y) + 36 );
        y_naive(37: end) = y;
        y_naive = y_naive(1:end - 36);
        ehat_naive = y-y_naive';

        y_naive = y_naive(length(model_t)+1:end);
        ehat_naive = ehat_naive(length(model_t)+1:length(t_predict));
        naive_vals{7} = [y_naive' ehat_naive];

    end
    
    % Predict
    [Fk, Gk] = polydiv(C, A, k) ;
    yhatk = filter(Gk, C, y_trend) ;
    yhatk = yhatk + (t_predict.*slope + intercept); % Add trend
    ehat = y-yhatk;

    yhatk = yhatk(length(model_t)+1:end);
    ehat = ehat(length(model_t)+1:end);

    t_predict = t_predict(length(model_t)+1:end);
    y = y(length(model_t)+1:end);
    figure()
    subplot(211)
    plot(t_predict,y)
    hold on
    plot(t_predict, yhatk)
    hold on
    plot(t_predict, y_naive)
    title('NVDI')
    legend('True data', append(int2str(k), '-step prediction'), 'Naive')

    % Form the prediction error and examine the ACF. Note that the prediction
    % residual should only be white if k=1. 
    subplot(212)
    plot(t_predict, ehat)
    var_ehat = var(ehat)./var(y)

    hold on
    plot(t_predict, ehat_naive)
    var_ehat_naive = var(ehat_naive)./var(y)
    legend('ehat = y-yhatk', 'naive')
    plotACFnPACF(ehat, 40, append(int2str(k), '-step prediction'));
    if(i == 1)
        var_noise = var(ehat);
    end
    mean_ehat = mean(ehat) ;
    var_theoretical = (norm(Fk).^2) .*var_noise;
    var_est = var(ehat);
    conf =0 + [-1 1].*norminv(0.975).*sqrt(var_theoretical);
    precentage_outside = (sum(ehat > conf(2)) + sum(ehat < conf(1)))./length(ehat);
end

%% Examine the data.
 % Extract rain for modelling
idx_rain_nvdimodel_dates = find(tt==model_t(1)): find(tt==model_t(1)) + length(nvdi_trend) - 1;
x = rain_reconstructed(idx_rain_nvdimodel_dates);
%x = rain_temp(idx_rain_nvdimodel_dates);
window = 7;

% area = zeros(length(x), 1);
% xtemp = [zeros(window-1,1); x];
% for i=1:length(area)
%     area(i) = sum(xtemp(i:i+window-1));
% end
%x = area;
y = nvdi_trend; 
% Transform
x = log(x+1);
mean_x = mean(x);
x = (x - mean_x)./30;

% manual inspection
figure; 
subplot(311);
plot(x); 
ylabel('Input signal')
title('Measured signals')

subplot(312);
%plot(area)
subplot(313); 
plot( y ); 
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
% Am = conv([1 1 1], [1 zeros(1, 35) 1]);%[1 1 1 zeros(1, 33) 1];
% %Am = [Am zeros(1, 37) 1];
% Cm = [1 zeros(1, 75) 0.1];%Cm = [1 0 0 1];

%idx = [1 2 7 36 37 38];
idx = [1 36 37];
%idx = [1 2 3 36];
arr = zeros(1, idx(end) +1);
arr(idx +1 ) = 1;
Am = arr.*0.3;
Am(1) =1;

idx = [36] ;
arr = zeros(1, idx(end) +1);
arr(idx +1 ) = 1;
Cm = arr.*0.3;
Cm(1) =1;
%Cm = [];
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
[acfEst, pacfEst] = plotACFnPACF(w_t, 150, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(w_t);

A = inputModel.A;
C = inputModel.C;
%%
n = length(x);
M = 100;
figure()
[Cxy,lags] = xcorr(eps_t, w_t, M, 'coeff' ); %(d,r,s) = (4,2,0)
stem(lags, Cxy)
title ('Cross_correlation_function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off

%%
d=7; r=0; s=0; 
A2 = [1 0.01.*ones(1, r)];
B = [zeros(1,d) 0.1.*ones(1, s+1)];
%B = [0 0 0 1 0 0 1 1 1 zeros(1,22) 1 1 1 ];

idx = [3 5 7 17 31 34];
%idx = [1 7 28 ];
%idx = [2 12 20 30];
%idx = [3 6 15 19 34];
%idx = [3 5 17 34 79];

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
title ('Cross_correlation_function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off

%% Model etilde
figure();
plot(etilde);
title('etilde data');
plotACFnPACF(etilde, M, 'Input Data' );

A2_diff = [1 0 -1];
Cm = [1 1];%conv([1 zeros(1, 35) 1], A2_diff);
model_init = idpoly([1 1], [], []) ;
%model_init.Structure.a.Free = Am;
%model_init.Structure.c.Free = Cm;
%model_init = idpoly([1 1 1], [], []) ;
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
ehat = ehat(length(MboxJ): end);

[acfEst, pacfEst] = plotACFnPACF(ehat, M, 'Box-Jenkins');
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(ehat);

figure()
[Cxy,lags] = xcorr(ehat, x, M, 'coeff' );
stem(lags, Cxy)
title ('Cross_correlation_function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off
%%
k =1;
idx_rain_nvdi_val = find(tt==validation_t(1)): find(tt==validation_t(end)) ;
idx_rain_nvdi_test = find(tt==test_t(1)): find(tt==test_t(end)) ;

% window = 6;
% area = zeros(length(rain_reconstructed), 1);
% xtemp = [zeros(window-1,1); (rain_reconstructed -mean_x)./50];
% for i=1:length(area)
%     area(i) = sum(xtemp(i:i+window-1));
% end
% 
% x_val = area(idx_rain_nvdi_val);
% x_test = area(idx_rain_nvdi_test) ;
x_val = (log(rain_reconstructed(idx_rain_nvdi_val) + 1) - mean_x)./30;
x_test = (log(rain_reconstructed(idx_rain_nvdi_test) + 1) - mean_x)./30;

x_model_val = cat(1, x, x_val);
x_model_val_test = cat(1, x, x_val, x_test);

t_predict_model_val = cat(1,model_t, validation_t);
t_predict_model_val_test = cat(1,model_t, validation_t, test_t);

y_model_val = cat(1, model_nvdi, validation_nvdi);
y_model_val_test = cat(1, model_nvdi, validation_nvdi, test_nvdi);

y_val = y_model_val(length(model_nvdi)+1:end);
y_test = y_model_val_test(length(model_nvdi) + length(validation_nvdi) + 1:end);

y_trend_model_val = y_model_val - (t_predict_model_val.*slope + intercept);
y_trend_model_val_test = y_model_val_test - (t_predict_model_val_test.*slope + intercept);

[Fx, Gx] = polydiv( inputModel.C, inputModel.A, k );
xhatk_model_val = filter(Gx, inputModel.C, x_model_val);
xhatk_model_val_test = filter(Gx, inputModel.C, x_model_val_test);

xhatk_val = xhatk_model_val(length(model_nvdi)+1: end);
xhatk_test = xhatk_model_val_test(length(model_nvdi) + length(validation_nvdi) + 1: end);

ehat_val = x_val - xhatk_val;
ehat_test = x_test - xhatk_test;
ehat_model_val_inp = x_model_val - xhatk_model_val; % need to throw here

t_predict_val = t_predict_model_val(length(model_nvdi) + 1 : end);
t_predict_test = t_predict_model_val_test(length(model_nvdi) + length(validation_nvdi) + 1: end);

%modelLim = 400
figure()
subplot(211)
plot(t_predict_val, xhatk_val)
hold on
plot(t_predict_val, x_val)
legend('xhatk', 'x')
subplot(212)
plot(t_predict_val, ehat_val);
legend('ehat')

figure()
subplot(211)
plot(t_predict_test, xhatk_test)
hold on
plot(t_predict_test, x_test)
legend('xhatk', 'x')
subplot(212)
plot(t_predict_test, ehat_test);
legend('ehat')

std_xk = sqrt( sum( Fx.^2 )*var_ex ); %NEED TO CHECK THIS
fprintf( 'The theoretical std of the %i-step prediction error is %4.2f.\n', k, std_xk)

%% Form the residual. Is it behaving as expected? Recall, no shift here!
%ehat = ehat(30:end);

figure
acf( ehat_val, 40, 0.05, 1 );
title( sprintf('ACF of the %i-step input prediction residual', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( ehat_val );
pacfEst = pacf( ehat_val, 40, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );

figure
acf( ehat_test, 40, 0.05, 1 );
title( sprintf('ACF of the %i-step input prediction residual', k) )
fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
checkIfWhite( ehat_test );
pacfEst = pacf( ehat_test, 40, 0.05 );
checkIfNormal( pacfEst(k+1:end), 'PACF' );
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

yhatk_val = yhatk_model_val(length(model_nvdi)+1:end) + (slope.*t_predict_val + intercept);
yhatk_test = yhatk_model_val_test(length(model_nvdi) + length(validation_nvdi) + 1:end) + (slope.*t_predict_test + intercept);

ehat_val = y_val - yhatk_val;
ehat_test = y_test - yhatk_test;
ehat_model_val = y_model_val - yhatk_model_val + (slope.*t_predict_model_val + intercept);% Need to throw SHould not use test here, but forget now

var_ehat_val = var(ehat_val);
var_ehat_test = var(ehat_test);
var_ehat_val_n = var_ehat_val/var(y_val)
var_ehat_test_n = var_ehat_test/var(y_test)

% A very common error is to forget to add the predicted inputs. Lets try
% that to see what happens.
% yhatk  = filter(Ghh, KC, x) + filter(Gy, KC, y);
naive = naive_vals{k}; % NEED TO IMPLEMENT NAIVE FOR TEST
figure()
subplot(211)
plot(t_predict_val, y_val)
hold on
plot(t_predict_val, yhatk_val)
hold on
plot(t_predict_val, naive(:,1))
legend('y', 'yhatk', 'naive')
subplot(212)
plot(t_predict_val, ehat_val.^2);
hold on
plot(t_predict_val, naive(:,2).^2)
legend('ehat', 'ehat naive')

figure()
subplot(211)
plot(t_predict_test, y_test)
hold on
plot(t_predict_test, yhatk_test)
hold on
%plot(t_predict_test, naive(:,1))
legend('y', 'yhatk', 'naive')
subplot(212)
plot(t_predict_test, ehat_test.^2);
hold on
%plot(t_predict_test, naive(:,2).^2)
legend('ehat', 'ehat naive')
%line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' )

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
N = length(t_predict_model_val);
y = y_trend_model_val;
%y(93:102) = nan;
%y1 = y;
x = x_model_val;
codeVersion = 1;
modelLim = 100;
switch codeVersion
    case 1
        nnz_KB  = find(KB ~= 0);
        noPar   = length(nnz_KB) + 1+2;                            % The vector of unknowns is [ -KA(2) -KA(3) KB(1) KB(2) KB(3) KC(3) ]
        xt      = zeros(noPar, N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,max(nnz_KB)-1) = [ -KA(2) 0.5 KB(nnz_KB) 0.5 ];
    case 2
        noPar   = 3;                            % The vector of unknowns is [ -KA(2) -KA(3) KC(3) ]
        xt      = zeros(noPar,N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,2) = [ -KA(2) -KA(3) KC(3) ];
    case 3
        noPar   = 2;                            % The vector of unknowns is [ -KA(2) -KA(3)  ]
        xt      = zeros(noPar,N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,2) = [ -KA(2) -KA(3) ];
end
%%
A     = eye(noPar);
Rw    = std(ehat_model_val).^2;                 % Measurement noise covariance matrix, R_w. Try using the noise estimate from the polynomial prediction.
Re    = 5e-6*eye(noPar);                        % System noise covariance matrix, R_e.
Re(2,2)    = 1e-4; 
Re(noPar,noPar)    = 1e-4; 
Rx_t1 = 1e-6*eye(noPar);                        % Initial covariance matrix, R_{1|0}^{x,x}
Rx_k  = Rx_t1;
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
yhatK = zeros(N,1);                             % Estimated output.
xStd  = zeros(noPar,N);                         % Stores one std for the one-step prediction.
startInd = max(nnz_KB);                                   % We use t-2, so start at t=3.

for t=startInd:N
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    switch codeVersion
        case 1                                  % Estimate all parameters.
            C = [ y(t-1)  y(t-2) x(t - nnz_KB + 1)' h_et(t-1)];
            % C = [yhatK_1(t) x(t - nnz_KB + 2)'];--> yhatK_2
            % C = [yhatK_2(t+1) x(t - nnz_KB + 3)']; ---> yhatK_3
            % ----- xhatk
            % C = [yhatK_3(t+2) xhatk_1 x(t - nnz_KB(2:end) + 4)'];
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
    
    if isnan( y(t) )
        xt(:,t) = x_t1;                         % x_{t|t} = x_{t|t-1} 
        Rx_t    = Rx_t1;                        % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} 
        y1(t)   = yhatK(t);                      % Replace the missing sample with the estimated value. 
        t
    else
        h_et(t) = y(t)-yhatK(t);                    % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
        xt(:,t) = x_t1 + Kt*( h_et(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 
    
        % Update the covariance matrix estimates.
        Rx_t  = Rx_t1 - Kt*Ry*Kt';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    end
    Rx_t1 = A*Rx_t*A' + Re;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Estimate a one std confidence interval of the estimated parameters.
    xStd(:,t) = sqrt( diag(Rx_t) );             % This is one std for each of the parameters for the one-step prediction.
end



%% Examine the estimated parameters.
% Compute the true parameters for the KA, KB, and KC polynomials.
% KA0 = conv( A1, A2 );
% KB0 = conv( A1, B );
% KC0 = conv( A2, C1 );
switch codeVersion
    case 1
        trueParams = [ -KA(2) KB(nnz_KB) 0.5];
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

yhatK = yhatK + (slope.*t_predict_model_val + intercept);
%% Show the one-step prediction. 
figure
plot( [y_model_val yhatK] )
title('One-step prediction of the validation data')
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Polynomial estimate', 'Location','SW')
xlim([0 N])

% Form the prediction residuals for the validation data.
eK = y_model_val(length(model_nvdi)+1:end) - yhatK(length(model_nvdi)+1:end);
var(eK)/var(y_model_val(length(model_nvdi)+1:end))
eK = y(100:end)-yhatK(100:end);
%plotACFnPACF( eP, 40, 'One-step prediction using the polynomial estimate');
plotACFnPACF( eK, 40, 'One-step prediction using the Kalman filter'); %WRONG BECAUSE TREND
%fprintf('The variance of the validation data is               %7.2f.\n', var(y(modelLim:end)))
%fprintf('The estimated variance of the polynomial estimate is %7.2f.\n', var(eP))
%fprintf('The estimated variance of the Kalman estimate is     %7.2f.\n', var(eK))

%% K = 7
%%%% Estimate the unknown parameters using a Kalman filter and form the one-step prediction.
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
N = length(t_predict_model_val);
y = y_trend_model_val;
x = x_model_val;
codeVersion = 1;
modelLim = 100;
switch codeVersion
    case 1
        nnz_KB  = find(KB ~= 0);
        noPar   = length(nnz_KB) + 1 ;                            % The vector of unknowns is [ -KA(2) -KA(3) KB(1) KB(2) KB(3) KC(3) ]
        xt      = zeros(noPar, N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,max(nnz_KB)-1) = [ -KA(2) KB(nnz_KB) ];
    case 2
        noPar   = 3;                            % The vector of unknowns is [ -KA(2) -KA(3) KC(3) ]
        xt      = zeros(noPar,N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,2) = [ -KA(2) -KA(3) KC(3) ];
    case 3
        noPar   = 2;                            % The vector of unknowns is [ -KA(2) -KA(3)  ]
        xt      = zeros(noPar,N);               % Estimated states. Set the initial state to the estimated parameters.
        xt(:,2) = [ -KA(2) -KA(3) ];
end
%%
A     = eye(noPar);
Rw    = std(ehat_model_val).^2;                                % Measurement noise covariance matrix, R_w. Try using the noise estimate from the polynomial prediction.
Re    = 1e-5*eye(noPar);                        % System noise covariance matrix, R_e.
Rx_t1 = 1e-5*eye(noPar);                        % Initial covariance matrix, R_{1|0}^{x,x}
Rx_k  = Rx_t1;
h_et  = zeros(N,1);                             % Estimated one-step prediction error.
yhatK = zeros(N,1);                             % Estimated output.
xStd  = zeros(noPar,N);                         % Stores one std for the one-step prediction.
yhatk_7 = zeros(N,1);  

noPar_inp = length(inputModel.A) - 1;
A_inp     = zeros(noPar_inp);
A_inp(1, :) = - inputModel.A(2:end);
A_inp(2: noPar_inp+1:end) = 1;

xt_inp      = zeros(noPar_inp, N);
Rw_inp    = 1e-1;%std(ehat_model_val_inp);                                % Measurement noise covariance matrix, R_w. Try using the noise estimate from the polynomial prediction.
Re_inp    = 1e-1*eye(noPar_inp);                        % System noise covariance matrix, R_e.
Rx_t1_inp = 1e-1*eye(noPar_inp);                        % Initial covariance matrix, R_{1|0}^{x,x}
Rx_k_inp  = Rx_t1_inp;
h_et_inp  = zeros(N,1);                             % Estimated one-step prediction error.
xhatK_inp = zeros(N,1);                             % Estimated output.
xhatK_inp_4 = zeros(N,1); % testing
xStd_inp  = zeros(noPar_inp,N);                         % Stores one std for the one-step prediction.

startInd = max(nnz_KB);                                   % We use t-2, so start at t=3.

for t=startInd:N
    % Input
    % Update the predicted state and the time-varying state vector.
    x_t1_inp = A_inp*xt_inp(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}
    
    C_inp = [inputModel.C zeros(1, noPar_inp - length(inputModel.C))];
    % C = [yhatK_1(t) x(t - nnz_KB + 2)'];--> yhatK_2
    % C = [yhatK_2(t+1) x(t - nnz_KB + 3)']; ---> yhatK_3
    % ----- xhatk
    % C = [yhatK_3(t+2) xhatk_1 x(t - nnz_KB(2:end) + 4)'];
    xhatK_inp(t) = C_inp*x_t1_inp;

    % Update the parameter estimates.
    Ry_inp = C_inp*Rx_t1_inp*C_inp' + Rw_inp;                       % R_{t|t-1}^{y,y} = C R_{t|t-1}^{x,x} + Rw
    Kt_inp = Rx_t1_inp*C_inp'/Ry_inp;                           % K_t = R^{x,x}_{t|t-1} C^T inv( R_{t|t-1}^{y,y} )
    h_et_inp(t) = x(t)-xhatK_inp(t);                    % One-step prediction error, \hat{e}_t = y_t - \hat{y}_{t|t-1}
    xt_inp(:,t) = x_t1_inp + Kt_inp*( h_et_inp(t) );            % x_{t|t}= x_{t|t-1} + K_t ( y_t - Cx_{t|t-1} ) 

    % Update the covariance matrix estimates.
    Rx_t_inp  = Rx_t1_inp - Kt_inp*Ry_inp*Kt_inp';                  % R^{x,x}_{t|t} = R^{x,x}_{t|t-1} - K_t R_{t|t-1}^{y,y} K_t^T
    Rx_t1_inp = A_inp*Rx_t_inp*A_inp' + Re_inp;                     % R^{x,x}_{t+1|t} = A R^{x,x}_{t|t} A^T + Re

    % Estimate a one std confidence interval of the estimated parameters.
    xStd_inp(:,t) = sqrt( diag(Rx_t_inp) );             % This is one std for each of the parameters for the one-step prediction.
    
    % 2 - 4 step predictions
    k= 4;
    Rx_k_inp = Rx_t1_inp;
    ksteps_inp = [xhatK_inp(t)];
    for k0=2:k
        Ck = C_inp;%[ -yk -y(t-6+k0) -y(t-7+k0) h_et(t+k0-1) h_et(t+k0-3) ]; % C_{t+k|t}
        xk = Ck*A_inp^k*xt_inp(:,t-1);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        Rx_k_inp = A_inp*Rx_k_inp*A_inp' + Re_inp;                  % R_{t+k+1|t}^{x,x} = A R_{t+k|t}^{x,x} A^T + Re  
        ksteps_inp(end + 1) = xk;
    end

    xhatK_inp_4(t - 1 + k) = xk;
    %yhatk(t - 1 +k) = yk;                            % Note that this should be stored at t+k.


    % Output
    % Update the predicted state and the time-varying state vector.
    x_t1 = A*xt(:,t-1);                         % x_{t|t-1} = A x_{t-1|t-1}

    C = [ y(t-1) x(t - nnz_KB + 1)'];
    % C = [yhatK_1(t) x(t - nnz_KB + 2)'];--> yhatK_2
    % C = [yhatK_2(t+1) x(t - nnz_KB + 3)']; ---> yhatK_3
    % ----- xhatk
    % C = [yhatK_3(t+2) xhatk_1 x(t - nnz_KB(2:end) + 4)'];
    yhatK(t) = C*x_t1;

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
    
    k= 7;
    Rx_k = Rx_t1;
    yk = yhatK(t);
    for k0=2:k
        if (k0 > 3)
            Ck = [yk ksteps_inp(1:k0 - 3) x(t - nnz_KB(k0 - 3 +1:end) + k0)']; % C_{t+k|t}
        else
            Ck = [yk x(t - nnz_KB + k0)']; % C_{t+k|t}
  
        end
        yk = Ck*A^k*xt(:,t-1);                    % \hat{y}_{t+k|t} = C_{t+k|t} A^k x_{t|t}
        Rx_k = A*Rx_k*A' + Re;                  % R_{t+k+1|t}^{x,x} = A R_{t+k|t}^{x,x} A^T + Re  
    end
    yhatk_7(t - 1 +k) = yk;  
end
yhatk_7 = yhatk_7(1:end-(k-1));
yhatk_7 = yhatk_7 + (slope.*t_predict_model_val + intercept);


%% Examine the estimated parameters.
% Compute the true parameters for the KA, KB, and KC polynomials.
% KA0 = conv( A1, A2 );
% KB0 = conv( A1, B );
% KC0 = conv( A2, C1 );
switch codeVersion
    case 1
        trueParams = [ -KA(2) KB(nnz_KB)];
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


% Show the one-step prediction. 
%xhatK_inp_4 = xhatK_inp_4(1:end-(k-1));

figure
plot( [y_model_val yhatk_7 ] )
title('One-step prediction of the validation data')
xlabel('Time')
legend('Realisation', 'Kalman estimate', 'Polynomial estimate', 'Location','SW')
xlim([0 N])

% Form the prediction residuals for the validation data.
eK = y_model_val(length(model_nvdi)+1:end)-yhatk_7(length(model_nvdi)+1:end);
var(eK)./var(y_model_val(length(model_nvdi)+1:end))


eK = y(100+1:end)-yhatk_7(100+1:end);
%plotACFnPACF( eP, 40, 'One-step prediction using the polynomial estimate');
plotACFnPACF( eK, 40, 'One-step prediction using the Kalman filter'); %WRONG BECAUSE TREND
% fprintf('The variance of the validation data is               %7.2f.\n', var(y(modelLim:end)))
% fprintf('The estimated variance of the polynomial estimate is %7.2f.\n', var(eP))
% fprintf('The estimated variance of the Kalman estimate is     %7.2f.\n', var(eK))
%%
%zero = true;
counter = 1;
rain_temp = zeros(1, length(rain_reconstructed));
while counter < length(rain_reconstructed) +1
    if(rain_reconstructed(counter)>0)
        for i =1:7
            rain_temp(counter) = rain_reconstructed(counter);
            counter = counter + 1;
        end
        while(rain_reconstructed(counter)>0)
            rain_temp(counter) = 0;
            counter = counter + 1;
        end
    else
        counter = counter + 1;
    end
end
rain_temp = rain_temp';