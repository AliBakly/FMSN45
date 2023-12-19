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
rain = ElGeneina.rain_org;
y = rain;
N = length(rain);
rain_init = zeros(3,N);
rain_init(1,:) = rain./3;  
rain_init(2,:) = rain./3;
rain_init(3,:) = rain./3;
%% Estimating parameters with Kalman Filter
windowsize = 20;
a1 = -0.8; % Intial AR param estimate

%for i = 1:20 % Update Ar 20 time
    %A =[-a1 0 0;1 0 0; 0 1 0];
    at = zeros(N,1);
    at(1) = a1;
    Rw    = 10e-3;                                      % Measurement noise covariance matrix, R_w. Note that Rw has the same dimension as Ry.
    Re    = eye(3)*10e-1;                        % System noise covariance matrix, R_e. Note that Re has the same dimension as Rx_t1.
    Rx_t1 = eye(3); %correct?                             % Initial covariance matrix, V0 = R_{1|0}^{x,x}
    e_hat  = zeros(N,1);                             % Estimated one-step prediction error.
    xt    = rain_init;                         % Estimated states. Intial state, x_{1|0} = 0.
    yhat  = zeros(N,1);                             % Estimated output.
    xStd  = zeros(3,N);                         % Stores one std for the one-step prediction.
    for t=2 :N                                       
        % Update the predicted state and the time-varying state vector.
        A =[-a1 0 0;1 0 0; 0 1 0];
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
        
        x_temp = xt(:,1:t);

        if(t > windowsize)
            x_temp = xt(:,t-windowsize:t);
        end
        x_temp = flip(x_temp,1);
        data = iddata(x_temp(:)); % All x data, should be AR
        model_init = idpoly([1 at(t-1)], [], []) ;
        model_armax_1 = pem(data , model_init);
        %present(model_armax_1)
        a1 = model_armax_1.A(2) % New Ar parameter, update and loop
        at(t) = a1;
    end

    % data = iddata(xt(:)); % All x data, should be AR
    % model_init = idpoly([1 a1], [], []) ;
    % model_armax_1 = pem(data , model_init);
    % present(model_armax_1)
    % a1 = -model_armax_1.A(2) % New Ar parameter, update and loop
%end
%% Reconstruct 
rain_rec = flip(xt,1); % DISCUSS THIS
rain_reconstructed = rain_rec(:);

% Our own time vector
tt = zeros(1440, 1);
tt(2:end) = ElGeneina.rain_t(1:end-1);
tt(1) = tt(2) - (tt(3)- tt(2));

% Or do this
% rain_rec = flip(xt,1); % DISCUSS THIS
% rain_reconstructed = rain_rec(:);
% rain_reconstructed = [rain_reconstructed(2:end); 0];
% % With Provided time vector
% tt = ElGeneina.rain_t;

%%
% Plot everything
figure(); 
hold on;
pl1 = plot(tt, rain_reconstructed); 

counter = 1;

for i=1:3:length(rain_reconstructed)-2
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

% % Or this if using provided time vector
% figure(); 
% hold on;
% pl1 = plot(tt, rain_reconstructed); 
% 
% 
% pl2 = plot(tt(1), rain_reconstructed(1),'*', 'color', 'r');
% hold on
% pl2 = plot(tt(2), rain_reconstructed(2),'*', 'color', 'r');
% hold on
% pl3 = plot(tt(2), rain(2),'*', 'color', 'k');
% hold on
% counter = 2;
% for i=3:3:length(rain_reconstructed)-2
%     plot(tt(i), rain_reconstructed(i),'*', 'color', 'r')
%     hold on 
%     plot(tt(i+1), rain_reconstructed(i+1),'*', 'color', 'r')
%     hold on
%     pl2 = plot(tt(i+2), rain_reconstructed(i+2),'*', 'color', 'r');
%     hold on
%     pl3 = plot(tt(i+2), rain(counter),'*', 'color', 'k');
%     hold on
%     counter = counter + 1;
% end
% 
% legend([pl1 pl2 pl3],{'Reconstructed Line ','Reconstructed Points', 'Original'})

%% Compare sums
sums = zeros(480, 1);
counter = 1;
for i=1:3:length(rain_reconstructed)-2
    sums(counter) = rain_reconstructed(i) + rain_reconstructed(i+1) + rain_reconstructed(i+2);
    counter = counter +1;
end

% % Or if using provided time vector
% sums = zeros(480, 1);
% sums(1) = 0;
% counter = 2;
% for i=3:3:length(rain_reconstructed)-2
%     sums(counter) = rain_reconstructed(i) + rain_reconstructed(i+1) + rain_reconstructed(i+2);
%     counter = counter +1;
% end
%% Plot
nvdi = ElGeneina.nvdi;
nvdi_t = ElGeneina.nvdi_t;
figure();
plot(nvdi_t, nvdi)

plotACFnPACF(nvdi, 40, 'data');
figure; 
lambda_max = bcNormPlot(nvdi,1);
%% Remove deterministic trend
mdl = fitlm(nvdi_t, nvdi);
m= mdl.Coefficients(1,1).Estimate;
k= mdl.Coefficients(2,1).Estimate;
nvdi_trend = nvdi- (nvdi_t.*k +m);
figure();
plot(nvdi_t, nvdi_trend)
plotACFnPACF(nvdi_trend, 40, 'data');
figure; 
lambda_max = bcNormPlot(nvdi_trend,1);
%% Split data 70-20-10
figure();
plot(nvdi_t, nvdi_trend)

model_t = nvdi_t(1:453);
validation_t = nvdi_t(454:583);
test_t = nvdi_t(584:648);

model_nvdi = nvdi_trend(1:453);
validation_nvdi = nvdi_trend(454:583);
test_nvdi = nvdi_trend(584:648);
%% Ar(1)
data = iddata(model_nvdi);
model_init = idpoly([1 1 ], [], [1 zeros(1,35) 1]) ;
model_init.Structure.c.Free = [1 zeros(1,35) 1] ;
model_armax_1 = pem(data , model_init);

e_hat = filter(model_armax_1.A , model_armax_1.C , model_nvdi) ;
e_hat = e_hat(length(model_armax_1.A ) : end ) ;

present(model_armax_1)
[acfEst, pacfEst] = plotACFnPACF(e_hat, 45, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(e_hat);
%% Predict
y = validation_nvdi;
A = model_armax_1.A;
C = model_armax_1.C;
ks = [1 3 5 7];

for i=1:4
    
    k = ks(i);
    [Fk, Gk] = polydiv(C, A, k) ;
    yhatk = filter(Gk, C, y) ;
    ehat = y-yhatk;
    ehat = ehat(k+20:end);  % Remove the corrupted samples. You might need to add a bit of a margin.

    figure()
    subplot(211)
    plot(validation_t,y)
    hold on
    plot(validation_t, yhatk)
    title('NVDI')
    legend('True data', append(int2str(k), '-step prediction'))

    % Form the prediction error and examine the ACF. Note that the prediction
    % residual should only be white if k=1. 
    subplot(212)
    plot(ehat)
    legend('ehat = y-yhatk')
    plotACFnPACF(ehat, 40, append(int2str(k), '-step prediction'));

    if(i>1)
        mean_ehat = mean(ehat) 
        var_theoretical = (norm(Fk).^2) .*var_noise
        var_est = var(ehat)
        conf =0 + [-1 1].*norminv(0.975).*sqrt(var_theoretical)
        precentage_outside = (sum(ehat > conf(2)) + sum(ehat < conf(1)))./length(ehat)
    else
        var_noise = var(ehat);
    end
end

%% Examine the data.
figure; 
subplot(211); 
x = rain_reconstructed(794: 793+ length(model_nvdi));
y = model_nvdi;
plot(x); % manual inspection
ylabel('Input signal')
title('Measured signals')
subplot(212); 
plot( y ); 
ylabel('Output signal')
xlabel('Time')

figure
[Cxy,lags] = xcorr( y, x, 50, 'coeff' );
stem( lags, Cxy )
hold on
condInt = 2*ones(1,length(lags))./sqrt( length(y) );
plot( lags, condInt,'r--' )
plot( lags, -condInt,'r--' )
hold off
xlabel('Lag')
ylabel('Amplitude')
title('Crosscorrelation between in- and output')
%%
% Load the data (replace with the path to your .mat file)

rain_data = ElGeneina.rain_org;

% Number of months and days
num_months = length(rain_data);
num_days = num_months * 3; % 3 ten-day intervals per month

% Kalman Filter Initialization
phi = 0.7; % AR(1) coefficient, adjust based on your data
Q = 1; % Process noise covariance, adjust based on your data
R = 1; % Measurement noise covariance, adjust based on your data
P = 1; % Initial estimate of state covariance
x_est = 0; % Initial state estimate

% Storage for estimated values
x_est_store = zeros(num_days, 1);

% Kalman Filter Implementation
for t = 1:num_days
    % Time update (Predict)
    x_pred = phi * x_est;
    P_pred = phi^2 * P + Q;

    % Measurement update (Correct) every 3rd interval
    if mod(t, 3) == 0
        y = rain_data((t/3));
        K = P_pred / (P_pred + R);
        x_est = x_pred + K * (y - x_pred * 3); % x_pred * 3 for accumulated sum
        P = (1 - K) * P_pred;
    else
        x_est = x_pred;
        P = P_pred;
    end

    x_est_store(t) = x_est;
end

% Reshape the estimated values for each ten-day interval
estimated_rain = reshape(x_est_store, [3, num_months])';

% Display the first few estimated values
disp(estimated_rain(1:5, :));


