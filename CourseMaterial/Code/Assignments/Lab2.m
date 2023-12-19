%% Clean up and add path
clear; clc;
close all;
addpath('../functions', '../data')
%% Generate data from a BJ model
rng(0)
n = 500; % Number of samples
A3 = [1 .5];
C3 = [1 -.3 .2];
w = sqrt(2) * randn(n + 100, 1) ;
x = filter(C3, A3, w) ; % Create the input
A1 = [1 -.65];
A2 = [1 .90 .78] ;
C = 1;
B = [0 0 0 0 .4] ;
e = sqrt(1.5 ) * randn(n + 100, 1) ;
y = filter(C, A1, e) + filter(B, A2, x) ; % Create the output
x = x(101 : end);  y = y(101 : end); % Omit initial samples
clear A1 A2 C B e w A3 C3

%% Modeling x
figure();
plot(x);
title('Input data');
plotACFnPACF(x, 50, 'Input Data' );

% Model as ARMA(1,2) component
data = iddata(x);
model_init = idpoly([1 1], [], [1 1 1]) ;
model_input = pem(data , model_init);

w_t = filter(model_input.A , model_input.C , x) ;
w_t = w_t(length(model_input.A ) : end ) ;

eps_t = filter(model_input.A , model_input.C , y) ;
eps_t = eps_t(length(model_input.A ) : end ) ;

present(model_input)
[acfEst, pacfEst] = plotACFnPACF(w_t, 50, 'AR(1, 2)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(w_t);

%%
M = 40;
[Cxy,lags] = xcorr(eps_t, w_t, M, 'coeff' ); %(d,r,s) = (4,2,0)
stem(lags, Cxy)
title ('Cross_correlation_function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off

%%
A2 = [1 1 1];
B = [0 0 0 0 1];
Mi = idpoly([1], [B], [], [], [A2]) ;
z = iddata(y, x) ;
Mba2 = pem(z, Mi ) ;present(Mba2)
etilde = resid(Mba2, z);
etilde(length(Mba2.B): end)

[Cxy,lags] = xcorr(etilde.y, x, M, 'coeff' );
stem(lags, Cxy)
title ('Cross_correlation_function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off

%% Model etilde
figure();
plot(etilde.y);
title('etilde data');
plotACFnPACF(etilde.y, M, 'Input Data' );

model_init = idpoly([1 1], [], []) ;
model_input = pem(iddata(etilde.y), model_init);

e_hat = filter(model_input.A , model_input.C , etilde.y) ;
e_hat = e_hat(length(model_input.A ) : end ) ;

present(model_input)
[acfEst, pacfEst] = plotACFnPACF(e_hat, M, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(w_t);

A1 = model_input.A;
C = model_input.C;
%% Full BJ Model
Mi = idpoly(1, B, C, A1, A2) ;
z = iddata(y, x);
MboxJ = pem (z, Mi);
present (MboxJ );

ehat = resid(MboxJ, z).y;
ehat = ehat(length(MboxJ): end);

[acfEst, pacfEst] = plotACFnPACF(ehat, M, 'Box-Jenkins');
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(ehat);

[Cxy,lags] = xcorr(ehat, x, M, 'coeff' );
stem(lags, Cxy)
title ('Cross_correlation_function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off

%% 2.2 Hairdryer
clear; clc;
close all;

load('tork.dat')
tork = tork - repmat(mean(tork), length(tork), 1);
y = tork(:, 1); x = tork(:, 2);
z = iddata(y , x);
plot(z(1:300))

%% Model input
M = 50;
plotACFnPACF(tork, M, 'tork inpu Data' );
input = tork(:, 2);
output = tork(:, 1);
n = length(input);

data = iddata(input);
model_init = idpoly([1 1], [], []) ;
model_input = pem(data, model_init);

w_t = filter(model_input.A , model_input.C , input) ;
w_t = w_t(length(model_input.A ) : end ) ;

eps_t = filter(model_input.A , model_input.C , output) ;
eps_t = eps_t(length(model_input.A ) : end ) ;

present(model_input)
[acfEst, pacfEst] = plotACFnPACF(w_t, M, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(w_t);
%% Check cross correlation to determine d,r,s
[Cxy,lags] = xcorr(eps_t, w_t, M, 'coeff' );
stem(lags, Cxy)
title ('Cross_correlation_function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off

% d,r,s =? 3, 2, 2
%% Build initial model and check that etilde and input is uncorrelated
A2 = [1 0.3 0.3];
B = [0 0 0 1 1 1];
Mi = idpoly([1], [B], [], [], [A2]) ;
z = iddata(output, input) ;
Mba2 = pem(z, Mi ) ;present(Mba2)
etilde = resid(Mba2, z);
etilde(length(Mba2.B): end)
M = 50;
[Cxy,lags] = xcorr(etilde.y, input, M, 'coeff' );
stem(lags, Cxy)
title ('Cross_correlation_function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off

%% Model etilde 
figure();
plot(etilde.y);
title('etilde data');
plotACFnPACF(etilde.y, M, 'Input Data' );

model_init = idpoly([1 1], [], []) ;
model_input = pem(iddata(etilde.y), model_init);

e_hat = filter(model_input.A , model_input.C , etilde.y) ;
e_hat = e_hat(length(model_input.A ) : end ) ;

present(model_input)
[acfEst, pacfEst] = plotACFnPACF(e_hat, M, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(w_t);

A1 = model_input.A;
C = model_input.C;

%% Final BJ Model
Mi = idpoly(1, B, C, A1, A2) ;
z = iddata(output, input);
MboxJ = pem (z, Mi);
present (MboxJ );

ehat = resid(MboxJ, z).y;
ehat = ehat(length(MboxJ): end);

[acfEst, pacfEst] = plotACFnPACF(ehat, M, 'Box-Jenkins');
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(ehat);

[Cxy,lags] = xcorr(ehat, input, M, 'coeff' );
stem(lags, Cxy)
title ('Cross_correlation_function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off

%% 2.3
load svedala
A = [1 -1.79 0.84];
C = [1 -0.18 -0.11];
y = svedala;
ks = [1 3 26];

for i=1:3
    
    k = ks(i);
    [Fk, Gk] = polydiv(C, A, k) ;
    yhatk = filter(Gk, C, y) ;
    ehat = y-yhatk;
    ehat = ehat(k+20:end);  % Remove the corrupted samples. You might need to add a bit of a margin.
    if(i>1)
        figure()
        subplot(211)
        plot(y)
        hold on
        plot(yhatk)
        title('Svedala')
        legend('True data', append(int2str(k), '-step prediction'))
    
        % Form the prediction error and examine the ACF. Note that the prediction
        % residual should only be white if k=1. 
        subplot(212)
        plot(ehat)
        legend('ehat = y-yhatk')

        mean_ehat = mean(ehat)
        var_theoretical = (norm(Fk).^2) .*var_noise
        var_est = var(ehat)
        conf = mean_ehat + [-1 1].*norminv(0.975).*sqrt(var_theoretical)
        precentage_outside = (sum(ehat > conf(2)) + sum(ehat < conf(1)))./length(ehat)
        plotACFnPACF(ehat, 40, append(int2str(k), '-step prediction'));
    else
        var_noise = var(ehat);
    end
  
end
%% 2.4 ARMAX
load svedala
xt = svedala;
k = 3;
A = [1 -1.79 0.84];
C = [1 -0.18 -0.11];
[Fk, Gk] = polydiv(C, A, k);

%xk_hat = xk_hat(k:end);

load sturup
y = sturup;
A = [1 -1.49 0.57];
B = [0 0 0 0.28 -0.26]; % delay = 3
C = [1];

[Fk_hat, Gk_hat] = polydiv(conv(B,Fk), C, k ) ;
xk_hat = filter(Gk, C, xt);
yhatk1 = filter(Fk_hat, 1, xk_hat); yhatk2 = filter(Gk_hat, C, xt); yhatk3= filter(Gk, C, y);

throw = max([length(Gk_hat(1:find(Gk_hat, 1, 'last')))-1, ...
            length(Gk(1:find(Gk, 1, 'last')))-1, ...
            length(Fk_hat(1:find(Fk_hat, 1, 'last')))-1]); % Should not be here?? Throw C?


yhatk1 = yhatk1(throw:end); yhatk2= yhatk2(throw:end); yhatk3=yhatk3(throw:end);
yhatk = yhatk1 + yhatk2 + yhatk3;

figure();
plot(y(throw:end))
hold on 
plot(yhatk)
legend('True', 'Predicted')

%%
load svedala.mat
figure() 
plot(svedala)
plotACFnPACF(svedala, 40, 'data');
svedala_trend = svedala;
A24 = [1 zeros(1, 23) -1];
svedala_s = filter(A24, 1, svedala_trend);
svedala_s = svedala_s(length(A24) : end );

plotACFnPACF(svedala_s, 40, 'Removed season of 24' );

data = iddata(svedala_s);
% Model 1 
model_init = idpoly([1 1 1], [], []) ;
model_armax_1 = pem(data , model_init);

e_hat = filter(model_armax_1.A , model_armax_1.C , svedala_s) ;
e_hat = e_hat(length(model_armax_1.A ) : end ) ;

present(model_armax_1)
plotACFnPACF(e_hat, 40, 'AR(2)' );
checkIfWhite(e_hat);


% Model 2
model_init = idpoly([1 1 1], [], [1 zeros(1,24)]) ;
model_init.Structure.c.Free = [zeros(1, 24) 1] ;

model_armax_2 = pem(data , model_init);

e_hat = filter(model_armax_2.A , model_armax_2.C , svedala_s) ;
e_hat = e_hat(length(model_armax_2.A ) : end ) ;

present(model_armax_2)
[acfEst, pacfEst] = plotACFnPACF(e_hat, 40, 'ARMA(2, 24)' );
checkIfNormal( acfEst(2:end), 'ACF' );
checkIfNormal( pacfEst(2:end), 'PACF' );

checkIfWhite(e_hat);
A = model_armax_2.A;
C = model_armax_2.C;
%% 2.5
ks = [3, 26];
for i=1:2
    k = ks(i);
    [Fk, Gk] = polydiv(C, conv(A, A24), k) ;
    yhatk = filter(Gk, C, svedala);
    
    yhatk = yhatk(length(Gk):end);
    figure()
    subplot(211)
    plot(svedala(length(Gk):end));
    hold on;
    plot(yhatk)
    legend('True', 'Prediction')
    title(append(int2str(k), '-step prediction'))

    
    subplot(212)
    ehat = svedala(length(Gk):end)-yhatk;
    plot(ehat)
    legend('Residuals')
    var(ehat)
end
