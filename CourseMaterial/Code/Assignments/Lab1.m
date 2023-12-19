%% Clean up and add path
clear; clc;
close all;
addpath('../functions', '../data')
%% Task 1
A1 = [ 1 -1.79 0.84 ];
C1 = [ 1 -0.18 0.11 ];

A2 = [ 1 -1.79 ] ;
C2 = [ 1 -0.18 -0.11 ] ;

ARMA_poly1 = idpoly ( A1, [ ] , C1 ) ;
figure()
pzmap( ARMA_poly1 )

ARMA_poly2 = idpoly ( A2, [ ] , C2 ) ;
figure()
pzmap( ARMA_poly2 )

rng(0);
N = 300;
sigma2 = 1.5;
e = sqrt ( sigma2 ) .* randn(N, 1);
y1 = filter ( ARMA_poly1.c , ARMA_poly1.a , e ) ;
y1 = y1(101: end);

y2 = filter ( ARMA_poly2.c , ARMA_poly2.a , e ) ;
y2 = y2(101: end);

figure()
subplot(2, 1, 1)
plot ( y1 )
subplot(2, 1, 2)
plot ( y2 )

%%
m = 30;
r_theo = kovarians(ARMA_poly1.c , ARMA_poly1.a, m);
figure()
stem(0 :m, r_theo* sigma2 )
hold on
r_est = covf(y1, m+1 );
stem(0 :m, r_est , 'r')
legend('Theoretical', 'Estimated')
%% Model 1
data = iddata(y1);
plotACFnPACF(y1,m, 'y1');
ar_model_1 = arx (y1, [1] ); %Ar 1, Why not PEM??
present(ar_model_1);

e_hat = filter(ar_model_1.A , ar_model_1.C , y1 ) ;
e_hat = e_hat(length(ar_model_1.A ) : end ) ;
plotACFnPACF( e_hat, m, 'Residual, model 1' );
checkIfWhite(e_hat);

%% Model 2, I WOULD STOP HERE
ar_model_2 = arx (y1, [2] ); %Ar 2
present(ar_model_2);

e_hat = filter(ar_model_2.A , ar_model_2.C , y1 ) ;
e_hat = e_hat(length(ar_model_2.A ) : end ) ;
plotACFnPACF( e_hat, m, 'Residual, model 2' );
checkIfWhite(e_hat);
%% Model 3 Gives better FPE and MSE
ar_model_3 = armax (y1, [2,1] ); %Ar 
present(ar_model_3);

e_hat = filter(ar_model_3.A , ar_model_3.C , y1 ) ;
e_hat = e_hat(length(ar_model_3.A ) : end ) ;
plotACFnPACF( e_hat, m, 'Residual, model 3' );
checkIfWhite(e_hat);

%% 2.2 AR
load data.dat
load noise.dat
plotACFnPACF(data, 30, 'data');
data=iddata(data);
for p=1:5
    ar_model = arx(data, [p]);
    present(ar_model)
    rar = resid(ar_model, data );
    rar = rar(length(ar_model.A ) : end);
    var(rar.y)
    figure()
    subplot(1,2,1)
    plot(rar.y)
    title(append('Residuals of model AR(' , int2str(p), ')'));
    subplot(1,2,2)
    plot(noise)
    title('Actual noise')

    plotACFnPACF( rar.y, m, append('Residual, model  AR(', int2str(p), ')'));
    checkIfWhite(rar.y);
end

%% 2.2 ARMA
for p = 1:2
    for q =1:2
        arma_model = armax(data, [p, q]);
        present(arma_model)
        rar = resid(arma_model, data );
        rar = rar(length(arma_model.A ) : end);
    
        figure()
        subplot(1,2,1)
        plot(rar.y)
        title(append('Residuals of model ARMA(' , int2str(p), ', ', int2str(q), ')'));
        subplot(1,2,2)
        plot(noise)
        title('Actual noise')
    
        plotACFnPACF( rar.y, m, append('Residuals of model ARMA(' , int2str(p), ', ', int2str(q), ')'));
        checkIfWhite(rar.y);
    end
end

%% 2.3 
rng(0)
A = [1 -1.5 0.7 ] ;
C = [1 zeros(1, 11) -0.5];
A12 = [1 zeros(1, 11) -1];
A_star = conv(A, A12) ;
e = randn(600, 1) ;
y = filter(C, A_star, e ) ;
y = y(101: end) ;
figure()
plot(y)
plotACFnPACF(y,30, 'y');
figure; 
lambda_max = bcNormPlot(y,1);
%% Filter out season
y_s = filter(A12, 1, y );
y_s = y_s(length(A12 ) : end );
data = iddata(y_s);
plotACFnPACF( y_s, 30, 'Season filtered out' );

model_init = idpoly([1 0 0], [], []) ;
model_armax_1 = pem(data , model_init);

e_hat = filter(model_armax_1.A , model_armax_1.C , y_s) ;
e_hat = e_hat(length(model_armax_1.A ) : end ) ;

present(model_armax_1)
plotACFnPACF(e_hat, 30, 'AR(1)' );
checkIfWhite(e_hat);
figure()
normplot(e_hat)

% Second model
model_init = idpoly([1 0 0], [], [1 zeros(1,12)]) ;
model_init.Structure.c.Free = [zeros(1, 12) 1] ;

model_armax_2 = pem(data , model_init);

e_hat = filter(model_armax_2.A , model_armax_2.C , y_s) ;
e_hat = e_hat(length(model_armax_2.A ) : end ) ;

present(model_armax_2)
plotACFnPACF(e_hat, 30, 'ARMA(2, 12)' );
checkIfWhite(e_hat);
figure()
normplot(e_hat)

% Do same but with N= 10 000

%% 2.4
load svedala.mat
figure() 
plot(svedala)
plotACFnPACF(svedala, 40, 'data');
figure; 
lambda_max = bcNormPlot(svedala,1);

% Damped sin --> Ar(2)
% Variance seems stable
% Maybe small trend?? Skip for now
% A1 = [1 -1];
% svedala_trend = filter(A1, 1, svedala);
% svedala_trend = svedala_trend(length(A1 ) : end );
% figure();
% plot(svedala_trend)
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


%% 2.4
load svedala.mat
figure() 
plot(svedala)
plotACFnPACF(svedala, 40, 'data');
figure; 
lambda_max = bcNormPlot(svedala,1);
% mdl = fitlm([1:length(svedala)]', svedala);
% m= mdl.Coefficients(1,1).Estimate;
% k= mdl.Coefficients(2,1).Estimate;

% Damped sin --> Ar(2)
% Variance seems stable
% Maybe small trend?? Skip for now
% A1 = [1 -1];
% svedala_trend = filter(A1, 1, svedala);
% svedala_trend = svedala_trend(length(A1 ) : end );
% figure();
% plot(svedala_trend)
%svedala_trend = svedala- [1:length(svedala)]'.*k +m;
% figure()
% plot(svedala_trend)


svedala_trend = svedala;
A24 = [1 zeros(1, 23) -1];

data = iddata(svedala_trend);
% Model 1 
Am = conv(A24, [1 1 1]);
model_init = idpoly(Am, [], [1 zeros(1,24)]) ;
model_init.Structure.a.Free = Am;
model_init.Structure.c.Free = [zeros(1, 24) 1] ;


model_armax_1 = pem(data , model_init);

e_hat = filter(model_armax_1.A , model_armax_1.C , svedala_trend) ;
e_hat = e_hat(length(model_armax_1.A ) : end ) ;

present(model_armax_1)
plotACFnPACF(e_hat, 40, 'ARMA(26,24)' );
checkIfWhite(e_hat);


