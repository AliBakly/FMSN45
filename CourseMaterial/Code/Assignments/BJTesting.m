%% Examine the data.


x = rain_reconstructed(find(tt==nvdi_t(1)): find(tt==nvdi_t(1)) + length(model_nvdi) - 1);
%x = x./max(x); % RESCALING
y = model_nvdi;
A36 = [1 zeros(1, 35) -1];

x = filter(A36, 1, x );
x = x(length(A36 ) : end );
%A36 = [1 zeros(1, 35) -1];
%y = filter(A36, 1, y );
y = y(length(A36 ) : end );

figure; 
subplot(211); 
plot(x); % manual inspection
ylabel('Input signal')
title('Measured signals')
subplot(212); 
plot( y ); 
ylabel('Output signal')
xlabel('Time')
%x = log(x+1);
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
%% Modeling x_diff
plot(x);
title('Input data');
plotACFnPACF(x, 50, 'Input Data' );

% Model as ARMA(1,2) component
data = iddata(x);
Am = [1 1 1 zeros(1, 33) 1];
%Cm = [1 1 1];
model_init = idpoly(Am, [], []) ;
model_init.Structure.a.Free = Am;
%model_init.Structure.c.Free = Cm;

model_input = pem(data , model_init);
present(model_input)
model_input.A = conv(A36, model_input.A);

w_t = filter(model_input.A , model_input.C , x) ;
w_t = w_t(length(model_input.A )+30 : end ) ;

eps_t = filter(model_input.A , model_input.C , y) ;
eps_t = eps_t(length(model_input.A )+30 : end ) ;

[acfEst, pacfEst] = plotACFnPACF(w_t, 50, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(w_t);

A = model_input.A;
C = model_input.C;
%%
n = length(x);
M = 70;
figure
[Cxy,lags] = xcorr(eps_t, w_t, M, 'coeff' ); %(d,r,s) = (4,2,0)
stem(lags, Cxy)
title ('Cross_correlation_function'), xlabel('Lag')
hold on
plot(-M:M, 2/ sqrt(n) * ones(1, 2*M+1), '--')
plot(-M:M, -2/sqrt(n) * ones(1, 2*M+1) , '--' )
hold off
d=3; r=0; s=3; 
%%
A2 = ones(1, r+1);
B = [zeros(1, d) ones(1, s +1) ];
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

%A2 = [1 0 -1];
%Am = conv([1 1], A2);
%model_init = idpoly(Am, [], [1 zeros(1,35) 1]) ;
%model_init.Structure.a.Free = Am;
%model_init.Structure.c.Free = [1 zeros(1,35) 1];
model_init = idpoly([1 1], [], [1 zeros(1,35) 1]) ;
model_init.Structure.c.Free = [1 zeros(1,35) 1];
model_input = pem(iddata(etilde.y), model_init);

e_hat = filter(model_input.A , model_input.C , etilde.y) ;
e_hat = e_hat(length(model_input.A ) : end ) ;

present(model_input)
[acfEst, pacfEst] = plotACFnPACF(e_hat, M, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(e_hat);

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
%%

k = 1;
[Fx, Gx] = polydiv( inputModel.C, inputModel.A, k );
xhatk = filter(Gx, inputModel.C, x);
 
figure
plot([x xhatk] )
line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' )
legend('Input signal', 'Predicted input', 'Prediction starts')
title( sprintf('Predicted input signal, x_{t+%i|t}', k) )
axis([1 N min(x)*1.5 max(x)*1.5])

std_xk = sqrt( sum( Fx.^2 )*var_ex );
fprintf( 'The theoretical std of the %i-step prediction error is %4.2f.\n', k, std_xk)

