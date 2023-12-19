%% Generate data from a BJ model
rng(0)
n = 500; % Number of samples
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
%clear A1 A2 C B e w A3 C3
%%
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
%%
figure();

plot(x);
title('Input data');
plotACFnPACF(x, 50, 'Input Data' );

% Model as ARMA(1,2) component
data = iddata(x);
Am = conv([1 1 1], [1 zeros(1, 35) 1]);%[1 1 1 zeros(1, 33) 1];
%Cm = [1 zeros(1, 35) 1];
model_init = idpoly(Am, [], []) ;
model_init.Structure.a.Free = Am;
%model_init.Structure.c.Free = [1 1];

model_input = pem(data , model_init);

w_t = filter(model_input.A , model_input.C , x) ;
w_t = w_t(length(model_input.A ) : end ) ;

eps_t = filter(model_input.A , model_input.C , y) ;
eps_t = eps_t(length(model_input.A ) : end ) ;

present(model_input)
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
Mi.Structure.B.Free = B;
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
%%
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
x = rain_reconstructed(find(tt==nvdi_t(1)): find(tt==nvdi_t(1)) + length(model_nvdi) - 1);
x= log(x+1);
figure();

plot(x);
title('Input data');
plotACFnPACF(x, 50, 'Input Data' );

% Model as ARMA(1,2) component
data = iddata(x);
Am = conv([1 1 1 1], [1 zeros(1, 35) 1]);%[1 1 1 zeros(1, 33) 1];
Cm = [1 zeros(1, 35) 1];
model_init = idpoly(Am, [], Cm) ;
model_init.Structure.a.Free = Am;
model_init.Structure.c.Free = Cm;

inputModel = pem(data , model_init);

w_t = filter(inputModel.A , inputModel.C , x) ;
w_t = w_t(length(inputModel.A )+20 : end ) ;
var_ex = var(w_t);

eps_t = filter(inputModel.A , inputModel.C , y) ;
eps_t = eps_t(length(inputModel.A ) +20: end ) ;

present(inputModel)
[acfEst, pacfEst] = plotACFnPACF(w_t, 50, 'AR(1)' );
checkIfNormal(acfEst(2:end), 'ACF' );
checkIfNormal(pacfEst(2:end), 'PACF' );
checkIfWhite(w_t);

A = inputModel.A;
C = inputModel.C;

k = 1;
[Fx, Gx] = polydiv( inputModel.C, inputModel.A, k );
xhatk = filter(Gx, inputModel.C, x);
modelLim = 400;
figure
plot([x xhatk] )
line( [modelLim modelLim], [-1e6 1e6 ], 'Color','red','LineStyle',':' )
legend('Input signal', 'Predicted input', 'Prediction starts')
title( sprintf('Predicted input signal, x_{t+%i|t}', k) )
axis([1 N min(x)*1.5 max(x)*1.5])

std_xk = sqrt( sum( Fx.^2 )*var_ex );
fprintf( 'The theoretical std of the %i-step prediction error is %4.2f.\n', k, std_xk)

