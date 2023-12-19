%
% Time series analysis
% Lund University
%
% Example code 12: Predicting a signal. This is an IMPORTANT example!
%
% Reference: 
%   "An Introduction to Time Series Modeling", 4th ed, by Andreas Jakobsson
%   Studentlitteratur, 2021
%
clear; 
close all;
addpath('../functions', '../data')              % Add this line to update the path


% Select parameters.
k = 5;                                          % Set prediction horizon;try some different ones.
N = 10000;                                      % Begin with lots of samples - then try fewer.

% Some examples of polynomials to test. Try mixing the A and Cs. Note that
% both polynomials have to have their roots inside the unit circle. Why is
% it that this is also required for the C(z) polynomial?
C = [ 1 ];
A = [ 1 -1.96 .97 ];                            % Strong periodicity; this is easy to predict.

%C = [ 1 ];
%A = [ 1 -.8 .6 ];                              % Weak periodicity; this is difficult to predict.

%C = [ 1 -2 ];                                  % Why is the prediction instable with this C polynomial?
%A = [ 1 -1.96 .97 ];                           % Strong periodicity.

%C = poly( [0.2+0.8i 0.2-0.8i] );               % You can also place the roots.
%A = poly( [0.7+0.6i 0.7-0.6i] );               

% Note: try the trends a few times each - due to the roots on the unit
% circle, some realizations will yield "odd" results. Examine these a bit
% closer.
%C = [ 1 .2 ];
%A = conv([1 -1],[ 1 -0.2 ]);                   % Signal with a slow trend.

%C = [ 1 -.2 ];
%A = conv([1 zeros(1,12) -1],[ 1 -0.7 ]);       % Signal with a season.



% Simulate some data and examine the location of the roots. 
noLags = 50;
extraN = 100;
e = randn( N+extraN, 1 );
y = filter( C, A, e );     y = y(extraN+1:end);

figure
subplot(211)
zplane(A)
title('Roots of the A(z) polynomial')
subplot(212)
zplane(C)
title('Roots of the C(z) polynomial')



%% Form the prediction polynomials.
% First, use the true parameters to test the theory.
[F, G] = polydiv( C, A, k );

% Then, try estimating the parameters and use these instead. Also try
% reducing N to see how this affects the parameter estimates and the
% predictions. You can use this approach to examine the quality you can
% expect of your predictions using your polynomials and data lengths.
%
foundModel = estimateARMA( y, A, C, 'Estimating the model parameters using known model orders', noLags, 1 );
[F, G] = polydiv( foundModel.C, foundModel.A, k ); 

% Form the predicted data.
yhatk  = filter( G, C, y );

% Plot the predictions.
figure
subplot(311)
maxInd = 40;
m1 = min([yhatk(1:maxInd)' y(1:maxInd)']);
m2 = max([yhatk(1:maxInd)' y(1:maxInd)']);
plot([y yhatk] )
hold on
line( [k k], [m1 m2], 'Color','red','LineStyle','--' )
hold off
axis([1 50 m1 m2] )
legend('y','yhat','Lost samples')
title(sprintf('Predicted signal. Note that the initial %i samples are lost', k))

subplot(312)
plot([y yhatk] )
xlim([200 700])
legend('y','yhat')

% Form the prediction error and examine the ACF. Note that the prediction
% residual should only be white if k=1. 
ehat = y-yhatk;
ehat = ehat(k+20:end);                          % Remove the corrupted samples. You might need to add a bit of a margin.
subplot(313)
plot(ehat)
legend('ehat = y-yhatk')
try                                             % Catch the error in case of instable polynomials.
    checkIfWhite( ehat );
    figure
    acfEst = acf( ehat, noLags, 0.05, 1 );
    hold on
    line( [k-1 k-1], [-2 2 ], 'Color','r','LineStyle','--' )
    hold off
    axis([0 noLags -1 1 ])
    legend('ACF', '95 %% confidence interval', sprintf('Is a MA(%i) model reasonable?', k-1), 'Location','NE')
    title( sprintf('Prediction residual, %i-step prediction', k) )
    fprintf('This is a %i-step prediction. Ideally, the residual should be an MA(%i) process.\n', k, k-1)
end

% Examine the variance of the prediction residual in comparison to the
% variance of the data.
fprintf('Prediction the signal %i-steps ahead.\n', k)
fprintf('  The variance of original signal is         %5.2f.\n', var(y)')
fprintf('  The variance of the prediction residual is %5.2f.\n', var(ehat)')
if var(ehat)<var(y)
    fprintf('  Amount of signal that was predicted is     %5.2f %%.\n', (1-var(ehat)/var(y))*100)
else
    fprintf('  **** BEWARE: the prediction is not accurate!!! ****\n')
end
