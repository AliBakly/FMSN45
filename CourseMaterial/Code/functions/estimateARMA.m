% FUNCTION [ foundModel, ey, acfEst, pacfEst ] = estimateARMA( data, Am, Cm, titleStr, noLags, [beSilent] )
%
% The function estimates the specified ARMA model (Am, Cm) for data and
% returns the found model and the residual. If beSilent is set, no results
% are presented.
%
% Reference: 
%   "An Introduction to Time Series Modeling", 4th ed, by Andreas Jakobsson
%   Studentlitteratur, 2021
%
function [ foundModel, ey, acfEst, pacfEst ] = estimateARMA( data, Am, Cm, titleStr, noLags, beSilent )

% Unless specified, print results and show the ACF and PACF.
if nargin<6
    beSilent = 0;
end

% It is not uncommon that PEM suffers instabilities for complicated model
% structures, getting stuck in a local minima. An often useful trick is to
% initiate the polynomials with smaller coefficients. If you gets very odd
% estimates, try reducing the multiple a bit further - if you are lucky, it
% might help!
Am(2:end) = Am(2:end) * 0.3;
Cm(2:end) = Cm(2:end) * 0.3;

dataContainer = iddata( data );
polyContainer = idpoly( Am,[],Cm );
if length(Cm)>1
    polyContainer.Structure.c.Free = Cm;        % Only estimate the specified C(z) parameters.
end
if length(Am)>1
    polyContainer.Structure.a.Free = Am;        % Only estimate the specified A(z) parameters.
end
foundModel = pem( dataContainer, polyContainer );

% Compute the residual and its variance.
ey = filter( foundModel.A, foundModel.C, data ); ey = ey(length(foundModel.A):end );

% Present results.
if ~beSilent
    present( foundModel );

    % Plot the ACF and PACF of the residual.
    fprintf('\nThe variance of the residual is %5.3f.\n', var(ey) )
    [acfEst, pacfEst] = plotACFnPACF( ey, noLags, titleStr );
    
    % Is the residual white? Lets examine the Monti test.
    checkIfWhite( ey );
else
    acfEst  = acf( ey, noLags );
    pacfEst = pacf( ey, noLags );
end
