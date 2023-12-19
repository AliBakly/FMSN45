% FUNCTION plotWithConf( time, data, xStd, [trueParams] )
%
% The function plots the time evolution of data with the corresponding
% confidence interval confInt. If the optional parameter trueParams is
% given, a dashed line indicating these values are plotted as well. 
%

% Reference: 
%   "An Introduction to Time Series Modeling", 4th ed, by Andreas Jakobsson
%   Studentlitteratur, 2021
%
function plotWithConf( time, data, xStd, trueParams )

time = time(:);
plot( time, data, 'b' )
hold on 
fill([time ; flipud(time)] , [data-xStd; flipud(data+xStd )], 'g', 'FaceAlpha', 0.1, 'LineStyle',':' )
if nargin>3
    for k0=1:length(trueParams)
        line([time(1) time(end)], [trueParams(k0) trueParams(k0)], 'Color','red','LineStyle','--')
    end
end
hold off
