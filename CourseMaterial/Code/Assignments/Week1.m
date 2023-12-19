%% Clean up and add path
clear; clc;
close all;
addpath('../functions', '../data')              % Add this line to update the path
%% 1
[y, Fs] = audioread('fa.wav');
% sound(y) % listen to file
figure()
plot(y)

%% 2
y_vowel = y(7600: 7800); % one example
sound(y_vowel)

figure()
plot(y_vowel)
figure()
rho = acf(y_vowel, 100, 0.05, 1);
% Every 35th lag same val periodicity
freq1 =(1/35).*Fs; 

% Also seems to exist a periodicity every 15-20th lag
freq2 =(1/17.5).*Fs; 

%% 3
y_vowel = y(7600: 7800); % one example
Padd = 1024;
N = length(y_vowel);
Y = fftshift( abs(fft(y_vowel, Padd) ).^2 / N );
figure()
title('Frequency-domain')
ff = (0:Padd-1)'/Padd-0.5;
semilogy(ff.*Fs, Y)

% Strong peaks at 230 & 460 !! as predicted in freq1 and fre2
