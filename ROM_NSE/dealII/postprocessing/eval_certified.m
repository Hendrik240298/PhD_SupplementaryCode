% close all;
clear all;
path2rom = '../result/ROM/test2/';
res = importdata([path2rom, 'residual_test.txt']);

Re = [50:10:90, 110:10:159];

res = sum(res');
% res = A(:,end);
figure
plot(Re,res/max(res));
