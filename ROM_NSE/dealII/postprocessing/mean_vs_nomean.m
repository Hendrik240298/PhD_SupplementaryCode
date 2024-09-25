close all
clear all

path2fem = '/media/hendrik/hard_disk/Nextcloud/Code/result/FEM/mu=0.000833/snapshots/';

n=200;

for i = 1:n
    pathfile = ['snapshot_',  num2str(i-1,'%6.6i') ,'.h5'];
    path = [path2fem, pathfile];
    A(:,i) = h5read(path, '/velocity');
end

%%
A_mean = A - mean(A')';
nber = 50;
[U,S,V] = svds(A,nber);
[U_mean,S_mean,V_mean] = svds(A_mean,nber);

S = diag(S);
S_mean = diag(S_mean);
%%
figure
semilogy(1:nber,S)
hold on
semilogy(2:nber+1,S_mean(1:end))

%%
norm(A-U*diag(S)*V')
norm(A-(mean(A')'+U_mean*diag(S_mean)*V_mean'))