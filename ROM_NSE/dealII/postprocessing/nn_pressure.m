close all;
clear variables;
%%
path2fom = '/media/hendrik/hard_disk/Nextcloud/Code/result/FEM/mu=0.001000/';
path2rbf = '/media/hendrik/hard_disk/Nextcloud/Code/nn_rom/';
path2save = '/media/hendrik/hard_disk/Nextcloud/Code/nn_rom/';
L = 10; %take 200 as maximum
N = 200;
N_h = 2696;%21024;
S = zeros(N_h,N);
V = zeros(N_h,L);
time = zeros(N,1);

% read fom snapshots
for i = 1:N
    path = [path2fom, 'snapshots/snapshot_'  num2str(i-1,'%6.6i') ,'.h5'];
    S(:,i) = h5read(path, '/pressure');
    time(i) = h5read(path, '/time');
end

[U,Sig,~] = svds(S,L);
T = (time-time(1))/(time(end)-time(1));
figure
semilogy(diag(Sig))

%% 
RHS = zeros(N,L);
for i=1:N
    RHS(i,:) = U'*S(:,i);
end

chaos = randperm(N);

N_train = 150;
 
trainX = T(chaos(1:N_train));
trainY = RHS(chaos(1:N_train),:);
testX = T(chaos(N_train+1:end):end);
testY = RHS(chaos(N_train+1:end):end,:);
%%
% csvwrite([path2save, 'trainX.csv'],trainX)
% csvwrite([path2save, 'trainY.csv'],trainY)
% csvwrite([path2save, 'testX.csv'],testX)
% csvwrite([path2save, 'testY.csv'],testY)