close all;
clear variables;
%%
path2fom = '/media/hendrik/hard_disk/Nextcloud/Code/nn_rom/';
path2rbf = '/media/hendrik/hard_disk/Nextcloud/Code/nn_rom/';
path2save = '/media/hendrik/hard_disk/Nextcloud/Code/nn_rom/';
L = 10; %take 200 as maximum
N = 1380;
N_h = 21024;
S = zeros(N_h,N);
V = zeros(N_h,L);
time = zeros(N,1);

% read fom snapshots
for i = 1:N
    path = [path2fom, 'snapshots/snapshot_'  num2str(i-1,'%6.6i') ,'.h5'];
    S(:,i) = h5read(path, '/velocity');
    time(i) = h5read(path, '/time');
end

% divide mean from snapshots
S_light = S - mean(S')';

% save modified snapshots
for i = 1:N
    path = [path2save, 'modified_snapshots/snapshot_'  num2str(i-1,'%6.6i') ,'.h5'];
    if ~isfile(path)
        h5create(path, '/velocity',[N_h,1]);
        h5create(path, '/time',[1,1]);
    end
    h5write(path, '/velocity',S_light(:,i));
    h5write(path, '/time',time(i));
end

% build V matrix out of reduced basis vectors
for i = 1:L
    path = [path2rbf, 'pod_vectors/pod_vectors'  num2str(i-1,'%6.6i') ,'.h5'];
    V(:,i) = h5read(path, '/mean_vector');
end
%% 
RHS = zeros(N,L);
for i=1:N
    RHS(i,:) = V'*S_light(:,i);
    T(i) = time(i);
end

% trainX = T(1:2:end);
% trainY = RHS(1:2:end,:);
% testX = T(2:2:end);
% testY = RHS(2:2:end,:);


trainX = T(1:1000);
trainY = RHS(1:1000,:);
testX = T(1001:end);
testY = RHS(1001:end,:);
%%
csvwrite([path2save, 'trainX.csv'],trainX)
csvwrite([path2save, 'trainY.csv'],trainY)
csvwrite([path2save, 'testX.csv'],testX)
csvwrite([path2save, 'testY.csv'],testY)