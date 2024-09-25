clear variables;
close all;

%%
path2results = '../result/FEM/mu=0.001000/';
path2h5 = '../result/FEM/mu=0.001000/snapshots/';
dirh5 = dir([path2h5,'*.h5']);

drag = importdata([path2results, 'drag.txt']);
lift = importdata([path2results, 'lift.txt']);

for i = 1:length(dirh5)
    velocity(:,i) = h5read([path2h5, 'snapshot_', num2str(i-1,'%06.f'), '.h5'],'/velocity');
    pressure(:,i) = h5read([path2h5, 'snapshot_', num2str(i-1,'%06.f'), '.h5'],'/pressure');
    time(i) = h5read([path2h5, 'snapshot_', num2str(i-1,'%06.f'), '.h5'],'/time');
end

[U_v, S_v,V_v] = svds(velocity,inf);

energy_overall = sum(diag(S_v).^2);

for i = 1:size(S_v,1)
    energy(i) = sum(diag(S_v(1:i,1:i)).^2)/energy_overall;
    if energy(i) == 1
       disp(['100% at: ', num2str(i)]); 
       break;
    end
end

figure
semilogy(diag(S_v));
figure
semilogy(1-energy);
% figure
% plot(drag(:,1),drag(:,2));
% figure
% plot(lift(:,1),lift(:,2));
% figure
% plot(time)

% %%
% path2h5= '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/ROM/matrices/';
% A = h5read([path2h5, 'A', '.h5'],'/A');
% A_test = h5read([path2h5, 'A_test', '.h5'],'/A');
% 
% 
% %% 
% path2h5 = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/pod_vectors/';
% mean_vector = h5info([path2h5, 'mean_vector_standalone', '.h5']);
% % mean_vector = h5read([path2h5, 'mean_vector', '.h5'],'/cells');
% %% 
% path2h5 = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/pod_vectors/h5/';
% pod_vec = h5info([path2h5, 'vector-000000', '.h5']);
% % mean_vector = h5read([path2h5, 'mean_vector', '.h5'],'/cells');