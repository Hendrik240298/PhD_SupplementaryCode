clear all
close all
%% 
% path2h5 = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/';
% vec = h5info([path2h5, 'W', '.h5']);
% vec = h5read([path2h5, 'W', '.h5'],'/mean_vector');
% vec = importdata([path2h5, 'sing_red_all.txt']);
% vec2 = importdata([path2h5, 'sing_all.txt']);
% vec3 = importdata([path2h5, 'sing_red_mass.txt']);
% %%
% figure
% plot(1:length(vec),vec)
% hold on; 
% plot(1:length(vec2),vec2)
% figure
% plot(1:length(vec(1:100)),abs(vec(1:100)-vec2(1:100)))
% % figure
% % plot(1:length(vec3),vec3)
% %%
close all
clear all
n=25;
path2h5 = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/mu=0.001000/';
for i = 1:n
    pathfile = ['pod_vectors/pod_vectors',  num2str(i-1,'%6.6i') ,'.h5'];
    path = [path2h5, pathfile];
%     h5info(path)
    pod_SVD(:,i) = h5read(path, '/mean_vector');
end
eig_SVD =  h5read([path2h5, 'eigenvalues.h5'], '/mean_vector');
weights = h5read([path2h5 ,'space_weights.h5'], '/mean_vector');


path2h5 = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/mu=0.001000_eig/';
for i = 1:n
    pathfile = ['pod_vectors/pod_vectors',  num2str(i-1,'%6.6i') ,'.h5'];
    path = [path2h5, pathfile];
    pod_eig(:,i) = h5read(path, '/mean_vector');
end
eig_eig =  h5read([path2h5, 'eigenvalues.h5'], '/mean_vector');

figure
subplot(1,2,1)
surf(pod_SVD'*diag(weights)*pod_SVD-eye(n))
subplot(1,2,2)
surf(pod_eig'*diag(weights)*pod_eig-eye(n))

norm(pod_SVD'*diag(weights)*pod_SVD-eye(n))
norm(pod_eig'*diag(weights)*pod_eig-eye(n))

figure
semilogy(eig_SVD);
hold on;
semilogy(eig_eig);
legend('SVD','eig');
grid on

figure
for i = 1:n
   scatter(i,norm(pod_SVD(:,i)-pod_eig(:,i))/norm(pod_eig(:,i)));
   hold on;
end
%%
% 
close all
clear all
path2h5{1} = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/mu=0.002000/';
path2h5{2} = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/mu=0.001667/';
path2h5{3} = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/mu=0.001429/';
path2h5{4} = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/mu=0.001250/';
path2h5{5} = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/mu=0.001111/';
path2h5{6} = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/mu=0.001000/';
path2h5{7} = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/mu=0.000909/';
path2h5{8} = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/mu=0.000833/';
path2h5{9} = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/mu=0.000769/';

% t= h5info(path2h5);
for i = 1:9
    data{i} = h5read([path2h5{i}, 'eigenvalues.h5'], '/mean_vector');
end

% A = importdata([path2h5, 'eigenvalues.txt']);
% % A_eig = importdata([path2h5, 'eigenvalues_old.txt']);
% A = A.data;
% A_eig = A_eig.data;
figure
semilogy(data{1}/data{1}(1));
hold on
for i = 2:9
    semilogy(data{i}/data{i}(1));
end
% semilogy(A_eig(:,1),A_eig(:,2));
legend('1', '2', '3', '4', '5', '6', '7', '8', '9')


%%
close all
clear all
path2h5 = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/POD_greedy/';
data = h5read([path2h5, 'eigenvalues.h5'], '/mean_vector');
figure
plot(data);
figure(2)
plot(data(1:end-1)-data(2:end));