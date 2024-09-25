close all
clear all

path0 = '/media/hendrik/hard_disk/Nextcloud/Code/result/POD/vp1/pod_vectors_press/';
path1 = '/media/hendrik/hard_disk/Nextcloud/Code/result/POD/vp1/pod_vectors/';

pod_p(:,1) =h5read([path0, 'pod_vectors_press000000.h5'], '/mean_vector');
pod_p(:,2) =h5read([path0, 'pod_vectors_press000001.h5'], '/mean_vector');

pod_v(:,1) =h5read([path1, 'pod_vectors000000.h5'], '/mean_vector');
pod_v(:,2) =h5read([path1, 'pod_vectors000001.h5'], '/mean_vector');

A = importdata('/home/hendrik/Code/rom-nse/pressure.txt');
A2= importdata('/home/hendrik/Code/rom-nse/pressure_system.txt');
P = sparse(max(A(2,:))+1,max(A(3,:))+1);
P2 = sparse(max(A2(2,:))+1,max(A2(3,:))+1);
for i = 1:length(A)
    P(A(2,i)+1,A(3,i)+1) = A(1,i);
end
for i = 1:length(A)
    P2(A2(2,i)+1,A2(3,i)+1) = A2(1,i);
end
%%
% pod_v'*(P*pod_p)
% 
% (pod_v')*(P*(1e10*pod_p))
% % pod_v'*(P2*pod_p)
% 
% pod_v'*(P2*pod_p)

disp('norm')
sum(sum(abs(P-P2)))

figure
spy(P)
figure
spy(P2)


% pod_v(:,1)'*pod_v(:,2)
% pod_p(:,1)'*pod_p(:,2)

norm(pod_p(:,1))
norm(pod_v(:,1))

max(max(P))

% figure
% plot(A2)
% hold on
% plot(pod_p(:,1))
% 
% 
% sum(A2'-pod_p(:,1))
%%
% figure
% plot(pod0)
% hold on
% plot(pod1)