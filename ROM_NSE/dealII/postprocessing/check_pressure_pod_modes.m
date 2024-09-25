clear all
close all

pwd2h5 = '/home/ifam/fischer/Code/result/POD/vp1/pod_vectors_press/';


for i =  1:5
    pwdread = [pwd2h5, 'pod_vectors_press' num2str(i-1,'%6.6i') ,'.h5'];
    A(:,i) = h5read(pwdread, '/mean_vector');
end

surf(A'*A)



%%
A = rand(100,10);

[U, ~,~] = svds(A,5);

norm(U'*U-eye(5))
norm(U*U'-eye(100))