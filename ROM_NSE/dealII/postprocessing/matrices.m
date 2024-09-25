clear all
close all

pwd2h5 = '/home/ifam/fischer/Code/result/ROM/vp1/matrices/';


L = h5read([pwd2h5, 'A.h5'], '/A');
B = h5read([pwd2h5, 'B.h5'], '/A')';
BT = h5read([pwd2h5, 'BT.h5'], '/A')';
D = h5read([pwd2h5, 'D.h5'], '/A');
C = h5read([pwd2h5, 'C.h5'], '/A');
C1 = h5read([pwd2h5, 'C1.h5'], '/A');
rhs = h5read([pwd2h5, 'lifting_rhs_vector_mass.h5'], '/mean_vector');
rhs(21:40) = 0;
A = (L+D+B+BT+C);;
u = A\rhs;


A_ONE = A;
A_ONE(41:60,21:40) = eye(20);
A_ONE(1:40,21:40) = 0;
u_ONE = A_ONE\rhs;


tol = 1e-14;
row = find(rhs>=tol);
test_rhs = 0*rhs;
test_rhs(row) = 1;
[row,col] = find(A_ONE>=tol);
test = 0*A_ONE;

for i = 1:length(row)
    test(row(i),col(i)) = 1;
end
figure
subplot(1,2,1)
spy(test);
subplot(1,2,2)
spy(test_rhs);


A_BS = A(1:40,[1:20,41:60]);
u_BS = A_BS\[rhs(1:20);zeros(20,1)];
u_BS = A_BS\[rhs([1:20,41:60])];%;zeros(20,1)];

precond = inv(A_BS-C(1:40,[1:20,41:60]));
cond(A_BS)
A_BS_precond=  precond*A_BS;
cond(A_BS_precond)

[row, col] = find(A_BS_precond >= tol);
test = 0*A_BS_precond;
for i = 1:length(row)
    test(row(i),col(i)) = 1;
end
figure
subplot(1,2,1)
spy(test)
subplot(1,2,2)
surf(A_BS_precond)
view(270,90)

[row,col] = find(A_BS>=tol);
test = 0*A_BS;
for i = 1:length(row)
    test(row(i),col(i)) = 1;
end
figure
subplot(1,2,1)
spy(test);
title(['non-zeros: ', num2str(length(row)/size(A_BS,1)^2)]);
subplot(1,2,2)
spy(test_rhs([1:20,41:60]));

test = 0*B;
[row,col] = find(A>=tol);
for i = 1:length(row)
    test(row(i),col(i)) = 1;
end
figure
subplot(1,2,1)
spy(test);
subplot(1,2,2)
spy(test_rhs);

figure
subplot(1,3,1)
plot(u(1:20))
hold on
plot(u_ONE(1:20),'*')
plot(u_BS(1:20),'--')
subplot(1,3,2)
plot(u(21:40))
hold on
plot(u_ONE(21:40),'*')
subplot(1,3,3)
plot(u(41:60))
hold on
plot(u_ONE(41:60),'*')
plot(u_BS(21:40),'--')
%%
CU = h5read([pwd2h5, 'nonlinearity-000000.h5'], '/A');
% lifting_rhs_vector_incompressibility.h5
% h5info([pwd2h5, 'lifting_rhs_vector_incompressibility.h5'])
l_inc = h5read([pwd2h5, 'lifting_rhs_vector_incompressibility.h5'], '/mean_vector');

rhs = h5read([pwd2h5, 'lifting_rhs_vector_mass.h5'], '/mean_vector');

A = L+B+BT+D+C+C1;

u = A\rhs;

u2 = A(1:5,1:5)\rhs(1:5);

cond(A)
% cond(
norm(B'+BT)

%%
pwd2h5 = '/home/ifam/fischer/Code/result/FEM/mu=0.001000/snapshot_supremizer/';

for i =  1:200
    pwdread = [pwd2h5, 'supremizer_' num2str(i-1,'%6.6i') ,'.h5'];
    SU(:,i) = h5read(pwdread, '/mean_vector');
end

[~,S,~] = svds(SU,200);

figure
semilogy(diag(S))
%%
sum(diag(S(1:13,1:13)).^2)/sum(diag(S).^2)


%%
clear all
close all
pwd2h5 = '/home/ifam/fischer/Code/result/FEM/long_refinements=4/mu=0.000667/snapshots/';

for i =  1:2001
    pwdread = [pwd2h5, 'snapshot_' num2str(i-1,'%6.6i') ,'.h5'];
    SU(:,i) = h5read(pwdread, '/velocity');
end
%%
mean_su = mean(SU');
%%
[~,S,~] = svds(SU,200);

%%
figure
semilogy(diag(S))
%%
sum(diag(S(1:9,1:9)).^2)/sum(diag(S).^2)