clear all
close all

r = 50;
n = 500;
m = 20000; %space
A = rand(m,m);
Y = rand(m,n);
D = eye(n);
disp('start svd');
% [U,S,~] = svds(A,r);
U = rand(m,r);
S = diag(rand(r,1));
disp('end svd');
% W = U*S.^(1/2)*U';
UU = U*S.^(1/4);
% WW = UU*UU';
% norm(W-WW)

tic 
UU*UU'*Y;
toc
tic
UU*(UU'*Y);
toc
% norm(

%%

path2h5 = '/home/ifam/fischer/Nextcloud/Code/dealii/MasterThesis/navier_stokes/dealii/result/POD/';

sing = importdata([path2h5, 'sing_red_mass.txt']);

figure 
plot(sing)

sum(sing.^2)/(sum(sing.^2)+sing(end-1)*19000)
%%

% res = zeros(m,n);
% for i = 1:m
%     for j = 1:n
%         temp =0;
%         for l = 1:m
% %             temp = 0;
% %             for k = 1:r
% %                 temp = temp + S(k,k)*U(i,k)*U(l,k);
% %             end
% %             temp = (diag(S)'.*U(i,:))*U(l,:)';
%             temp = temp+ (diag(S)'.*U(i,:))*U(l,:)'*Y(l,j);
%         end
%         res(i,j) = D(j,j)*temp;
%     end
% end
% 
% res_test = U*S*U'*Y*D;
% 
% norm(res-res_test)
% res_test= U*S*U';
% 
% figure
% surf(res-res_test);