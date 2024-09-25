close all
clear variables

pass2non = '/home/ifam/fischer/Code/result/ROM/vp1/matrices/';

A = []; 
for i = 1:1
    A = [A, h5read([pass2non, 'nonlinearity-',num2str(i-1,'%6.6i'),'.h5'],'/A')] ;
end

%%
% A = [A, 0*h5read([pass2non, 'nonlinearity-',num2str(i-1,'%6.6i'),'.h5'],'/A')];
% N=size(A,1)*size(A,2)
% K=1:floor(sqrt(N))+2;
% D = K(rem(N,K)==0);
% d1 = D(end);
% d2 = N/d1;
% A = reshape(A,[d1,d2]);

%%
[U, S, V] = svds(A,max([size(A,1),size(A,2)]));

for i = 1:size(S,1)
   energy(i) = sum(diag(S(1:i,1:i)).^2)/sum(diag(S).^2);
end
find(energy>=0.9999,1)
figure
semilogy(diag(S))
figure
semilogy(energy)

t = 10;

norm(A-U(:,1:t)*S(1:t,1:t)*V(:,1:t)')
norm(A-U(:,1:t)*S(1:t,1:t)*V(:,1:t)','fro')