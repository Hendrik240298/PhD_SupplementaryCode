close all
clear all


n = 250;
X = linspace(0,1,n);
dx = X(2)-X(1);

k = 100;
mu = linspace(0.5,1,k);

g1 = 2;
g2 = -2;


A0 = 1/dx^2*toeplitz([-2 1 zeros(1,n-2)]);

A0(1,:) = [1 zeros(1,n-1)];
A0(end,:) = [ zeros(1,n-1) 1];

b0 = [g1 ;ones(n-2,1);g2 ];

u0 = A0\b0;


A = 1/dx^2*toeplitz([-2 1 zeros(1,n-2)]);
b = ones(n,1);
AA = zeros(n,n);

g1 = 0;
g2 = 0;

bb = [g1 ;zeros(n-2,1);g2 ];
tau = 1;
AA(1,1) = tau;
AA(end:end,end) = tau;
u = ones(n,1);
while max([abs(u(1)-g1),abs(u(end)-g2)]) > 1e-5
    A_solve = A +tau*AA;
    b_solve = b +tau*bb; 
    u = A_solve\b_solve;
    disp(['fom: ', num2str(tau), ': ', num2str(u(1))])
    tau = tau*2;
end
norm(u-u0)
tau = tau/2;

figure
plot(X,u0)
hold on
plot(X,u);

figure
plot(X,abs(u-u0));

%%
A_r = u0' * A0 * u0;
b_r = u0'*b0;
AA_r = u0' * (tau*AA) * u0;
bb_r = u0'*(tau*bb);
u_r = (A_r+AA_r)\(b_r+bb_r);
u_r = A_r\b_r;
u_L = u_r*u0;

figure
plot(X,u_L)


%%

S = zeros(n,k);
for i = 1:k
    S(:,i) = [1, zeros(1,n-1);  mu(i)*A0(2:n-1,:);  zeros(1,n-1),1]\b0;
end

[U, Sig,~] = svds(S,10);

close all
figure
% plot(X,u0);
% hold on
plot(X,S)
legend(num2str(mu'))

figure
semilogy(diag(Sig))

close all;
A_r = U' * A * U;
b_r = U'*b;
AA_r =  U' * (AA) * U;
bb_r =  U'*(bb);
u_r = (A_r+AA_r)\(b_r+bb_r);
u_L = U*u_r;
tau = 1e14;
while max([abs(u_L(1)-g1),abs(u_L(end)-g2)]) > 1e-5
    u_r = (A_r+tau*AA_r)\(b_r+tau*bb_r);
    u_L = U*u_r;
    disp(['rom: ', num2str(tau), ': ', num2str(u_L(1)-g1)])
    tau = tau*2;
end

figure
plot(X,u_L)
hold on;
plot(X,u)
plot(X,S(:,end))
legend('rom','fom','fom - normal bc');

figure
plot(X,abs(u-u_L))