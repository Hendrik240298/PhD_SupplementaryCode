close all
clear all

A=importdata('/home/ifam/fischer/Code/result/POD/lonely_150_lift/error_iteration_greedy.txt');
A_com = importdata('/home/ifam/fischer/Code/result/POD/lonely_150_lift/error_comparer_greedy.txt');
RE = 100:10:150;
% RE = [100,120:10:150];

% A=importdata('/media/hendrik/hard_disk/Nextcloud/Code/result/POD/lonely2/error_iteration_greedy.txt');
% A_com = importdata('/media/hendrik/hard_disk/Nextcloud/Code/result/POD/lonely2/error_comparer_greedy.txt');
% 
% RE = 100;
% A=importdata('/media/hendrik/hard_disk/Nextcloud/Code/result/POD/greedy_mean2/error_iteration_greedy.txt');
% A_com = importdata('/media/hendrik/hard_disk/Nextcloud/Code/result/POD/greedy_mean2/error_comparer_greedy.txt');

% A=importdata('/home/ifam/fischer/Code/result/POD/greedy_no_mean2/error_iteration_greedy.txt');
% A_com = importdata('/home/ifam/fischer/Code/result/POD/greedy_no_mean2/error_comparer_greedy.txt');


% RE = [100,130,150];

size_surro = length(RE);

iterations = floor(size(A,1)/size_surro);
max_A = zeros(iterations,1);
max_A_index = zeros(iterations,1);
min_A = zeros(iterations,1);
min_A_index = zeros(iterations,1);
mean_A = zeros(iterations,1);
index = 1:size_surro;
plot_tmp  =0;
error_ind = 4;
%%
f{1} = figure;
f{1}.Position(1:3) = [300, 558, 625];
for i =1:iterations
    [max_A(i), max_A_index(i)] = max(A(index,2));
    [min_A(i), min_A_index(i)] = min(A(index,2));
    mean_A(i) = mean(A(index,2));
    if (i==1 || i == iterations)
        plot(RE,A(index,2));
    end
    hold on
    index = index +size_surro;
end
% legend(string(1:iterations));
legend(string([1,2,iterations]));
legend('initial','result','FontSize',14,'Location','northwest');


tmp = size(A,1) -iterations*size_surro;
if tmp >0 && plot_tmp ==1
    plot(RE(1:tmp),A(end-tmp+1:end,2))
    legend(string(1:iterations+1));
end
grid on;
xticks(RE);
xlabel('$Re$','Interpreter','latex','FontSize',16);
ylabel('$\Delta(Re)$','Interpreter','latex','FontSize',16);

saveas(gcf,'error_RE','epsc');
 f{3} = figure;
%  f{3}.Position(1:3) = [300, 558, 625];
 semilogy((1:size(max_A,1))*5,max_A)
%  semilogy(A_com(:,9))
%  hold on
%  plot(mean_A)
%  plot(min_A)
grid on;
% legend('max','mean','min')
xlim([5, 5*iterations]);
xlabel('$N_u$','Interpreter','latex','FontSize',16);
ylabel('$L^2$-error','Interpreter','latex','FontSize',16);
saveas(gcf,'error','epsc');


 f{7} = figure;
 f{7}.Position(1:3) = [300, 558, 625];
 semilogy(max_A(1:end-1)-max_A(2:end));
%  semilogy(A_com(:,9))
%  hold on
%  plot(mean_A)
%  plot(min_A)
grid on;
% legend('max','mean','min')
xlim([1, iterations]);
xlabel('#iterations','Interpreter','latex','FontSize',16);
ylabel('$chnange in \Delta(Re)$','Interpreter','latex','FontSize',16);
saveas(gcf,'error_change','epsc');



% f{4} = figure;
%  semilogy(abs(max_A(2:end)-max_A(1:end-1))./abs(max_A(1:end-1)))
%  hold on
%  semilogy(abs(mean_A(2:end)-mean_A(1:end-1))./abs(mean_A(1:end-1)))

 f{5} = figure;
plot(RE(max_A_index),1:iterations)

f{6} = figure;
plot(A_com(1:iterations,1),1:iterations)

index = 1:size_surro;

f{2} = figure;
f{2}.Position(1:3) = [1000, 558, 625];
for i =1:iterations
%     plot(RE,A_com(index,2));
    hold on
%     plot(RE,A_com(index,3));
    plot(RE,A_com(index,6));
%     plot(RE,A_com(index,5));
%     plot(RE,A_com(index,7));
    index = index +size_surro;
end

tmp = size(A_com,1) -iterations*11;
if tmp >0
%     plot(RE(1:tmp),A_com(end-tmp+1:end,2))
    hold on;
%     plot(RE(1:tmp),A_com(end-tmp+1:end,3))
    plot(RE(1:tmp),A_com(end-tmp+1:end,6))
%     plot(RE(1:tmp),A_com(end-tmp+1:end,5))
%     plot(RE(1:tmp),A_com(end-tmp+1:end,7))
end
legend('inf','2','M');
grid on;
xticks(RE);

%%
% close all
% path2h5 ='result/POD/POD_greedy/';
% eig_SVD =  h5read([path2h5, 'eigenvalues.h5'], '/mean_vector');
% semilogy(eig_SVD)