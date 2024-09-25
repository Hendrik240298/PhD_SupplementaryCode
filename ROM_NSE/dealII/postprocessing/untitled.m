close all 
clear all

path2time = '/home/ifam/fischer/Code/result/ROM/vp1_long_40_release/time_counter.txt';

T = importdata(path2time);
% T(:,3) = 3*T(:,3);
%%
mean(T(5:end,:))/1e6

figure
plot(T)
hold on
plot([1, length(T)],[5e6,5e6])
legend('sum','solve','matrix','rhs','maximum')
xlim([5,length(T)]);

% T(:,3) = 3*T(:,3);
% figure
% plot(T)
% legend('sum','solve','3*matrix','rhs')
% xlim([5,length(T)]);

%%
pathloop{1} =  '/home/ifam/fischer/Code/result/ROM/vp1_long_10/time_counter.txt';
pathloop{2} =  '/home/ifam/fischer/Code/result/ROM/vp1_long_15/time_counter.txt';
pathloop{3} =  '/home/ifam/fischer/Code/result/ROM/vp1_long/time_counter.txt';
pathloop{4} =  '/home/ifam/fischer/Code/result/ROM/vp1_long_25/time_counter.txt';
pathloop{5} =  '/home/ifam/fischer/Code/result/ROM/vp1_long_30/time_counter.txt';
pathloop{6} =  '/home/ifam/fischer/Code/result/ROM/vp1_long_40/time_counter.txt';
pathloop{7} =  '/home/ifam/fischer/Code/result/ROM/vp1_long_50/time_counter.txt';


pathloopslow{1} = '/home/ifam/fischer/Code/result/ROM/vp1_long_10_slow/time_counter.txt';
pathloopslow{2} = '/home/ifam/fischer/Code/result/ROM/vp1_long_15_slow/time_counter.txt';
pathloopslow{3} = '/home/ifam/fischer/Code/result/ROM/vp1_long_20_slow/time_counter.txt';
pathloopslow{4} = '/home/ifam/fischer/Code/result/ROM/vp1_long_25_slow/time_counter.txt';
pathloopslow{5} = '/home/ifam/fischer/Code/result/ROM/vp1_long_30_slow/time_counter.txt';
pathloopslow{6} = '/home/ifam/fischer/Code/result/ROM/vp1_long_40_slow/time_counter.txt';
pathloopslow{7} = '/home/ifam/fischer/Code/result/ROM/vp1_long_50_slow/time_counter.txt';

time_FOM = importdata('/home/ifam/fischer/Code/result/FEM/long/mu=0.001000/exectution_time.txt');
time_FOM = str2double(time_FOM{1}(1:5))/2;

for i = 1:7
    T = importdata(pathloop{i});
    mean_T(i) = sum(T(5:end,1)/1e9);
end

for i = 1:7
    T = importdata(pathloopslow{i});
    mean_T_slow(i) = sum(T(5:end,1))/1e9;
end
Time = length(T(5:end,1))*0.005;
close all
figure
plot([10:5:30,40, 50],mean_T/Time)
hold on
plot([10:5:30,40,50],mean_T_slow/Time)
plot([10, 30],[1,1],'--','color','black');
text(10.1,1.1,'real-time','Interpreter','latex','FontSize',14)
grid on;
legend('$N_s=0$','$N_s=N_p$','Interpreter','latex','FontSize',16,'location','northwest');
% xticks(10:5:30);
xlim([10,25]);
xlabel('$N_u=N_p$','Interpreter','latex','FontSize',16);
ylabel('$T_{ROM}/T$','Interpreter','latex','FontSize',16);
saveas(gcf,['time_measurements'],'epsc');


figure
semilogy([10:5:30,40, 50],time_FOM./mean_T)
hold on
semilogy([10:5:30,40,50],time_FOM./mean_T_slow)
grid on;
legend('$N_s=0$','$N_s=N_p$','Interpreter','latex','FontSize',16,'location','northeast');
% xticks(10:5:50);

xlabel('$N_u=N_p$','Interpreter','latex','FontSize',16);
ylabel('$T_{FOM}/T_{ROM}$','Interpreter','latex','FontSize',16);
saveas(gcf,['time_measurements_speed_up'],'epsc');
%%
figure
plot(mean_T_slow./(mean_T))