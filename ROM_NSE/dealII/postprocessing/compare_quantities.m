close all;
clear all; 

path2fem = '/home/ifam/fischer/Code/result/FEM/';
path2rom = '/home/ifam/fischer/Code/result/ROM/greedy/';

path2fem = '/home/ifam/fischer/Code/result/FEM/long/';
path2rom = '/home/ifam/fischer/Code/result/ROM/vp1_long_15/';

% path2fem = '/home/ifam/fischer/Code/result/FEM/';
% path2rom = '/home/ifam/fischer/Code/result/ROM/vp1/';

RE =[100];
saven = 0;
plot_intervall = [9,10];
for i = 1:length(RE)

parameter = 0.1/RE(i);

duration = 1999;
start_rom=1;
pressure = importdata([path2fem, 'mu=', num2str(parameter,'%6.6f'), '/pressure.txt']);
pressure_rom = importdata([path2rom, 'mu=', num2str(parameter,'%6.6f'), '/pressure.txt']);
pressure = pressure(end-duration:end,:);
% pressure_rom = pressure_rom(end-duration:end,:);
dt = pressure(2,1)-pressure(1,1);
pressure(:,1) = pressure(:,1)-pressure(1,1)+dt;
figure
plot(pressure(start_rom:end,1),pressure(:,2));
hold on
plot(pressure_rom(:,1),pressure_rom(:,2));
legend('fom', 'rom')
% figure
% plot(pressure_rom(start_rom:end,1),abs(pressure(:,2)-pressure_rom(start_rom:end,2))./abs(pressure(:,2)));

drag = importdata([path2fem, 'mu=', num2str(parameter,'%6.6f'), '/drag.txt']);
drag_rom = importdata([path2rom, 'mu=', num2str(parameter,'%6.6f'), '/drag.txt']);
% drag_rom = drag_rom(end-duration:end,:);
drag = drag(end-duration:end,:);
drag(:,1) = drag(:,1)-drag(1,1)+dt;


lift = importdata([path2fem, 'mu=', num2str(parameter,'%6.6f'), '/lift.txt']);
lift_rom = importdata([path2rom, 'mu=', num2str(parameter,'%6.6f'), '/lift.txt']);
ylim([8,10])
% lift_rom = lift_rom(end-duration:end,:);
lift = lift(end-duration:end,:);
lift(:,1) = lift(:,1)-lift(1,1)+dt;


% figure
% plot(pressure_rom(:,1),pressure(:,2))c
% hold on
% plot(pressure_rom(:,1),pressure_rom(:,2));
% legend('full','rom');

if length(pressure(:,2)) == length(pressure_rom(:,2))
    if norm(pressure(:,2)-pressure_rom(:,2)) > 1e-10
        disp(['Achtung falscher Druck bei: ', num2str(RE(i))]);
    end
end

f{i} = figure('Name',['Re: ', num2str(0.1/parameter)]);
f{i}.Position(1:4) = [650*(i-1), 558, 625, 800];

subplot(4,1,1)
% plot(drag(end-duration:end,1)-drag(1,1),drag(end-duration:end,2));
plot(drag(start_rom:end,1),drag(:,2));
hold on
plot(drag_rom(:,1),drag_rom(:,2),'x');
xlim([drag_rom(1,1),drag_rom(end,1)])
xlim(plot_intervall)
grid on;
title('drag')
legend('fom','rom');

drag_error = abs((drag(1:end-1,2)-drag_rom(2:end,2)));
subplot(4,1,2)
semilogy(drag_rom(start_rom:end-1,1),drag_error);%./drag(:,2)));
xlim([drag_rom(1,1),drag_rom(end,1)])
grid on;
legend('error');
xlim(plot_intervall)

xlabel(['||drag_h - drag_N||/||drag_h||: ', num2str( norm(drag(:,2)-drag_rom(start_rom:end,2))/norm(drag(:,2)))])


subplot(4,1,3)
plot(lift(start_rom:end,1),lift(:,2));
hold on
plot(lift_rom(:,1),lift_rom(:,2));
xlim([drag_rom(1,1),drag_rom(end,1)])
grid on;
title('lift');
xlim(plot_intervall)
ylim([-1.1,1.1])
legend('fom','rom');

subplot(4,1,4)
lift_error = abs((lift(1:end-1,2)-lift_rom(2:end,2)));
semilogy(lift_rom(start_rom:end-1,1),lift_error);
xlim([drag_rom(1,1),drag_rom(end,1)])
grid on;
xlabel(['||lift_h - lift_N||/||lift_h||: ', num2str(norm(lift(:,2)-lift_rom(start_rom:end,2))/norm(lift(:,2)))])
legend('error');
xlim(plot_intervall)

% figure
% plot(drag_rom(start_rom:end,1),drag(:,2));
% hold on
% plot(drag_rom(:,1),drag_rom(:,2));
% xlim([drag_rom(1,1),drag_rom(end,1)])
% grid on;
% legend('fom','rom');
% if saven == 1
%     saveas(gcf,['drag_',num2str(RE(i))],'epsc');
% end
% 
% figure
% plot(lift_rom(start_rom:end,1),lift(:,2));
% hold on
% plot(lift_rom(:,1),lift_rom(:,2));
% xlim([drag_rom(1,1),drag_rom(end,1)])
% grid on;
% legend('fom','rom');
% if saven == 1
%     saveas(gcf,['lift_',num2str(RE(i))],'epsc');
% end

figure
subplot(2,1,1)
% plot(drag(end-duration:end,1)-drag(1,1),drag(end-duration:end,2));
plot(drag(start_rom:end,1),drag(:,2),'LineWidth',1);
hold on
plot(drag_rom(:,1),drag_rom(:,2),'--','LineWidth',2.0);
xlim([drag_rom(1,1),drag_rom(end,1)])
xlim(plot_intervall)
ylim([3.11,3.18])
grid on;
leg1 = legend('FOM','ROM','Interpreter','latex','FontSize',12,'Orientation','horizontal','Position',[0.2 0.92 0.15 0.0869]);
% legend('Line 1','Line 2','Position',[0.2 0.92 0.15 0.0869]);
% legend('Orientation','horizontal')
leg1 = legend('boxoff')
ylabel('$C_D$','Interpreter','latex','FontSize',14)
xlabel('$t\;$[s]','Interpreter','latex','FontSize',14)

subplot(2,1,2)
plot(lift(start_rom:end,1),lift(:,2),'LineWidth',1);
hold on
plot(lift_rom(:,1),lift_rom(:,2),'--','LineWidth',2.0);
xlim([drag_rom(1,1),drag_rom(end,1)])
grid on;
xlim(plot_intervall)
ylim([-1.1,1.1])

% set(leg1,'Box','off');
ylabel('$C_L$','Interpreter','latex','FontSize',14)
xlabel('$t\;$[s]','Interpreter','latex','FontSize',14)
end
