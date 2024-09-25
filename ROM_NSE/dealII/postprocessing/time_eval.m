close all 
clear all

path2res = '/media/hendrik/hard_disk/Nextcloud/Code/result/ROM/lonely/mu=0.001000/';
path2fem = '/media/hendrik/hard_disk/Nextcloud/Code/result/FEM/mu=0.001000_time_measure/execution_time.txt';
intervall = 5:5:100;

tmp = importdata(path2fem);
time_fem =str2double(tmp{1}(1:4)) + str2double(tmp{1}(8:10))/1000;

for i = 1:length(intervall)
    tmp = importdata([path2res, 'exectution_time_',num2str(intervall(i)),'.txt']);
    if tmp{1}(3) == 's'
        extime(i) =str2double(tmp{1}(1:2)) + str2double(tmp{1}(5:7))/1000;
    elseif tmp{1}(4) == 's'
        extime(i) =str2double(tmp{1}(1:3)) + str2double(tmp{1}(6:8))/1000;
    else
        tmp{1}
        extime(i) = 1;
    end 
    if i == 13
        extime(i) =str2double(tmp{1}(1:3)) + str2double(tmp{1}(6))/1000;
    end
end

 f{1} = figure;
 f{1}.Position(1:3) = [300, 558, 625];
plot(intervall,time_fem./extime)
grid on;
xlim([intervall(2), intervall(end)]);
xlabel('$N_h$','Interpreter','latex','FontSize',16);
ylabel('Speed-up','Interpreter','latex','FontSize',16);
saveas(gcf,'speed_up','epsc');