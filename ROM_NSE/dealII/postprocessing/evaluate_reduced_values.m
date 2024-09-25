clear all
close all

P = importdata("/home/ifam/fischer/Code/result/ROM/vp1/reduced_pressure_modes.txt");

for i = 1:size(P,2)
    figure(i)
    plot(1:length(P),P(:,i))
end

%%
close all
V = importdata("/home/ifam/fischer/Code/result/ROM/vp1/reduced_velocity_modes.txt");

for i = 1:size(V,2)
    figure(i)
    plot(1:length(V),V(:,i))
end
%%
close all
V = importdata("/home/ifam/fischer/Code/result/ROM/vp1/reduced_velocity_modes.txt");

for i = 31:size(V,2)
    figure(i)
    semilogy(1:length(V),abs(V(:,i)))
end
%%
% close all
% mean_vector = h5read('/home/ifam/fischer/Code/result/POD/vp1/mean_vector.h5','/mean_vector');
% 
% plot(mean_vector);
figure
plot(P(1,:))