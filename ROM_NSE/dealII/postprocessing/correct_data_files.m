clear all;
close all;

input_dir = '/media/hendrik/hard_disk/Nextcloud/Code/result/FEM/standard_data/';
output_dir = '/media/hendrik/hard_disk/Nextcloud/Code/result/FEM/';

offset = 1;
beginn = 500-offset;

start = 3.1;
ende = 4.1;

switch1 = 2;
switch2 = 0;

folders = dir([input_dir, 'mu*']);
folders = {folders.name};
%%
for i = 1:length(folders) %1:length(folders)
   disp(folders{i})
   mkdir([output_dir,char(folders{i})]);
   mkdir([output_dir,char(folders{i}), '/snapshots']);
   mkdir([output_dir,char(folders{i}), '/solution']);
   drag = importdata([input_dir, folders{i} ,'/drag.txt']);
   lift = importdata([input_dir, folders{i} ,'/lift.txt']);
   pressure = importdata([input_dir, folders{i} ,'/pressure.txt']);
   
   index1_drag = find(drag(:,1)==start,1,'last')+switch1;
   index2_drag = find(drag(:,1)==ende,1,'last')+switch2;
   index1_lift = find(lift(:,1)==start,1,'last')+switch1;
   index2_lift = find(lift(:,1)==ende,1,'last')+switch2;
   index1_pressure = find(pressure(:,1)==start,1,'last')+switch1;
   index2_pressure = find(pressure(:,1)==ende,1,'last')+switch2;
%    
%    disp(num2str([
%                 index1_drag,index2_drag;
%                 index1_lift,index2_lift;
%                 index1_pressure,index2_pressure
%                 ]));
   
   drag = drag(index1_drag:index2_drag,:);
   lift = lift(index1_lift:index2_lift,:);
   pressure = pressure(index1_pressure:index2_pressure,:);
%    writematrix(drag, [output_dir, folders{i} ,'/drag.txt'],'Delimiter',',');
   dlmwrite( [output_dir, folders{i} ,'/drag.txt'], drag,'Delimiter',',','precision',17);
   dlmwrite( [output_dir, folders{i} ,'/lift.txt'], lift,'Delimiter',',','precision',17);
   dlmwrite( [output_dir, folders{i} ,'/pressure.txt'], pressure,'Delimiter',',','precision',17);
   
   files = dir([input_dir, folders{i}, '/snapshots/snap*']);
    files = {files.name};
    files_vtk = dir([input_dir, folders{i}, '/solution/sol*']);
    files_vtk = {files_vtk.name};
    
    index_snap1=1;
    index_snap2=1;
    
   for j = 1:length(files)
        time = h5read([char(input_dir), char(folders{i}), '/snapshots/', char(files(j))], '/time');
        if abs((time+1.6) - start) <=1e-5
          index_snap1 = j+1;
        end
        if abs((time+1.6) - ende) <=1e-5
          index_snap2 = j;
        end
   end
    
   disp(num2str([index_snap1, index_snap2]))
    
    for j = index_snap1:index_snap2
%         disp([files(j); files(j-offset+1)])
%         disp([input_dir, folders{i}, '/snapshots/', files(j)]);
%         disp([output_dir, folders{i}, '/snapshots/', files(j-offset+1)]);
        copyfile([char(input_dir), char(folders{i}), '/snapshots/', char(files(j))],[char(output_dir), char(folders{i}), '/snapshots/', char(files(j-index_snap1+1))]);
        copyfile([char(input_dir), char(folders{i}), '/solution/', char(files_vtk(j))],[char(output_dir), char(folders{i}), '/solution/', char(files_vtk(j-index_snap1+1))]);
    end
end

%%


