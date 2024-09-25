clear all
disp('load data')
A_block = importdata("../M_block.txt");
A_full = importdata("../M_full.txt");
disp('loaded data')
B_data = importdata("../B.txt");
%%
M_size_block = max(max(A_block(1:2,:)));
M_size_full = max(max(A_full(1:2,:)));
M_block = sparse(M_size_block);
M_full = sparse(M_size_full);

B_row = max(max(A_block(1,:)));
B_col = max(max(A_block(2,:)));
B
disp('edit mtx')
for i = 1:length(A_block')
    M_block(A_block(2,i)+1,A_block(3,i)+1) = A_block(1,i);
end

for i = 1:length(A_full')
    M_full(A_full(2,i)+1,A_full(3,i)+1) = A_full(1,i);
end

for i = 1:length(B_data')
    B(A_full(2,i)+1,A_full(3,i)+1) = A_full(1,i);
end


clear A_block A_full
disp('edited mtx')
%%
disp('nonozeros1')
[row1,col1]= find(M_block~=0);
M_block_ind = sparse(M_size_block);
for i = 1:length(row1)
    M_block_ind(row1(i),col1(i)) = 1;
end
disp('nonozeros2')
[row2,col2]= find(M_full~=0);
M_full_ind = sparse(M_size_full);
for i = 1:length(row2)
    M_full_ind(row2(i),col2(i)) = 1;
end
disp('nonozerosed')
%%
close all

figure('Name','indizes')
subplot(1,2,1)
spy(M_block_ind);
subplot(1,2,2)
spy(M_full_ind);



figure('Name','values')
subplot(1,2,1)
spy(M_block);
subplot(1,2,2)
spy(M_full)

figure('Name','diffs')
subplot(1,2,1)
spy(M_block_ind-M_full_ind)
subplot(1,2,2)
spy((M_full-M_block));
% 

sum(sum(abs(M_block-M_full)))
sum(sum(abs(M_block-M_full)))/sum(sum(abs(M_full)))