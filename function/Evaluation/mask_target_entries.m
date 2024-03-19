function [Y] = mask_target_entries(target_train, per)



%============gen random integer nos======
[n,m] = size(target_train);
x=randi([1,n],1,n)';
%========================================
Y = target_train;

sample_per = floor(per*n);
for i = 1 : sample_per
   Y(x(i),:) = 0; 
end


end