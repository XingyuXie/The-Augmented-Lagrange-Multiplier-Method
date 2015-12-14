function [ S_out ] = binary_S( S_in )
%BINARY_S 此处显示有关此函数的摘要
%   此处显示详细说明
[m,n] = size(S_in);
S_out = zeros(m,n);
[~,max_index] = max(S_in);
for i=1:n
    S_out(max_index(i),i) = 1;
end

end

