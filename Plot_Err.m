function  [err] = Plot_Err(min_k, max_k, X, A)
% Dec 8 2015 
% This matlab code implements the figure of reconstruction err and k
%
% Min_k - The point the figure start from (required input)
% Max_k - The point where the figure stop  (required input)


%[~, m] = sixe(A);
err = zeros(1,max_k-min_k+1);
for index_k = min_k:max_k
    [~,~,reconstruct_err] = ALM_Select2(X, A, index_k);
%    S_tmp = zeros(m,index_k);
%    [~,max_index] = max(S);
%    for i=1:index_k
%        S_tmp(max_index(i),i) = 1;
%    end
%    reconstruct_err = norm(X-A*S_tmp*Y,'fro');
    err(index_k - min_k) = reconstruct_err;
end
%x = min_k:max_k;
%plot(x,err);

end

