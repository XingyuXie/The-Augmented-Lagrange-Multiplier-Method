function [S_hat, S_0,err] = ALM_Select4(X, A, k, tol, maxIter)

% Dec 8 2015 
% This matlab code implements the augmented Lagrange multiplier method for
% min_{S,Y}  ||X - ASY||_F^2  s.t.  S^T*S = I , S >= 0  Yi = ek for some k
%
% X - Dim¡Án matrix of observations/data (required input)
% A - Dim¡Ám matrix of observations/data (required input)
% k - the number need to select (required input)


%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-8 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 2000, if omitted or -1.
% 
[Dim, n] = size(X);
[~, m] = size(A);
if m < k
    disp('k must be smaller than m!');
    return
end

if nargin < 4
    tol = 1e-10;
end

if nargin < 5
    maxIter = 2000;
end

% initialize
Y = zeros(k, n);
M = zeros(k, n);
Z_4 = zeros(k, n);
temp = randsample(1:m,k);
S_hat = zeros(m,k);
for i=1:k
    S_hat(temp(i),i) = 1;
end
S_0 = S_hat;
K = A*S_0;
Z_3 = zeros( Dim, k);
Q = zeros( m, k);
J = zeros( m, k);
Z_1 = zeros( m, k);
Z_2 = zeros( m, k);
%parameters 
rho = 1.1126;      % this one can be tuned
mu = 1e-6;         % this one can be tuned

 %variable
I_k = eye(k);
I_2m = eye(m)*2;
div_normF_A = 1.0/norm(A,'fro');
%inverse_XTX = inv(X'*X);
iter = 0;
while iter < maxIter       
    iter = iter + 1;
    
    %% Get K
    K = (2*(X*M') + mu*(A*S_hat) +Z_3)/(2*(M*M')+ mu*I_k);
    
    %% Get M
    M = (2*(K'*K) + mu*I_k)\(Z_4+mu*Y+2*(K'*X));
    
    %% Get S
    S_hat = (I_2m + A'*A)\(A'*K + Q + J - (A'*Z_3 + Z_1 + Z_2)*(1.0/mu));
    
    %% Get Y
    M_tmp_matrix = (M - Z_4*(1.0/mu));
    [~,max_index] = max(M_tmp_matrix);
    Y = zeros(k, n);
    for index_i = 1:n
        Y(max_index(index_i),index_i) = 1;  
    end
    %% Get Q
    [U,~,V] = svd(mu*S_hat+Z_1, 'econ');
    Q = U*V';
    
    %% Get J
    process_matrix = (sign(S_hat + Z_2*(1.0/mu)) + 1) / 2;
    J = (S_hat + Z_2*(1.0/mu)).* process_matrix;
    
    %% Update
    Z_1 = Z_1 + mu*(S_hat - Q);
    Z_2 = Z_2 + mu*(S_hat - J);
    Z_3 = Z_3 + mu*(A*S_hat - K);
    Z_4 = Z_4 + mu*(Y - M);
    mu = min(10^30, mu * rho);
    
    %% Stop Criterion    
    sC1 = max(max(abs(S_hat - J)));
    sC2 = max(max(abs(S_hat - Q)));
    sC4 = max(max(abs(Y - M)));
    sC3 = norm(A*S_hat - K,'fro')*div_normF_A;
    sC = max([sC1,sC2,sC3,sC4]);
    
    disp(['iter =', num2str(iter) ', mu=' num2str(mu) ',stopALM= ' num2str(sC)]);
    if sC < tol
        disp(['iter =', num2str(iter) ', stopALM= ' num2str(sC)]);
        break;
    end   
end
err = norm(X-A*S_hat*Y,'fro');
end

