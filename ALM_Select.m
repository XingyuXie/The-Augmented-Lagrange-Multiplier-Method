function [S_hat, Y_hat] = ALM_Select(X, A, k, alpha, tol, maxIter)

% Dec 5 2015 
% This matlab code implements the augmented Lagrange multiplier method for
% min_{S,Y}  ||X - ASY||_F^2 (+ alpha* ||Y||_F^2)   s.t.  S^T*S = I , S >= 0
%
% X - Dim¡Án matrix of observations/data (required input)
% A - Dim¡Ám matrix of the clustering observations/data (required input) 
% k - the number need to select (required input)


% alpha - weight on sparse error term in the cost function
%       - DEFAULT 1e-6 if omitted or -1.
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-8 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
% 
[~, m] = size(A);
[~, n] = size(X);
if m < k
    disp('k must be smaller than m!');
    return
end

if nargin < 4 
    alpha = 1e-6;
end

if nargin < 5
    tol = 1e-8;
end

if nargin < 6
    maxIter = 1000;
end

% initialize
Y_hat = rand(k, n);
S_hat = zeros( m, k);
Q = zeros( m, k);
J = zeros( m, k);
Z_1 = zeros( m, k);
Z_2 = zeros( m, k);
%parameters 
tol = tol * m * k;
rho = 1.1126;      % this one can be tuned
mu = 1;             % this one can be tuned

 %variable
 ATA = A'*A;
 alpha_I = alpha * eye(k);
 ATX = A'*X;
 
iter = 0;
while iter < maxIter       
    iter = iter + 1;
    %% Get Y
    Y_hat = (S_hat'*ATA*S_hat + alpha_I)\(S_hat'*ATX);
   
    %% Get S
    right_side = ATX*Y_hat' + 0.5*(mu* (Q+J) - Z_1 - Z_2);
    Vec_S = (kron(Y_hat*Y_hat',ATA) + mu*eye(m*k))\ reshape(right_side, [], 1);
    S_hat = reshape(Vec_S, m, k);
    
    %% Get Q
    [U,~,V] = svd(mu*S_hat+Z_1, 'econ');
    Q = U*V';
    
    %% Get J
    process_matrix = (sign(S_hat + Z_2/mu) + 1) * 0.5;
    J = (S_hat + Z_2/mu).* process_matrix;
    
    %% Update
    Z_1 = Z_1 + mu*(S_hat - Q);
    Z_2 = Z_2 + mu*(S_hat - J);
    mu = mu * rho;
    
    %% Stop Criterion    
    if norm(S_hat - J, 'fro') < tol &&  norm(S_hat - Q, 'fro') < tol
        disp(['iter =', num2str(iter)]);
        break;
    end     
end