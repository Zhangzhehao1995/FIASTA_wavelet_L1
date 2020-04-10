function [X_out, fun_val]=fista_lasso(observe, A, para, ref)
% Require Wavelet Toolbox
% INPUTS:
%
% observe ............................. The observed / acquired data
% A ................................... The system matrix for CT scan
% para ................................ Parameters structure
% ref ................................. Reference image
%
% OUTPUT
% 
% X_out ............................... Solution of the problem (image)
% X_iter .............................. Solution of the problem (wavelet)
% fun_val ............................. Value of the optimization function
%                                       1/2*||A(X)-b||^2+lambda*||X||_1
%% Assigning parameters according to para and/or default values
flag = exist('para', 'var');
if (flag && isfield(para,'maxIter'))
    maxIter=para.maxIter;
else
    maxIter=200;
end
if(flag && isfield(para,'waveName'))
    wname = para.waveName;
else
    wname = 'haar';
end
if (flag && isfield(para,'waveLevel'))
    wlevel = para.waveLevel;
else
    wlevel = 3;
end
if (flag && isfield(para,'lambda'))
    lambda = para.lambda;
else
    lambda = 2e-5;
end

%% Initialization
[~,n]=size(A);
res = sqrt(n);
assert(res == fix(res), 'The shape of the image should be a square.');

X_iter = A'*observe;
Y = X_iter;
b = observe;
t_new = 1;

%The Lipschitz constant of the gradient of 1/2*||A(X)-b||^2
% L = max(max(A.^2));
L=max(eig(A'*A));

%% Iteration
fprintf('Start FISTA iteration\n');
fprintf('*********************\n');
fprintf('#iter  fun-val\n==============================\n');
for i=1:maxIter
    % Store the old value of the iterate and the update t-constant
    X_old = X_iter;
    t_old = t_new;
    t_new=(1+sqrt(1+4*t_old^2))/2;
    
    % Calculating the derivation of f(y) for the proximal problem pL(Y) 
    fy = A*Y-b;
    softy = Y-(1/L)*A'*fy;
    
    % Soft thresholding to update X
    ply = abs(softy)-lambda/L;
    X_iter = sign(softy).*((ply>0).*ply);
     
    % Update Y
    Y = X_iter+(t_old-1)/t_new*(X_iter-X_old);

    % Compute the data fidelity term and the l1 norm of the wavelet transform
    % fun_val = 1/2*||A(X)-b||^2+lambda*||X||_1
    fx = 1/2*norm(A*X_iter-b)^2;
    gx = lambda*norm(X_iter,1);
    fun_val =  fx + gx;

    % printing the information of the current iteration
    fprintf('%3d    %1.5g\n',i,fun_val);

end

X_out = X_iter;
end
