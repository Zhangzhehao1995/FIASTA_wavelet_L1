function [X_out, X_iter]=fista_wavelet_lasso_backtracking(observe, A, para, ref)
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
% For backtracking
if (flag && isfield(para,'L0'))
    L = para.L0;
else
    L = 0.0001;
end
if (flag && isfield(para,'eta'))
    eta = para.eta;
else
    eta = 1.5;
end

%% Initialization
[~,n]=size(A);
res = sqrt(n);
assert(res == fix(res), 'The shape of the image should be a square.');
X_image = reshape(A'*observe, res, res);
% X and Y are in wavelet domain, column vector
[X_iter, s]=wavedec2(X_image, wlevel, wname);
X_iter = X_iter';
Y = X_iter;
b = observe;
t_new=1;

% function: do wavelet decompose for image domain data
function out = vec_wavedec(vecimage)
    % input: vector in imgae domain
    % output: column vector in wavelet domain
    image = reshape(vecimage, res, res);
    [cof,~] = wavedec2(image, wlevel, wname);
    out = cof'; 
end

% function: do wavelet reconstruction for wavelet coefficient
function vectorImg = vec_waverec(wavecof)
    % output: column vector in imgae domain
    image = waverec2(wavecof', s, wname);
    vectorImg = image(:); 
end

% function: calculate optimization func value(data fidelity + the l1 norm)
% fun_val = fx + gx = 1/2*||AWrec(X)-b||^2+lambda*||X||_1
function Fx = F_val(X_iter)
    % input: wavelet coff, vector form 
    fx = 1/2*norm(A*vec_waverec(X_iter)-b)^2;
    gx = lambda*norm(X_iter,1);
    Fx = fx + gx;
end

% function: calculate Q(x,y)=f(y)+<x-y,f'(y)>+L/2||X-Y||_2+g(x)
function out = Q_val(x,y,L)
    % input: x,y are vector, representing wavelet coff
	f_y = A*vec_waverec(y)-b;
	term1 = 1/2*norm(f_y)^2;
	gfy = vec_wavedec(A'*f_y);
	term2 = (x-y)'*gfy;
	term3 = L/2 * (x-y)' * (x-y);
	term4 = lambda*norm(x,1);  
	out = term1 + term2 + term3 + term4;
end

%% Iteration
fprintf('\nStart FISTA iteration...\n');
fprintf('=========================================\n')
fprintf('#iter   current_L    Fun-val    Error-val\n');
fprintf('=========================================\n')
for i=1:maxIter
    % Backtracking for L
    Lbar = L; 
    while true
        fy_bt = A*vec_waverec(Y)-b;
        softy_bt = Y-(1/Lbar)*vec_wavedec(A'*fy_bt);
        ply_bt = abs(softy_bt)-lambda/L;
        pLbar_y = sign(softy_bt).*((ply_bt>0).*ply_bt);
        F = F_val(pLbar_y);
        Q = Q_val(pLbar_y, X_iter, Lbar);
        if F <= Q
            break
        end
        Lbar = Lbar*eta; 
        L = Lbar;
    end
    % Store the old value of the iterate and the update t-constant
    X_old = X_iter;
    t_old = t_new;
    t_new=(1+sqrt(1+4*t_old^2))/2;
    
    % Calculating the derivation of f(y) for the proximal problem pL(Y) 
    fy = A*vec_waverec(Y)-b;
    softy = Y-(1/L)*vec_wavedec(A'*fy);
   
    % Soft thresholding to update X
    ply = abs(softy)-lambda/L;
    X_iter = sign(softy).*((ply>0).*ply);
     
    % Update Y
    Y = X_iter+(t_old-1)/t_new*(X_iter-X_old);

    % Calculate the optimization function value and mean error with refImg
    X_image = waverec2(X_iter', s, wname);
    fun_val =  F_val(X_iter);
    error = abs(X_image - ref);
    error_val = norm(error(:))/n;
    
    % printing the information of the current iteration
    fprintf('%3d      %1.4f     %7.6g       %6.3g\n',i, L, fun_val, error_val);
end

X_out = X_image;
end
