function [X_out, X_iter, fun_val]=fista_wavelet_lasso(observe, A, para, ref)
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

%The Lipschitz constant of the gradient of 1/2*||AWrec(X)-b||^2
% L = 2*max(max(abs(A).^2));
L = 0.0154;


function out = vec_wavedec(vecimage)
    % input: vector in imgae domain
    % output: column vector in wavelet domain
    image = reshape(vecimage, res, res);
    [cof,~] = wavedec2(image, wlevel, wname);
    out = cof'; 
end

function vectorImg = vec_waverec(wavecof)
    % output: column vector in imgae domain
    image = waverec2(wavecof', s, wname);
    vectorImg = image(:); 
end

%% Iteration
fprintf('\nStart FISTA iteration...\n');
fprintf('==============================\n')
fprintf('#iter  fun-val\n==============================\n');
for i=1:maxIter
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

    % Compute the data fidelity term and the l1 norm of the wavelet transform
    % fun_val = 1/2*||AWrec(X)-b||^2+lambda*||X||_1
    X_image = waverec2(X_iter', s, wname);
    fx = 1/2*norm(A*vec_waverec(X_iter)-b)^2;
    gx = lambda*norm(X_iter,1);
    fun_val =  fx + gx;

    % printing the information of the current iteration
    fprintf('%3d    %10.5g\n',i,fun_val);

end

X_out = X_image;
end
