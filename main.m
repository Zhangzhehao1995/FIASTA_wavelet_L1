clc;
clear;
close all
%% Load
% Load data
load project_data
figure;
imagesc(sinogram)
title('sinogram')
figure;
imagesc(imgref)
title('imgref')

observe = sinogram(:);
% Load functions
addpath('utils/')
addpath('fista/')

%% Construct system matrix
% sino = A*img, where img should be column-wise arranged as img(:)
% use reshape(sino, projections, angles) to convert to a martix-form

% Load pre-calculated system matrix if have system_matrix.mat file
if exist('system_matrix.mat','file')   
    load system_matrix
else
    % Size of the region of interest (unit: m)
    L =  0.06144 ;
    % Number of pixels in each direction
    npixels = 256;
    % pixel size
    pixel_size = L/npixels;
    % Numer of views
    nviews = 540;
    % Angle increment between views (unit: degree)
    dtheta = 5/12;
    % Views
    views = (0:nviews-1)*dtheta ;
    % Numer of rays for each view
    nrays = 512;
    % Distance between first and last ray (unit: pixels)
    d = npixels*(nrays-1)/nrays;
    % Construct imaging operator (unit: pixels)
    A = paralleltomo(npixels, views, nrays, d) ;
    % Rescale A to physical units (unit: m)
    A = A* pixel_size;
    save system_matrix A
end

%% Optimization problem
% Use Lasso(Least Absolute Shrinkage and Selection Operator) regularization
% argmin{1/2*||A*Wrec(x)-b||^2+lambda*||x||_1}
para.maxIter    = 200;
para.waveName   = 'haar';
para.waveLevel  = 3;
para.lambda     = 2e-5;
para.L0         = 0.0001;
para.eta        = 1.1;

% X_out = fista_wavelet_lasso(observe, A, para);
X_out = fista_wavelet_lasso_backtracking(observe, A, para, imgref);

figure;
imagesc(X_out)
title('Recon')

error_img = abs(X_out - imgref);
figure;
imagesc(error_img)
title('Error')
% norm(x-X_out1(:))