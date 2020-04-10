% Size of the region of interest (unit: mm)
L = 61.44;
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
% Rescale A to physical units (unit: mm)
V = A* pixel_size;

[m,n] = size(A);
psize = 500; % chunk size
count = 0;
while count < n
    p = min(psize,n-count);
    j = count+(1:p);
    E(:,j) = A'*A(:,j);
    count = count + p;
end
