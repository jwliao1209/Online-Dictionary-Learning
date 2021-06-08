clear; clc;
%% Read the image and determine the inpainting part
I         = double(rgb2gray(imread('barbara_with_text.png'))) / 255;
M         = logical(I == 1);
J         = imgData('barbara.png');
J(M == 1) = 0;

% Get the basic information for the image
[r,c]         = size(J);
m             = 100;
sm            = sqrt(m);
TotalPatchNum = getPatchNum(r,c,sm);

%% Preparing the patch index from image that is uncontaminated
Uncontaminated = ones(r-sm+1, c-sm+1);
[x, y] = find(M == 1);
for ii = 1:numel(x)
    xmin = max(x(ii)-sm+1, 1);   xmax = min(x(ii), r);
    ymin = max(y(ii)-sm+1, 1);   ymax = min(y(ii), c);
    Uncontaminated(xmin:xmax, ymin:ymax) = 0;
end; clear x y ii xmin xmax ymin ymax
figure(); imshow(Uncontaminated);
Uncontaminated = find(Uncontaminated);

%% Construct the Contaminated set
Contaminated = setdiff(1:TotalPatchNum, Uncontaminated)';

%% ====================Inpaint==================
% Basic setup for recovering
lambda = 1.2 / sm;
lgood = 5;
load('Gray_ODL_barbara.mat');

X = imgInpaint_v1(J, M, D, lgood, lambda, Contaminated);

figure(2)
subplot(221); imshow(I); title('Original Image');
subplot(222); imshow(M); title('Pixel that need to inpaint');
subplot(223); imshow(J); title('Contaminated Image');
subplot(224); imshow(X); title('Reocver Image');

figure(3);
subplot(121); imshow(J); title('Contaminated Image');
subplot(122); imshow(X); title('Recovered Image');
