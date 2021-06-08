clear; clc;

% Read the image and determine the inpainting part
RGB = 3;  % 1R 2G 3B
I = imgData('kino.jpg'); I = I(:,:,RGB);
M = double(rgb2gray(imread('mask2.png'))) / 255;
Inpaint = logical(M == 0);

% Get the basic information for the image
[r,c] = size(I);
m = 100; sm = 10;
TotalPatchNum = getPatchNum(r,c,sm);

% Preparing the patch index from image that is uncontaminated
% Initially, assume all pixels is uncontaminate
Uncontaminated = ones(r-sm+1, c-sm+1);
[i,j] = find(Inpaint == 1);
for ii = 1:numel(i)
    xmin = max(i(ii)-sm+1, 1);
    xmax = min(i(ii), r);
    ymin = max(j(ii)-sm+1, 1);
    ymax = min(j(ii), c);
    Uncontaminated(xmin:xmax, ymin:ymax) = zeros(xmax-xmin+1, ymax-ymin+1);
end; clear i j ii xmin xmax ymin ymax
Uncontaminated = find(Uncontaminated);

% Construct the data set
X = getPatch(I, sm, Uncontaminated);

% Basic setup for training
[m, n] = size(X);
k = 256;
lambda = 1.2 / sm;

%D = rand(m, k);
%D = D ./ sqrt(sum(D.^2, 1));
load('Gray_ODL_dict.mat');

N_Train = floor(n * 0.995);
X = X(:, randperm(n));

Epochs   = 3;
INN_ITER = 2000;

%% Train

% Train the dictionary D and a sparse coefficent matrix Z
%D = ODL(X(:, 1:N_Train), X(:, N_Train+1:end), lambda, D, OUT_ITER, INN_ITER);
%D = ODL_v1(X(:, 1:N_Train), X(:, N_Train+1:end), 128, lambda, D, OUT_ITER, INN_ITER);
D = ODL_v2(X(:, 1:N_Train), X(:, N_Train+1:end), lambda, D, 256, Epochs, INN_ITER);