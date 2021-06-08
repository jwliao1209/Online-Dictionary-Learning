clear; clc;
%% Read the image and determine the inpainting part
I            = imgData('kino.jpg');
M            = (rgb2gray(imread('mask2.png')) == 0);
Temp         = repmat(M,1,1,3); 
J            = I;
J(Temp == 1) = 0; clear Temp;

% Get the basic information for the image
[r,c,~]       = size(J);
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
load('Gray_ODL_dict.mat');

X = zeros(r,c,3);
X(:,:,1) = imgInpaint_v1(J(:,:,1), M, D, lgood, lambda, Contaminated);
X(:,:,2) = imgInpaint_v1(J(:,:,2), M, D, lgood, lambda, Contaminated);
X(:,:,3) = imgInpaint_v1(J(:,:,3), M, D, lgood, lambda, Contaminated);

figure(2)
subplot(221); imshow(I); title('Original Image');
subplot(222); imshow(M); title('Pixel that need to inpaint');
subplot(223); imshow(J); title('Contaminated Image');
subplot(224); imshow(X); title('Reocver Image');