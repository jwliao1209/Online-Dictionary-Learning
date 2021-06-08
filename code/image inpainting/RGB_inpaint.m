clear; clc;
%% Read the image and determine the inpainting part
I    = imgData('kino.jpg');
M    = (rgb2gray(imread('mask.png')) == 0);

figure();
Temp = repmat(M,1,1,3); J = I; J(Temp == 1) = 0;
subplot(221); imshow(I); title('Original image');
subplot(222); imshow(M); title('Pixels that need to be inpainted');
subplot(223); imshow(J); title('Contaminated image'); clear Temp;

%% Get the basic information from the image
[r,c,~]       = size(J);
m             = 64;
sm            = sqrt(m);
TotalPatchNum = getPatchNum(r,c,sm);

%% Preparing the patch index from image that is uncontaminated
Uncontaminated = ones(r-sm+1, c-sm+1);
[x, y] = find(M == 1);
for ii = 1:numel(x)
    xmin = max(x(ii)-sm+1, 1);   xmax = min(x(ii), r);
    ymin = max(y(ii)-sm+1, 1);   ymax = min(y(ii), c);
    Uncontaminated(xmin:xmax, ymin:ymax) = zeros(xmax-xmin+1, ymax-ymin+1);
end; clear x y ii xmin xmax ymin ymax
figure(); imshow(Uncontaminated);
Uncontaminated = find(Uncontaminated);

%% Construct the data set (Train + Test + Inpaint)
XR = getPatch(J(:,:,1), sm, Uncontaminated);
XG = getPatch(J(:,:,2), sm, Uncontaminated);
XB = getPatch(J(:,:,3), sm, Uncontaminated);

%% Basic setup for training
[m, n] = size(XR);
k = 256;
lambda = 1.2 / sm;
N_Train = floor(n * 0.95);

DR = rand(m, k); DR = DR./sqrt(sum(DR.^2, 1));
DG = rand(m, k); DG = DG./sqrt(sum(DG.^2, 1));
DB = rand(m, k); DB = DB./sqrt(sum(DB.^2, 1));

Randid = randperm(n);
XR = XR(:, Randid);
XG = XG(:, Randid);
XB = XB(:, Randid);
clear Randid;

Epochs = 3;
INN_ITER = 2000;

%% Training (3 dictionaries for R, G, B)
DR = ODL_v2(XR(:,1:N_Train), XR(:,N_Train+1:end), lambda, DR, 256, Epochs, INN_ITER);
DG = ODL_v2(XG(:,1:N_Train), XG(:,N_Train+1:end), lambda, DG, 256, Epochs, INN_ITER);
DB = ODL_v2(XB(:,1:N_Train), XB(:,N_Train+1:end), lambda, DB, 256, Epochs, INN_ITER);