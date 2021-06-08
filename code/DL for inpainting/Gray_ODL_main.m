clear; clc; addpath(genpath('.'));

% Some parameters
sm = 10;          % patch width
m  = sm * sm;
k  = 256;        % atoms (columns) in dictionary
bs = 256;        % batch size
Epochs   = 2;
INN_ITER = 2000; % Max inner iteration number in Z-subproblem


% Main Online dictionary learning for gray image
GrayPath  = '.\Training image.\Gray Image.\';
ImgsName  = extractfield( dir(strcat(GrayPath, '*.png')), 'name' );
TrainImgs = cell( numel(ImgsName), 1 );
n         = zeros(numel(ImgsName), 1);  % number of patches in each image
for ii = 1 : numel(ImgsName)
    TrainImgs{ii} = imgData( ImgsName{ii} );
    [r,c,~] = size(TrainImgs{ii});
    n(ii)       = getPatchNum(r,c,sm);
end; clear GrayPath ImgsName r c ii

% Training
D = rand(m, k); D = D./sqrt(sum(D.^2,1));
load('GTest.mat');

[D] = ODL_gray(TrainImgs, n, TestData, D, bs, Epochs, INN_ITER);