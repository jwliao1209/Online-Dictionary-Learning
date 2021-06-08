clear; clc;

load('Gray_ODL_dict.mat');
load('GTest.mat');
[~,n] = size(TestData);

id = randperm(n,1);
rpix = TestData(:,id);    % real block

mask = randperm(100,5);
M = zeros(100,1); M(mask) = 1;
cpix = rpix;
cpix(mask) = 1;    % contaminated block

[fpix, Z] = SparseCoding(rpix, D, 1.2/10, M, 10);

figure(1);
subplot(2,2,1:2); imshow(reshape(rpix,10,10)); title('original block');
subplot(2,2,3); imshow(reshape(cpix,10,10)); title('contaminated block');
subplot(2,2,4); imshow(reshape(D*Z,10,10)); title('recovered block');
