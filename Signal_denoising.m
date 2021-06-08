clear; clc;
% An application of Sparse representation problem (SR problem)
m = 512;
n = 2048;
norm0 = 32;
P = [0.5, 1, 5];
lambda = [5, 10, 20, 30];

%% Step 1: Generate the basic information
% Random a fixed dictionary matrix D (with size m*n)
D = rand(m, n)*2-1;

% Random a sparse vector z (with size n*1) with |z|_0 = norm0
z = rand(n, 1)*2-1;
z(randperm(n,n-norm0)) = 0;

% The true signal is computed by x = Dz
x = D*z;

% Generate white gaussian noise (wgn) with power P
noise = zeros(m, numel(P));
for ii = 1:numel(P)
    noise(:,ii) = wgn(m, 1, P(ii));
end

% So, the noise signal is xn = x + n
xn = x + noise;

%% Step 2: Denoise
% Sovling SR problem by using ADMM with different lambda
recover_z = zeros(n, numel(P));
lambda = 5;
for ii = 1 : numel(P)
    [recover_z(:, ii), iter, pri_res, dual_res] = SR_ADMM(xn(:, ii), D, lambda, 6);
    fprintf('finish a job\n');
end

for ii = 1 : numel(P)
    figure()
    subplot(211); 
    plot(1:m, x, 'r'); hold on; plot(1:m, xn(:,ii), 'g'); ylim([-20,20])
    subplot(212); 
    plot(1:m, x, 'r'); hold on; plot(1:m, D*recover_z(:,ii), 'g'); ylim([-20,20]);
    
    figure()
    subplot(211); 
    plot(1:n, z, 'r'); hold on; plot(1:n, D\x, 'g'); ylim([-4,4]);
    subplot(212); 
    plot(1:n, z, 'r'); hold on; plot(1:n, recover_z(:,ii), 'g'); ylim([-4,4]);
end