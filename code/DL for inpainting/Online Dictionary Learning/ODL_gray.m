% Online dictionary learning for gray image
% Ref: Julien Mairal at el. (Online Dictionary Learning for Sparse Coding)

% Input:
% TrainImgs  : Training gray images, cell of p images
% TotPatches : Total number of patches for each image
%   TestData : Test data, should be a (m * T) matrix, each column specify a
%              testing data
%          D : Initial dictionary
%         BS : Batch Size
%      Epoch : Epochs
%   INN_ITER : Inner iteration of Z_subproblem

function [D] = ODL_gray(TrainImgs, TotPatches, TestData, D, BS, Epoch, INN_ITER)
    % Get the basic information (say, dimensions)
    [m, k]   = size(D);
    sm       = sqrt(m);
    lambda   = 1.2 / sm;
    n        = sum(TotPatches);
    NumBatch = ceil(n/BS);
    InFreq   = 0.25;           % infrence frequency
    InStep   = 0 : floor(n*InFreq) : n-floor(n*InFreq);
    InStep(1) = []; %remember to cancel
    
    % Basic setup
    Dprev = D;
    Aprev = zeros(k, k);    % A = sum(alpha_i * alpha_i')
    Bprev = zeros(m, k);    % B = sum(x_i * alpha_i')
    
    Acur  = zeros(k, k);
    Bcur  = zeros(m, k);
    QUIET = 1;
    
    RandID = randperm(n);
    Uppbdd = cumsum(TotPatches);
    Lowbdd = [1; Uppbdd+1]; Lowbdd(end) = [];
    
    X      = zeros(m, BS);  % Temporary for a batch
    
    for ii = 1 : Epoch
        fprintf('%d th epoch...\n', ii);
        
        % Run through the training set
        for jj = 0 : NumBatch-1     
            % Inference
            if (~isempty( find(InStep==jj, 1) ))
                [totval, avgval, reclos] = obj_func(TestData, D, lambda, INN_ITER);
                fprintf('Total: %d, Avg: %d, RecoverLoss: %d\n', totval, avgval, reclos);
            end
            
            fprintf('  Batch: %d / %d >> ', jj+1, NumBatch);
            % Pick a batch
            col_X = 1;
            temp  = RandID(jj*BS+1 : (jj+1)*BS);
            for kk = 1 : numel(Lowbdd)
                ExtractID = intersect(temp(temp >= Lowbdd(kk)), temp (temp <= Uppbdd(kk))) - Lowbdd(kk) + 1;
                X(:,col_X:col_X+numel(ExtractID)-1) = getPatch(TrainImgs{kk}, sm, ExtractID);
                col_X = col_X + numel(ExtractID);
            end
            
            % Sparse coding step
            [alpha, ~, flag] = Z_subproblem(X, D, lambda, INN_ITER, QUIET);
            
            % Update A, B and S       
            Acur = Acur + alpha * alpha';
            Bcur = Bcur + X * alpha';
            
            % Dictionary update step
            [D, iter, res] = DictUpdate(Dprev, Aprev+Acur, Bprev+Bcur, 100, 1e-6);
            fprintf('Epoch: %d/%d, Z-flag: %d, D-iter: %d, res: %d, diff: %d\n',ii, Epoch, flag, iter, res, norm(D-Dprev,'fro'));
            Dprev = D;
        end
        
        % A and B only store this and previous epoch information
        Aprev = Acur; Acur = zeros(k,k);
        Bprev = Bcur; Bcur = zeros(m,k);
    end
end

function [v] = vec(x)
    v = x(:);
end

% Compute the objective function over the test set X
% f_n(D) = sum(l(x_i, D))/n;
% l(x_i, D) = min_y (0.5 * |x_i-Dy|_2^2 + lambda * |y|_1)
function [totval, avgval, reclos] = obj_func(X, D, lambda, INN_ITER)
    fprintf('=========Start checking objective function over the test set========\n');
    QUIET = 1;
    % How many signal does X have?
    [~, n] = size(X);
    fprintf('There are %d signals in the test set\n', n);
    
    [Y, l] = Z_subproblem(X, D, lambda, INN_ITER, ~QUIET);
    totval = l;
    avgval = l/n;
    
    % Recover loss: sum(norm(x_i-D*y_i))/n
    reclos = sum(sqrt(sum((X-D*Y).^2,1)))/n;
end