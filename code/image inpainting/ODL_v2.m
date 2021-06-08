% Online dictionary learning (ODL) ver 1.1
% with mini batch size and handing fixed-size data
% Ref: Julien Mairal at el. (Online Dictionary Learning for Sparse Coding)
% Input:
%   Train: Training signals
%    Test: Test signals
% BatSize: Batch size
%  lambda: Regulaization parameter
%       D: Initial dictionary
%   Epoch: Number of epoch

function [D] = ODL_v2(Train, Test, lambda, D, BatSize, Epoch, INN_ITER)
    % Get the basic information (say, dimensions)
    [m, k] = size(D);
    [~, n] = size(Train);
    NumBatch = ceil(n/BatSize);
    
    % Basic setup
    A = zeros(k, k);    % A = sum(alpha_i * alpha_i')
    B = zeros(m, k);    % B = sum(x_i * alpha_i')
    S = zeros(k, n);    % Record the sparse coding for every training data
    QUIET = 1;
    
    Dold = D;
    
    for ii = 1 : Epoch
        fprintf('%d th epoch...\n', ii);
        
        % Shuffle the Training set
        I = randperm(n);
        Train = Train(:, I);
        S = S(:,I);
        
        % Run through the training set
        for jj = 1:NumBatch
            fprintf('  Batch: %d / %d >> ', jj, NumBatch);
            
            % Pick a batch
            if (jj ~= NumBatch)
                BatchIdx = (jj-1)*BatSize+1 : jj*BatSize;
            else
                BatchIdx = (jj-1)*BatSize+1 : n;
            end
            X = Train(:,BatchIdx);
            
            % Sparse coding step
            [alpha, ~, flag] = Z_subproblem(X, D, lambda, INN_ITER, QUIET);
            
            % Update A, B and S       
            A = A + alpha * alpha' - S(:,BatchIdx) * S(:,BatchIdx)';
            B = B + X * alpha' - X * S(:,BatchIdx)';
            S(:,BatchIdx) = alpha;
            
            % Dictionary update step
            [D, iter, res] = DictUpdate(D, A, B, 100, 1e-6);
            fprintf(' Z-flag: %d, D-iter: %d, res: %d, diff: %d\n', flag, iter, res, norm(D-Dold,'fro'));
            Dold = D;
        end
        
        % Inference
        [totval, avgval, reclos] = obj_func(Test, D, lambda, INN_ITER);
        fprintf('Total: %d, Avg: %d, RecoverLoss: %d\n', totval, avgval, reclos);
    end
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