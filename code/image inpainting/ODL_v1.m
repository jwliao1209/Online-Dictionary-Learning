% Online dictionary learning (ODL) ver 1.1
% with mini batch size and handing fixed-size data
% Ref: Julien Mairal at el. (Online Dictionary Learning for Sparse Coding)
% Input:
%   Train: Training signals
%    Test: Test signals
%      BS: Batch size
%  lambda: Regulaization parameter
%       D: Initial dictionary
%       T: Number of iteration

function [D] = ODL_v1(Train, Test, BS, lambda, D, T, INN_ITER)
    % Get the basic information (say, dimensions)
    [m, k] = size(D);
    [~, n] = size(Train);
    
    % Basic setup
    A = zeros(k, k);    % A = sum(alpha_i * alpha_i')
    B = zeros(m, k);    % B = sum(x_i * alpha_i')
    S = zeros(k, n);    % Record the sparse coding for every training data
    Catch = zeros(n, 1);
    QUIET = 1;
    
    checkpoint = floor(T / 10);
    outputstep = checkpoint;    
    
    for ii = 1 : T
        % Random some indices
        I = randperm(n, BS);
        Catch(I) = Catch(I) + 1;
        
        % pick BS data from the training set
        X = Train(:, I);
        
        % Sparse coding step
        [alpha, ~, flag] = Z_subproblem(X, D, lambda, INN_ITER, QUIET);
        
        % Update A, B and S
        
        % The method in paper (I don't think this is work)
        %if (ii < BS)
        %    beta = 1 - BS/(ii*BS+1);
        %else
        %    beta = 1 - BS / (BS^2 - BS + ii);
        %end
        %A = beta * A + alpha * alpha' - S(:,I) * S(:,I)';
        %B = beta * B + X * alpha' - X * S(:,I)';
        
        A = A + alpha * alpha' - S(:,I) * S(:,I)';
        B = B + X * alpha' - X * S(:,I)';
        S(:,I) = alpha;
        
        % Dictionary update step
        [D, iter, res] = DictUpdate(D, A, B, 100, 1e-6);
        fprintf('%d iter, flag in Z-problem: %d, iter in Dictionary update: %d and res: %d, %d\n',...
                 ii, flag, iter, res, any(isnan(D(:))));
        Rand_reg(Catch);
        
        % Inference
        if (ii >= outputstep)
            fprintf('Step %d with test cost:%d\n', ii, obj_func(Test, D, lambda, INN_ITER, QUIET));
            outputstep = outputstep + checkpoint;
        end
    end
end

% Compute the objective function over the test set X
% f_n(D) = sum(l(x_i, D))/n;
% l(x_i, D) = min_y (0.5 * |x_i-Dy|_2^2 + lambda * |y|_1)
function val = obj_func(X, D, lambda, INN_ITER, QUIET)
    fprintf('=========Start checking objective function over the test set========\n');
    % How many signal does X have?
    [~, n] = size(X);
    fprintf('There are %d signals in the test set\n', n);
    val = 0;
    
    %percent = 1;
    %for ii = 1 : n
    %    if (ii >= floor(n*percent/100))
    %        fprintf('Finish %d percent\n', percent);
    %        percent = percent + 1;
    %    end
    %    [~, l] = Z_subproblem(X(:,ii), D, lambda, INN_ITER, ~QUIET);
    %    val = val + l;
    %end
    
    [~, l] = Z_subproblem(X, D, lambda, INN_ITER, ~QUIET);
    val = l;
end

% This function do nothing, just help me to debug
function Rand_reg(Catch)
    fprintf('Max: %d, Remaining: %d\n', max(Catch), length(find(~Catch))); 
end