% Online dictionary learning (ODL)
% Ref: Julien Mairal at el. (Online Dictionary Learning for Sparse Coding)
% Input:
%   Train: Training signals
%    Test: Test signals
%  lambda: Regulaization parameter
%       D: Initial dictionary
%       T: Number of iteration

function [D] = ODL(Train, Test, lambda, D, T, INN_ITER)
    % Get the basic information (say, dimensions)
    [m, k] = size(D);
    [~, n] = size(Train);
    
    % Basic setup
    A = zeros(k, k);    % A = sum(alpha_i * alpha_i')
    B = zeros(m, k);    % B = sum(x_i * alpha_i')
    QUIET = 1;
    
    checkpoint = floor(T / 100);
    outputstep = checkpoint;    
    
    for ii = 1 : T
        x = Train(:,randi(n));   % pick a data
        [alpha, ~, flag] = Z_subproblem(x, D, lambda, INN_ITER, QUIET);
        A = A + alpha * alpha';
        B = B + x * alpha';
        [D, iter, res] = DictUpdate(D, A, B, 100, 1e-6);
        fprintf('%d iter, flag in Z-problem: %d, iter in Dictionary update: %d and res: %d\n',...
                 ii, flag, iter, res);
        
        
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