% Dictionary update
% Ref: Julien Mairal at el. (Online Dictionary Learning for Sparse Coding)
% Input:
%     D: Dictionary (m-by-k array)
%     A: sum_(alpha_i * alpha_i') (k-by-k array)
%     B: sum_(x_i * alpha_i') (m-by-k array)

function [D, iter, res] = DictUpdate(D, A, B, MAX_ITER, TOL)
    % Get the basic information (say, dimensions)
    [~,k] = size(D);
    
    % Basic setup for the loop
    res = inf;
    iter = 1;
    Dold = D;
    
    % loop until converge
    while (res > TOL && iter <= MAX_ITER)
        % update the jth column
        for j = 1 : k
            u = D(:,j) + (B(:,j)-D*A(:,j))/A(j,j);
            D(:,j) = u/max(norm(u), 1);
        end
        res = norm(D-Dold);
        Dold = D;
        %fprintf('%d iter, with res: %d\n', iter, res);
        iter = iter + 1;
        if any(isnan(D))
            warning('stop here!!!!!!!!!!!!!!\n');
        end
    end
    
end
