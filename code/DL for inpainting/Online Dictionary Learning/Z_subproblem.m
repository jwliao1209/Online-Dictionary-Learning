function [Z, val, flag] = Z_subproblem(X, D, lambda, MAX_ITER, QUIET)
    % solving the sparse coefficient subproblem
    % That is argmin(0.5*|X-D*Z|_F^2 + lambda*|Z|_(1,1)) over Z
    % Rewrite L_rho(Z,Y;U) = 0.5*|X-D*Z|_F^2 + lambda*|Y|_(1,1)
    
    % Basic setup
    rho = 4.5;
    ABS_TOL = 1e-6;  % Ref. to Boyd. sec 3.3.1 (Stop criteria)
    REL_TOL = 1e-4;  % Ref. to Boyd. sec 3.3.1
    alpha   = 1.618;   % Ref. to Boyd. sec 3.4.3 (Over relexation)
    
    ITER = 1;        % Iteration number
    res_pri = inf;   % primal residual 
    res_dual = inf;  % dual residual   
    eps_pri = 0;     % primal tolerance 
    eps_dual = 0;    % dual tolerance
    
    % Initial information and guess
    [~, n] = size(D);
    [~, N] = size(X);
    
    Z = zeros(n, N);
    Y = zeros(n, N);
    U = zeros(n, N);
    
    [ML, MU] = factorize(D, rho);
    checkpoint = 1; step_len = 20;
    if (~QUIET)
        fprintf('%4s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
            'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
    end
  
    while (ITER <= MAX_ITER && (res_pri > eps_pri || res_dual > eps_dual))        
        % Z-update
        Z = solver(D, ML, MU, rho, D'*X+rho*(Y-U));
        Zhat = alpha * Z + (1-alpha) * Y;   % over-relation with alpha in [1.5,1.8]
        
        % Y-update
        Yold = Y;
        Y = soft_thresholding(Zhat+U, lambda/rho);
        
        % U-update
        U = U + Zhat - Y;
        
        % Calculate the residual and tolerance
        res_pri = norm(vec(Y-Z));
        res_dual = norm(vec(rho * (Y - Yold)));
        eps_pri = sqrt(numel(Z)) * ABS_TOL + ...
                  max([norm(Z(:)), norm(Y(:))]) * REL_TOL;
        eps_dual = sqrt(numel(Z)) * ABS_TOL + ...
                  norm(U(:)) * REL_TOL;
        
        % print the information every step_len-steps
        if (~QUIET && ITER >= checkpoint)
            fprintf('%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.6f\n', ITER, ...
                res_pri, eps_pri, res_dual, eps_dual, obj(X,D,Z,lambda));
            checkpoint = checkpoint + step_len;
        end
        
        ITER = ITER + 1;
    end
    
    % Calculate the loss
    val = obj(X, D, Z, lambda);
    
    % Give a flag that indicates convergence or not
    if (res_pri < eps_pri && res_dual < eps_dual)
        flag = 0;
    elseif (res_pri >= eps_pri && res_dual >= eps_dual)
        flag = 3;
    elseif (res_pri >= eps_pri)
        flag = 1;
    else
        flag = 2;
    end
end

% LU-factorization for D^T*D+rho*In
function [L, U] = factorize(D, rho)
    [m, n] = size(D);
    % If m < n (i.e, D is a "fat" matrix), then D^T*D is "huge"
    % So, it is more efficient to factorize D*D^T+rho*Im
    % and recover (D^T*D+rho*In)^(-1) by shermann-morrison
    if (m > n)
        [L, U] = lu(D'*D+rho*eye(n));
    else
        [L, U] = lu(D*D'+rho*eye(m));
    end
end

% solve (D^T*D+rho*In)x=b
% with (D^T*D+rho*In=LU or D*D^T+rho*Im=LU, see the function above)
function [x] = solver(D, L, U, rho, b)
    [m, n] = size(D);
    if (m > n)
        x = U \ (L \ b);
    else
        x = (b-D'*(U\(L\(D*b))))/rho;
    end
end

% soft-thresholding
% Recall that S_lambda(x) = sign(x) .* max(0, x-lambda)
function y = soft_thresholding(x, lambda)
    y = sign(x) .* max(0, abs(x)-lambda);
end

% vertorize a matrix
function v = vec(x)
    v = x(:);
end

% Calculate the objective value (value)
function val = obj(X, D, Z, lambda)
    [~, n] = size(X);
    val = (0.5 * norm(X-D*Z, 'fro')^2 + lambda * sum(abs(Z(:))));
end