function [D] = D_subproblem(X, Z, MAX_ITER)
    % solving the sparse coefficient subproblem
    % That is argmin(0.5*|X-D*Z|_F^2 + ind(C, D)) over D
    % where ind(C, D) is the indicator function 
    % C = {D: m*n matrix with all column 2 norm <=1}
    
    % Basic setup
    rho = 1;
    ABS_TOL = 1e-4;
    REL_TOL = 1e-4;
    
    ITER = 1;
    res_pri = inf;   % primal residual 
    res_dual = inf;  % dual residual   
    eps_pri = 0;     % primal tolerance 
    eps_dual = 0;    % dual tolerance
    
    % Initial information and guess
    [n, ~] = size(Z);
    [m, ~] = size(X);
    
    D = zeros(m, n);
    G = zeros(m, n);
    H = zeros(m, n);
    
    [L, U] = factorize(Z, rho);
    checkpoint = 1000;
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
    
    while (ITER <= MAX_ITER && (res_pri > eps_pri || res_dual > eps_dual))
        % D-update
        D = solver(Z, L, U, rho, X*Z-rho*(G-H));
        
        % G-update
        Gold = G;
        G = proj(D);
        
        % H-update
        H = H + D - G;
        
        % Calculate the residual and tolerance
        res_pri = norm(vec(D-G));
        res_dual = norm(vec(rho*(G-Gold)));
        eps_pri = sqrt(numel(D)) * ABS_TOL + ...
                  max(norm(Z(:)), norm(G(:))) * RES_TOL;
        eps_dual = sqrt(numel(D)) * ABS_TOL + ...
                  norm(H(:)) * RES_TOL;
        
        % print out the information every 1000 steps
        if (ITER >= checkpoint)
            fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', ITER, ...
                res_pri, eps_pri, res_dual, eps_dual, obj(X,D,Z));
            checkpoint = checkpoint + 1000;
        end
        ITER = ITER + 1;
    end
end

% LU-factorization for Z*ZT-rho*In
function [L, U] = factorization(Z, rho)
    [n, N] = size(Z);
    % if n > N, (i.e, Z is a shink matrix)
    % then it is more efficient that we try to factorize Z^T*Z-rho*IN
    % and recover (Z*Z^T-rho*Im)^(-1) by using sherman-morrison
    if (N > n)
        [L, U] = lu(Z*Z^T-rho*eye(n));
    else
        [L, U] = lu(Z^T*Z-rho*eye(N));
    end
end

% Solve the linear system D * (Z*Z^T-rho*In) = b (with D:unknown)
% with (Z*Z^T-rho*In = LU or Z^T*Z-rho*IN = LU, see the functin above)
function D = solver(Z, L, U, rho, b)
    [n, N] = size(Z);
    if (N > n)
        D = (b / U) / L;
    else
        D = (((((b*Z)/U)/L)*Z') - b ) / rho;
    end
end

% normalize the column vector
function Gnew = proj(Gold)
    col_norm = sqrt(sum(Gold.^2, 1));
    col_norm(col_norm == 0) = 1;
    Gnew = Gold ./ col_norm;
end

% vectorize a matrix
function v = vec(x)
    v = x(:);
end

% calculate the objective function value
function val = obj(X, D, Z)
    val = 0.5 * norm(X-D*Z, 'fro');
end