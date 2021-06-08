% Input
% x     : (m * 1) signal vector
% D     : (m * n) dictionary matrix
% lambda: a positive penalty parameter
% Ouput
% z     : (n * 1) sparse coefficient vector such that
% z = argmin_z (0.5*|x-Dz|_2^2 + lambda*|z|_1)
function [z, iter, pri_res, dual_res] = SR_ADMM(x, D, lambda, rho)
    % Stop criterion
    MAX_ITER = 1e4;
    eps_pri = 1e-6;
    eps_dual= 1e-6;
    
    % Information
    [~, n] = size(D);
    
    % Initial: (mu: multiplier) constraint (z = y)
    z = zeros(n,1);
    y = zeros(n,1);
    mu = zeros(n,1);
    
    % Allocate memory to record residual
    pri_res = zeros(MAX_ITER,1);
    dual_res= zeros(MAX_ITER,1);
    
    % Update z we need to solve z = (D^T*D + rho*I) \ (D^T*x + rho*(y-mu))
    A = D'*D + rho*speye(n);
    R = chol(A);
    RT = R';
    DTx = D'*x;
    
    temp_check = 500;
    for iter = 1 : MAX_ITER
        if (iter >= temp_check)
            fprintf('iter %d times\n', temp_check);
            fprintf('obj val : %d, pri_res : %d, dual_res : %d\n', obj_val(x,D,z,lambda), pri_res(iter-1), dual_res(iter-1));
            temp_check = temp_check + 500;
        end
        % update z = (D^T*D + rho*I) \ (D^T*x + rho*(y-mu))
        z = R \ (RT \ (DTx + rho*(y-mu)));
        
        % update y = soft_threshold_(lambda/rho)(z+mu)
        yold = y;
        y = soft_threshold(z+mu, lambda/rho);
        
        % update mu = mu + z - y
        mu = mu + z - y;
        
        % Check the stop criteria
        pri_res(iter) = norm(z - y);
        dual_res(iter)= norm(rho*(y-yold));
        if (pri_res(iter) < eps_pri && dual_res(iter) < eps_dual)
            break;
        end
    end
    pri_res = pri_res(1:iter);
    dual_res = dual_res(1:iter);
end

function val = obj_val(x, D, z, lambda)
    val = 0.5 * norm(x-D*z)^2 + lambda * norm(z,1);
end

function y = soft_threshold(x, kappa)
    y = max(0, x-kappa) - max(0, -x-kappa);
end