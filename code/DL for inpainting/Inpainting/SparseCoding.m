% This function is used to reconstruct the pixel that is containminated
% Only the sparse coding part

% Input argument
% block = the block that we need to do sparse encoding
% D = a dictionary that trained by good pixels
% llasso = the lasso parameter
% mask = mask
% sm = the square root of the patch size m

% Goal: find argmin(0.5*|m.*block - m.*(D*alpha)|_2^2 + llasso * |alpha|_1)
function [block, Z] = SparseCoding(block, D, llasso, mask, sm)
    m = sm * sm;
    block = reshape(block, m, 1);
    mask = reshape(mask, m, 1);
    
    Mask_D = D(~mask, :);
    Mask_x = block(~mask, :);
    MAX_ITER = 2000;
    QUIET = 1;
    
    [Z, ~, ~] = Z_subproblem(Mask_x, Mask_D, llasso, MAX_ITER, QUIET);
    if (~QUIET)
        fprintf('Z-problem loss: %d\n', 0.5 * norm(Mask_x - Mask_D * Z, 'fro')^2);
    end
    temp = D * Z;
    block(mask == 1) = temp(mask == 1);
    block = reshape(block, sm, sm);
end