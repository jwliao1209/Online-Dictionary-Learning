% This function is used to reconstruct the pixel that is containminated
% Only the sparse coding part

% Input argument
% block = the block that we need to do sparse encoding (should be 1d-array)
% D = a dictionary that trained by good pixels
% llasso = the lasso parameter
% mask = mask (1 indicates the place that need to inpaint)
% sm = the square root of the patch size m

% Goal: find argmin(0.5*|m.*block - m.*(D*alpha)|_2^2 + llasso * |alpha|_1)
function [block, Z] = SparseCoding(block, D, llasso, mask, sm, Type)
    m = sm * sm;
    block = reshape(block, m, 1);
    mask = reshape(mask, m, 1);
    
    Mask_D = D(mask == 0, :);
    Mask_block = block(mask == 0);
    
    MAX_ITER = 2000;
    QUIET = 1;
    
    [Z, ~, ~] = Z_subproblem(Mask_block, Mask_D, llasso, MAX_ITER, QUIET);
    if (Type == 1)
        % Type 1
        block = D * Z;
    elseif (Type == 2)
        % type 2
        block(mask > 0) = D(mask > 0, :) * Z;
    else
        error('No such choice for Type');
    end
    block = reshape(block, sm, sm);
end