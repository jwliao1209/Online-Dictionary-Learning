% the starting position of the ipatch-th patch in a r × c image
function [i, j] = getPatchPosition(r, c, sm, ipatch)    
    % Check ipatch is legal or not
    if  (any(ipatch) < 1 || any(ipatch) > getPatchNum(r,c,sm))
        error('ipatch out of index');
    end
    [i, j] = ind2sub([r-sm+1,c-sm+1], ipatch);
end