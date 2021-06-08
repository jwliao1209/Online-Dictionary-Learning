% Get the r*c (0,1)-matrix indicates the appearance of the ith patch
function Mask = getPatchMask(r, c, sm, ipatch) 
    Mask = zeros(r,c, numel(ipatch));
    [i, j] = getPatchPosition(r, c, sm, ipatch);
    for ii = 1 : numel(ipatch)
        Mask(i(ii):i(ii)+sm-1, j(ii):j(ii)+sm-1, ii) = 1;
    end
end