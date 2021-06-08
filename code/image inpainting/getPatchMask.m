% Get the r*c (0,1)-matrix indicates the appearance of the ith patch
function Mask = getPatchMask(r, c, sm, ipatch) 
    Mask = zeros(r,c);
    [i, j] = getPatchPosition(r, c, sm, ipatch);
    Mask(i:i+sm-1, j:j+sm-1) = 1;
end