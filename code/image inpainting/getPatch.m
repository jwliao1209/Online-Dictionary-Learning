% Get the ipatch from the matrix mat (a m*1 vector) --resize sm*sm--
function Block = getPatch(mat,sm,ipatch)
    [r, c] = size(mat);
    [i, j] = getPatchPosition(r,c,sm,ipatch);
    Block = zeros(sm*sm, numel(ipatch));
    
    for ii = 1 : numel(ipatch)
        Block(:,ii) = reshape(mat(i(ii):i(ii)+sm-1, j(ii):j(ii)+sm-1), sm*sm, 1);
    end
    
end