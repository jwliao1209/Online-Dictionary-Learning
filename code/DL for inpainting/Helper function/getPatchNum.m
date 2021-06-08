% Number of possible sm × sm patches in a r × c image
function num = getPatchNum(r,c,sm)
    num = (r-sm+1)*(c-sm+1);
end