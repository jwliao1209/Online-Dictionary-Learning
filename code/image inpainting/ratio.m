function [r, dx] = ratio(X, k, l, m, D, lambda, MAX_ITER)
xpixel = X(:,k:l);
[M,N] = size(A);
zpixel = Z_subproblem(xpixel, D, lambda, MAX_ITER, 1);
newx = D*zpixel;
dx= abs(newx-xpixel)*255;
%dx = reshape(dx,sqrt(m),sqrt(m));
r = length(find(dx<10))/(M*N);
end