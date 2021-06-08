function X = imgInpaint_v1(X, M, D, lgood, llasso, BadPixId)
    [r, c] = size(X); 
    [m, ~] = size(D);
    sm = sqrt(m);
    tol = 3;
    Max_sampled = 10000;
    Type = 1;
    
    t = 0;
    while (sum(M(:)) > 0)
        t = t + 1;
        fprintf('step: %d, %d pixels needs to recover\n', t, sum(M(:)));
        C = lgood * (1 - M);
        Xnew = X .* C;
        Sample_id = BadPixId(randperm(numel(BadPixId), Max_sampled));
        for ii = 1 : Max_sampled
            patch = getPatch(X, sm, Sample_id(ii));
            mask = getPatch(M, sm, Sample_id(ii));
            if (sum(mask(:)) > 0 && sum(mask(:)) < tol)
                p_prime = SparseCoding(patch, D, llasso, mask, sm, Type);
                [sr, sc] = getPatchPosition(r,c,sm,Sample_id(ii));
                xrange = sr : sr+sm-1;
                yrange = sc : sc+sm-1;
                C(xrange, yrange) = C(xrange, yrange) + 1;
                Xnew(xrange, yrange) = Xnew(xrange, yrange) + p_prime;
            end
        end
        X(C~=0) = Xnew(C~=0) ./ C(C~=0);
        M = (C == 0);
    end
end