function [s, U, V, f, time, i] = ERFDtime(r, s, U, V, g0, g1, a, b, c, subclass, f_tol)
%% Description
% Author: Guillaume Olikier (2025-10-12)
% This function implements four subclasses of ERFD, whose iteration map is
% [OA24, Algorithm 3.1]:
%   - RFD if subclass = 0;
%   - CRFD with the ith cone from [OA24, Table 6.1] if subclass = i.
% Input:
%   - a positive integer r;
%   - a row vector s of at most r positive real numbers in decreasing order;
%   - an m-by-length(s) matrix U having orthonormal columns, where m > r;
%   - an n-by-length(s) matrix V having orthonormal columns, where n > r;
%   - functions g0 and g1 that, given (L, R) with L m-by-k, R n-by-k, and
%     k <= r, return respectively the objective function and its gradient
%     at L*R';
%   - a positive real number a;
%   - b and c in (0, 1);
%   - subclass in {0, 1, 2, 3} for choosing between RFD and CRFD with
%     each of the three cones from [OA24, Table 6.1];
%   - a positive real number f_tol.
% Output:
%   - the first (s, U, V) such that g0(U.*s, V) <= f_tol;
%   - the value f of g0 at (U.*s, V);
%   - the running time required to generate (s, U, V);
%   - the number of iterations required to generate (s, U, V).
% The method is stopped if its running time exceeds 60 seconds.
% This function computes only what is necessary to generate (s, U, V).
tic
[m, ~] = size(U);
[n, ~] = size(V);
f = g0(U.*s, V);
i = 0;
while f > f_tol && toc <= 60
    [s, U, V, f] = ERFDmap(m, n, r, length(s), s, U, V, g0, g1, a, b, c, subclass);
    i = i+1;
end
time = toc;
end
