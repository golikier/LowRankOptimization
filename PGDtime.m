function [s, U, V, f, time, i] = PGDtime(r, s, U, V, f0, f1, a, b, c, f_tol)
%% Description
% Author: Guillaume Olikier (2025-10-12)
% This function implements monotone PGD [OW25, Algorithm 4.2 with l = 0 or p = 1]
% on the real determinantal variety.
% Input:
%   - a positive integer r;
%   - a row vector s of at most r positive real numbers in decreasing order;
%   - an m-by-length(s) matrix U having orthonormal columns, where m > r;
%   - an n-by-length(s) matrix V having orthonormal columns, where n > r;
%   - functions f0 and f1 that, given an m-by-n matrix, return respectively
%     the objective function and its gradient at that matrix;
%   - a positive real number a;
%   - b and c in (0, 1);
%   - a positive real number f_tol.
% Output:
%   - the first (s, U, V) such that f0((U.*s)*V') <= f_tol;
%   - the value f of f0 at (U.*s)*V';
%   - the running time required to generate (s, U, V);
%   - the number of iterations required to generate (s, U, V).
% This function computes only what is necessary to generate (s, U, V).
tic
f = f0((U.*s)*V');
i = 0;
while f > f_tol && toc <= 60
    [s, U, V, f] = PGDmap(r, s, U, V, f0, f1, a, b, c);
    i = i+1;
end
time = toc;
end
