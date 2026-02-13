%% Running time on a matrix completion problem
% Author: Guillaume Olikier (2025-11-01)
% This script computes, for N randomly generated instances of a matrix
% completion problem, the running time and number of iterations needed by
% each of the following methods to bring the objective function below f_tol:
%   1. monotone PGD [OW25, Algorithm 4.2 with l = 0 or p = 1];
%   2. P2GD [SU15, Algorithm 3];
%   3. P2GDR [OGA25, Definition 6.1];
%   4. P2GD-PGD [OGA25, Definition 7.1];
%   5. RFD [SU15, Algorithm 4];
%   6. RFDR [OA23, Algorithm 3].
% The final iterate, function value, B-stationarity measure, smallest
% singular value, and measure of the lack of orthonormality are also
% computed.
N = 100;
f_tol = 1e-15;
%% Problem parameters
m = 450;
n = 300;
r = 15;
% Weight matrix
W = cell([N 1]);
dim = m*n;
for i = 1:N
    W{i} = zeros(dim, 1);
    ObsEntries = randperm(dim, dim/20);% dim/20 < (m+n-r)*r
    W{i}(ObsEntries) = ones(dim/20, 1);
    W{i} = reshape(W{i}, [m n]);
end
clear dim ObsEntries
% Matrix to be approximated
s_A = rand(N, r);
U_A = cell([N 1]);
V_A = cell([N 1]);
for i = 1:N
    U_A{i} = randn(m, r);
    [U_A{i}, ~] = qr(U_A{i}, 'econ');
    V_A{i} = randn(n, r);
    [V_A{i}, ~] = qr(V_A{i}, 'econ');
end
%% Methods parameters
a = 0.8;
b = 0.5;
c = 0.1;
Delta = 0.01;
Delta_ = 0.1;
%% Initialization
% Performance criteria
time = zeros(N, 9);
obj = zeros(N, 9);
B = zeros(N, 9);
sigma_r = zeros(N, 9);
iter = zeros(N, 9);
orthU = zeros(N, 9);
orthV = zeros(N, 9);
% Factorization of the first and last iterates
s0 = zeros(N, r);
U0 = cell([N 1]);
V0 = cell([N 1]);
s = cell([N 9]);
U = cell([N 9]);
V = cell([N 9]);
%% Running time
for i = 1:N
    % Objective function and gradient
    A = (U_A{i}.*s_A(i, :))*V_A{i}';
    f0 = @(X) 0.5*norm(sqrt(W{i}).*(X-A), 'fro')^2;
    f1 = @(X) W{i}.*(X-A);
    g0 = @(L, R) f0(L*R');
    g1 = @(L, R) f1(L*R');
    % Initial iterate from [SU15, section 3.4]: projection of the negative gradient at zeros(m, n) onto the variety 
    [U0{i}, S0, V0{i}] = svds(-f1(zeros(m, n)), r, 'largest', 'Tolerance', 1e-14, 'MaxIterations', 300);
    s0(i, :) = diag(S0)';
    % PGD
    [s{i, 1}, U{i, 1}, V{i, 1}, obj(i, 1), time(i, 1), iter(i, 1)] = PGDtime(r, s0(i, :), U0{i}, V0{i}, f0, f1, a, b, c, f_tol);
    % P2GD
    [s{i, 2}, U{i, 2}, V{i, 2}, obj(i, 2), time(i, 2), iter(i, 2)] = P2GDtime(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, f_tol);
    % P2GDR with Delta
    [s{i, 3}, U{i, 3}, V{i, 3}, obj(i, 3), time(i, 3), iter(i, 3)] = P2GDRtime(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, Delta, f_tol);
    % P2GDR with Delta_
    [s{i, 4}, U{i, 4}, V{i, 4}, obj(i, 4), time(i, 4), iter(i, 4)] = P2GDRtime(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, Delta_, f_tol);
    % P2GD-PGD with Delta
    [s{i, 5}, U{i, 5}, V{i, 5}, obj(i, 5), time(i, 5), iter(i, 5)] = P2GDPGDtime(r, s0(i, :), U0{i}, V0{i}, f0, f1, g0, g1, a, b, c, Delta, f_tol);
    % P2GD-PGD with Delta_
    [s{i, 6}, U{i, 6}, V{i, 6}, obj(i, 6), time(i, 6), iter(i, 6)] = P2GDPGDtime(r, s0(i, :), U0{i}, V0{i}, f0, f1, g0, g1, a, b, c, Delta_, f_tol);
    % RFD
    [s{i, 7}, U{i, 7}, V{i, 7}, obj(i, 7), time(i, 7), iter(i, 7)] = ERFDtime(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, 0, f_tol);
    % RFDR with Delta
    [s{i, 8}, U{i, 8}, V{i, 8}, obj(i, 8), time(i, 8), iter(i, 8)] = ERFDRtime(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, 0, Delta, f_tol);
    % RFDR with Delta_
    [s{i, 9}, U{i, 9}, V{i, 9}, obj(i, 9), time(i, 9), iter(i, 9)] = ERFDRtime(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, 0, Delta_, f_tol);
end
clear S0
%% Smallest singular value and lack of orthogonality
for i = 1:N
    A = (U_A{i}.*s_A(i, :))*V_A{i}';
    f0 = @(X) 0.5*norm(sqrt(W{i}).*(X-A), 'fro')^2;
    f1 = @(X) W{i}.*(X-A);
    g0 = @(L, R) f0(L*R');
    g1 = @(L, R) f1(L*R');
    for j = 1:9
        [~, ~, ~, ~, B(i, j)] = P2GDmap(r, length(s{i, j}), s{i, j}, U{i, j}, V{i, j}, g0, g1, a, b, c);
        sigma_r(i, j) = s{i, j}(r);
        orthU(i, j) = norm(U{i, j}'*U{i, j}-eye(r), 'fro');
        orthV(i, j) = norm(V{i, j}'*V{i, j}-eye(r), 'fro');
    end
end
clear j A f0 f1 g0 g1 S0
%% Generate tables
TimeTable = [min(time)' median(time)' max(time)'];
IterTable = [min(iter)' median(iter)' max(iter)'];
% [~, fastestN] = min(time, [], 2);
% fastest = zeros(9, 1);
% for i = 1:9
%     fastest(i) = sum(fastestN == i);
% end
clear i