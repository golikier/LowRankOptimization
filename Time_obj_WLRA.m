%% Running time on a weighted low-rank approximation (WLRA) problem
% Author: Guillaume Olikier (2025-06-30)
% This script computes, for N randomly generated instances of a WLRA
% problem, the time needed by each of the following methods to bring the
% objective function below f_tol:
%   - monotone PGD [OW25, Algorithm 4.2 with l = 0 or p = 1];
%   - P2GD [SU15, Algorithm 3];
%   - RFD [SU15, Algorithm 4];
%   - P2GDR [OGA24, Definition 6.1];
%   - P2GD-PGD [OGA24, Definition 7.1];
%   - RFDR [OA23, Algorithm 3];
%   - HRTR [LKB23, Algorithm 1].
% As this goal might be out of reach of P2GD and RFD, these are
% stopped as soon as their running time exceeds 60 seconds. The final
% function value, B-stationarity measure, smallest singular value, and a
% measure of the lack of orthonormality are also computed.
f_tol = 1e-12;
N = 50;
%% Initialization
time = zeros(N, 7);
obj = zeros(N, 7);
B = zeros(N, 7);
sigma_r = zeros(N, 7);
orthU = zeros(N, 6);
orthV = zeros(N, 6);
i_apo = 1e4;
i_ZeroSingularValue = zeros(N, 1);
S_apo = cell([N i_apo]);
s_apo = cell([N i_apo]);
U_apo = cell([N i_apo]);
V_apo = cell([N i_apo]);
s_PGD = cell([N 1]);
U_PGD = cell([N 1]);
V_PGD = cell([N 1]);
s_P2GD = cell([N 1]);
U_P2GD = cell([N 1]);
V_P2GD = cell([N 1]);
s_RFD = cell([N 1]);
U_RFD = cell([N 1]);
V_RFD = cell([N 1]);
s_P2GDR = cell([N 1]);
U_P2GDR = cell([N 1]);
V_P2GDR = cell([N 1]);
s_P2GDPGD = cell([N 1]);
U_P2GDPGD = cell([N 1]);
V_P2GDPGD = cell([N 1]);
s_RFDR = cell([N 1]);
U_RFDR = cell([N 1]);
V_RFDR = cell([N 1]);
L_HRTR = cell([N 1]);
R_HRTR = cell([N 1]);
%% Problem parameters and initial iterate
m = 150;
n = 100;
r = 5;
r_ = 2;
%[U, ~] = qr(rand(m, r+r_), 'econ');
U = [eye(r+r_) ; zeros(m-r-r_, r+r_)];
%[V, ~] = qr(rand(n, r+r_), 'econ');
V = [eye(r+r_) ; zeros(n-r-r_, r+r_)];
U0 = U(:, 1:r);
V0 = V(:, 1:r);
a3 = cell([N 1]);
a2 = cell([N 1]);
W = cell([N 1]);
s0 = rand(N, r);
for i = 1:N
    a3{i} = randn(r_);
    a2{i} = randn(r-r_);
    W{i} = rand(m, n);
    s0(i, :) = sort(s0(i, :), 'descend');
end
%% Methods parameters
a = 0.8;
b = 0.5;
c = 0.1;
Delta = 0.01;
gamma = 1;
gamma_c = 0.5;
eta = 0.1;
for i = 1:N
    %% Problem parameters
    A = U(:, (r_+1):r)*a2{i}*V(:, (r_+1):r)' + U(:, (r+1):(r+r_))*a3{i}*V(:, (r+1):(r+r_))';
    f0 = @(X) 0.5*norm(sqrt(W{i}).*(X-A), 'fro')^2;
    f1 = @(X) W{i}.*(X-A);
    f2 = @(X, dX) W{i}.*dX;
    g0 = @(L, R) f0(L*R');
    g1 = @(L, R) f1(L*R');
    %% P2GD and RFD iterates based on [OGA25, Proposition 8.1]
    S_apo{i, 1} = diag(s0(i, :));
    s_apo{i, 1} = s0(i, :);
    U_apo{i, 1} = U(:, 1:r);
    V_apo{i, 1} = V(:, 1:r);
    j = 1;
    while s_apo{i, j}(r) > 0 && j < i_apo
        j = j+1;
        S_apo{i, j} = blkdiag(diag(((diag(ones(r_)-a*W{i}(1:r_, 1:r_))').^(j-1)).*s0(i, 1:r_)), (ones(r-r_)-a*W{i}(r_+1:r, r_+1:r)).^(j-1).*(diag(s0(i, r_+1:r))-a2{i})+a2{i});
        [U_hat, s_apo{i, j}, V_hat] = svd(S_apo{i, j});
        s_apo{i, j} = diag(s_apo{i, j})';
        U_apo{i, j} = U(:, 1:r)*U_hat;
        V_apo{i, j} = V(:, 1:r)*V_hat;
    end
    i_ZeroSingularValue(i) = j;
    %% Running time
    % PGD
    [s_PGD{i}, U_PGD{i}, V_PGD{i}, obj(i, 1), time(i, 1)] = PGDtime(r, s0(i, :), U0, V0, f0, f1, a, b, c, f_tol);
    [~, ~, ~, ~, B(i, 1)] = P2GDmap(r, length(s_PGD{i}), s_PGD{i}, U_PGD{i}, V_PGD{i}, g0, g1, a, b, c);
    sigma_r(i, 1) = s_PGD{i}(r);
    orthU(i, 1) = norm(U_PGD{i}'*U_PGD{i}-eye(r), 'fro');
    orthV(i, 1) = norm(V_PGD{i}'*V_PGD{i}-eye(r), 'fro');
    % P2GD
    [s_P2GD{i}, U_P2GD{i}, V_P2GD{i}, obj(i, 2), time(i, 2)] = P2GDtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, f_tol);
    [~, ~, ~, ~, B(i, 2)] = P2GDmap(r, length(s_P2GD{i}), s_P2GD{i}, U_P2GD{i}, V_P2GD{i}, g0, g1, a, b, c);
    sigma_r(i, 2) = s_P2GD{i}(r);
    orthU(i, 2) = norm(U_P2GD{i}'*U_P2GD{i}-eye(r), 'fro');
    orthV(i, 2) = norm(V_P2GD{i}'*V_P2GD{i}-eye(r), 'fro');
    % RFD
    [s_RFD{i}, U_RFD{i}, V_RFD{i}, obj(i, 3), time(i, 3)] = ERFDtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, 0, f_tol);
    [~, ~, ~, ~, B(i, 3)] = P2GDmap(r, length(s_RFD{i}), s_RFD{i}, U_RFD{i}, V_RFD{i}, g0, g1, a, b, c);
    sigma_r(i, 3) = s_RFD{i}(r);
    orthU(i, 3) = norm(U_RFD{i}'*U_RFD{i}-eye(r), 'fro');
    orthV(i, 3) = norm(V_RFD{i}'*V_RFD{i}-eye(r), 'fro');
    % P2GDR
    [s_P2GDR{i}, U_P2GDR{i}, V_P2GDR{i}, obj(i, 4), time(i, 4)] = P2GDRtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, Delta, f_tol);
    [~, ~, ~, ~, B(i, 4)] = P2GDmap(r, length(s_P2GDR{i}), s_P2GDR{i}, U_P2GDR{i}, V_P2GDR{i}, g0, g1, a, b, c);
    sigma_r(i, 4) = s_P2GDR{i}(r);
    orthU(i, 4) = norm(U_P2GDR{i}'*U_P2GDR{i}-eye(r), 'fro');
    orthV(i, 4) = norm(V_P2GDR{i}'*V_P2GDR{i}-eye(r), 'fro');
    % P2GD-PGD
    [s_P2GDPGD{i}, U_P2GDPGD{i}, V_P2GDPGD{i}, obj(i, 5), time(i, 5)] = P2GDPGDtime(r, s0(i, :), U0, V0, f0, f1, g0, g1, a, b, c, Delta, f_tol);
    [~, ~, ~, ~, B(i, 5)] = P2GDmap(r, length(s_P2GDPGD{i}), s_P2GDPGD{i}, U_P2GDPGD{i}, V_P2GDPGD{i}, g0, g1, a, b, c);
    sigma_r(i, 5) = s_P2GDPGD{i}(r);
    orthU(i, 5) = norm(U_P2GDPGD{i}'*U_P2GDPGD{i}-eye(r), 'fro');
    orthV(i, 5) = norm(V_P2GDPGD{i}'*V_P2GDPGD{i}-eye(r), 'fro');
    % RFDR
    [s_RFDR{i}, U_RFDR{i}, V_RFDR{i}, obj(i, 6), time(i, 6)] = ERFDRtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, 0, Delta, f_tol);
    [~, ~, ~, ~, B(i, 6)] = P2GDmap(r, length(s_RFDR{i}), s_RFDR{i}, U_RFDR{i}, V_RFDR{i}, g0, g1, a, b, c);
    sigma_r(i, 6) = s_RFDR{i}(r);
    orthU(i, 6) = norm(U_RFDR{i}'*U_RFDR{i}-eye(r), 'fro');
    orthV(i, 6) = norm(V_RFDR{i}'*V_RFDR{i}-eye(r), 'fro');
    % HRTR
    sqrt_s0 = sqrt(s0(i, :));
    L0 = U0.*sqrt_s0;
    R0 = V0.*sqrt_s0;
    [L_HRTR{i}, R_HRTR{i}, obj(i, 7), time(i, 7)] = HRTRtime(L0, R0, f0, f1, f2, gamma, gamma_c, eta, f_tol);
    [U_L, R_L] = qr(L_HRTR{i}, 'econ');
    [V_HRTR, s_HRTR, U_HRTR] = svd(R_HRTR{i}*R_L', 'econ');
    U_HRTR = U_L*U_HRTR;
    s_HRTR = diag(s_HRTR)';
    [~, ~, ~, ~, B(i, 7)] = P2GDmap(r, sum(s_HRTR > 0), s_HRTR, U_HRTR, V_HRTR, g0, g1, a, b, c);
    sigma_r(i, 7) = s_HRTR(r);
end
