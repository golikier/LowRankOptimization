%% Plots on a weighted low-rank approximation problem
% Author: Guillaume Olikier (2025-06-30)
% This script plots, for each of the following methods applied to a
% weighted low-rank approximation problem, the objective function and a
% B-stationarity measure as functions of the running time until a function
% value smaller than or equal to f_tol is reached:
%   - monotone PGD [OW25, Algorithm 4.2 with l = 0 or p = 1];
%   - P2GD [SU15, Algorithm 3];
%   - RFD [SU15, Algorithm 4];
%   - P2GDR [OGA24, Definition 6.1];
%   - P2GD-PGD [OGA24, Definition 7.1];
%   - RFDR [OA23, Algorithm 3];
%   - HRTR [LKB23, Algorithm 1].
load('WLRA_time_obj_data_randn.mat')
i = 18; % minimum running time of HRTR among the 50 instances
%i = 44; % minimum running time of HRTR among the 14 instances on which P2GD fails
sqrt_s0 = sqrt(s0(i, :));
L0 = U0.*sqrt_s0;
R0 = V0.*sqrt_s0;
%% Problem parameters
A = U(:, (r_+1):r)*a2{i}*V(:, (r_+1):r)' + U(:, (r+1):(r+r_))*a3{i}*V(:, (r+1):(r+r_))';
f0 = @(X) 0.5*norm(sqrt(W{i}).*(X-A), 'fro')^2;
f1 = @(X) W{i}.*(X-A);
f2 = @(X, dX) W{i}.*dX;
g0 = @(L, R) f0(L*R');
g1 = @(L, R) f1(L*R');
%% Required number of iterations
% [~, ~, ~, ~, time_PGD, i_PGD] = PGDtime(r, s0(i, :), U0, V0, f0, f1, a, b, c, f_tol);
% [~, ~, ~, ~, time_P2GD, i_P2GD] = P2GDtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, f_tol);
% [~, ~, ~, ~, time_RFD, i_RFD] = ERFDtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, 0, f_tol);
% [~, ~, ~, ~, time_P2GDR, i_P2GDR] = P2GDRtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, Delta, f_tol);
% [~, ~, ~, ~, time_P2GDPGD, i_P2GDPGD] = P2GDPGDtime(r, s0(i, :), U0, V0, f0, f1, g0, g1, a, b, c, Delta, f_tol);
% [~, ~, ~, ~, time_RFDR, i_RFDR] = ERFDRtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, 0, Delta, f_tol);
% [~, ~, ~, time_HRTR, i_HRTR] = HRTRtime(L0, R0, f0, f1, f2, gamma, gamma_c, eta, f_tol);
% Instance 18
i_PGD = 233;
i_P2GD = 1238;
i_RFD = 264549;
i_P2GDR = 233;
i_P2GDPGD = 233;
i_RFDR = 233;
i_HRTR = 120;
% Instance 44
% i_PGD = 663;
% i_P2GD = 322850;
% i_RFD = 524827;
% i_P2GDR = 697;
% i_P2GDPGD = 695;
% i_RFDR = 733;
% i_HRTR = 356;
%% Iterates, function values, and running times
[s_PGD, U_PGD, V_PGD, f_PGD, time_PGD] = PGDiterinfo(r, s0(i, :), U0, V0, f0, f1, a, b, c, i_PGD);
[s_P2GD, U_P2GD, V_P2GD, f_P2GD, time_P2GD] = P2GDiterinfo(r, s0(i, :), U0, V0, g0, g1, a, b, c, i_P2GD);
[s_RFD, U_RFD, V_RFD, f_RFD, time_RFD] = ERFDiterinfo(r, s0(i, :), U0, V0, g0, g1, a, b, c, 0, i_RFD);
[s_P2GDR, U_P2GDR, V_P2GDR, f_P2GDR, time_P2GDR, R_P2GDR] = P2GDRiterinfo(r, s0(i, :), U0, V0, g0, g1, a, b, c, Delta, i_P2GDR);
[s_P2GDPGD, U_P2GDPGD, V_P2GDPGD, f_P2GDPGD, time_P2GDPGD, R_P2GDPGD] = P2GDPGDiterinfo(r, s0(i, :), U0, V0, f0, f1, g0, g1, a, b, c, Delta, i_P2GDPGD);
[s_RFDR, U_RFDR, V_RFDR, f_RFDR, time_RFDR, R_RFDR] = ERFDRiterinfo(r, s0(i, :), U0, V0, g0, g1, a, b, c, 0, Delta, i_RFDR);
[L_HRTR, R_HRTR, f_HRTR, NormGrad_HRTR, SmallestEig_HRTR, time_HRTR] = HRTRiterinfo(L0, R0, f0, f1, f2, gamma, gamma_c, eta, i_HRTR);
%% B-stationarity measure
B_PGD = zeros(i_PGD+1, 1);
for i = 1:i_PGD+1
    [~, ~, ~, ~, B_PGD(i)] = P2GDmap(r, length(s_PGD{i}), s_PGD{i}, U_PGD{i}, V_PGD{i}, g0, g1, a, b, c);
end
B_P2GD = zeros(i_P2GD+1, 1);
for i = 1:i_P2GD+1
    [~, ~, ~, ~, B_P2GD(i)] = P2GDmap(r, length(s_P2GD{i}), s_P2GD{i}, U_P2GD{i}, V_P2GD{i}, g0, g1, a, b, c);
end
B_RFD = zeros(i_RFD+1, 1);
for i = 1:i_RFD+1
    [~, ~, ~, ~, B_RFD(i)] = P2GDmap(r, length(s_RFD{i}), s_RFD{i}, U_RFD{i}, V_RFD{i}, g0, g1, a, b, c);
end
B_P2GDR = zeros(i_P2GDR+1, 1);
for i = 1:i_P2GDR+1
    [~, ~, ~, ~, B_P2GDR(i)] = P2GDmap(r, length(s_P2GDR{i}), s_P2GDR{i}, U_P2GDR{i}, V_P2GDR{i}, g0, g1, a, b, c);
end
B_P2GDPGD = zeros(i_P2GDPGD+1, 1);
for i = 1:i_P2GDPGD+1
    [~, ~, ~, ~, B_P2GDPGD(i)] = P2GDmap(r, length(s_P2GDPGD{i}), s_P2GDPGD{i}, U_P2GDPGD{i}, V_P2GDPGD{i}, g0, g1, a, b, c);
end
B_RFDR = zeros(i_RFDR+1, 1);
for i = 1:i_RFDR+1
    [~, ~, ~, ~, B_RFDR(i)] = P2GDmap(r, length(s_RFDR{i}), s_RFDR{i}, U_RFDR{i}, V_RFDR{i}, g0, g1, a, b, c);
end
B_HRTR = zeros(i_HRTR+1, 1);
for i = 1:i_HRTR+1
    [U_L, R_L] = qr(L_HRTR{i}, 'econ');
    [V_HRTR, s_HRTR, U_HRTR] = svd(R_HRTR{i}*R_L', 'econ');
    U_HRTR = U_L*U_HRTR;
    s_HRTR = diag(s_HRTR)';
    s_HRTR = s_HRTR(s_HRTR > 0);
    r_HRTR = length(s_HRTR);
    U_HRTR = U_HRTR(:, 1:r_HRTR);
    V_HRTR = V_HRTR(:, 1:r_HRTR);
    [~, ~, ~, ~, B_HRTR(i)] = P2GDmap(r, r_HRTR, s_HRTR, U_HRTR, V_HRTR, g0, g1, a, b, c);
end
%% P2GD and RFD iterates based on [OGA25, Proposition 8.1]
%j_max = 1160; % instance 18
j_max = i_ZeroSingularValue(i); % instance 44
X_diff_P2GD = zeros(j_max, 1);
X_diff_RFD = zeros(j_max, 1);
sigma_diff_P2GD = zeros(j_max, 1);
sigma_diff_RFD = zeros(j_max, 1);
sigma_r_diff_P2GD = zeros(j_max, 1);
sigma_r_diff_RFD = zeros(j_max, 1);
for j = 1:j_max
    X_diff_P2GD(j) = norm((U_P2GD{j}.*s_P2GD{j})*V_P2GD{j}'-U(:, 1:r)*S_apo{i, j}*V(:, 1:r)', 'fro');
    X_diff_RFD(j) = norm((U_RFD{j}.*s_RFD{j})*V_RFD{j}'-U(:, 1:r)*S_apo{i, j}*V(:, 1:r)', 'fro');
    sigma_diff_P2GD(j) = norm(s_P2GD{j}-s_apo{i, j}, 'fro');
    sigma_diff_RFD(j) = norm(s_RFD{j}-s_apo{i, j}, 'fro');
    sigma_r_diff_P2GD(j) = abs(s_P2GD{j}(r)-s_apo{i, j}(r));
    sigma_r_diff_RFD(j) = abs(s_RFD{j}(r)-s_apo{i, j}(r));
end
%% Plot objective function and B-stationarity measure as functions of time
figure
% Objective function as a function of time
subplot(2, 1, 1)
box
hold on
plot(log10(time_PGD), log10(f_PGD), 'k.-', 'MarkerSize', 8);
plot(log10(time_P2GD), log10(f_P2GD), 'm.-', 'MarkerSize', 8);
plot(log10(time_RFD), log10(f_RFD), 'c.-', 'MarkerSize', 8);
plot(log10(time_P2GDR), log10(f_P2GDR), 'r.-', 'MarkerSize', 8);
plot(log10(time_P2GDPGD), log10(f_P2GDPGD), 'g.-', 'MarkerSize', 8);
plot(log10(time_RFDR), log10(f_RFDR), 'b.-', 'MarkerSize', 8);
plot(log10(time_HRTR), log10(f_HRTR), '.-', 'MarkerSize', 8, 'Color', '#A2142F');
xlim([-3.5 1.5])
xticks(0.5*(-7:3))
ylim([-13 1])
yticks(-13:1)
xlabel('log_{10}(t_i)');
ylabel('log_{10}(f(X_i))');
legend('PGD', 'P^2GD', 'RFD', 'P^2GDR', 'P^2GD-PGD', 'RFDR', 'HRTR', 'location', 'southwest');
set(gca, 'FontSize', 11);
% B-stationarity measure as a function of time.
subplot(2, 1, 2)
box
hold on
plot(log10(time_PGD), log10(B_PGD), 'k.-', 'MarkerSize', 8);
plot(log10(time_P2GD), log10(B_P2GD), 'm.-', 'MarkerSize', 8);
plot(log10(time_RFD), log10(B_RFD), 'c.-', 'MarkerSize', 8);
plot(log10(time_P2GDR), log10(B_P2GDR), 'r.-', 'MarkerSize', 8);
plot(log10(time_P2GDPGD), log10(B_P2GDPGD), 'g.-', 'MarkerSize', 8);
plot(log10(time_RFDR), log10(B_RFDR), 'b.-', 'MarkerSize', 8);
plot(log10(time_HRTR), log10(B_HRTR), '.-', 'MarkerSize', 8, 'Color', '#A2142F');
xlim([-3.5 1.5])
xticks(0.5*(-7:3))
ylim([-17 1])
yticks(2*(-9:0)+1)
xlabel('log_{10}(t_i)');
ylabel('log_{10}(s(X_i; f, R_{\leq r}^{m\times n}))');
legend('PGD', 'P^2GD', 'RFD', 'P^2GDR', 'P^2GD-PGD', 'RFDR', 'HRTR', 'location', 'southwest');
set(gca, 'FontSize', 11);
%% Lower bound max{epsilon(1)^(-2), epsilon(2)^(-2)} on the right-hand side of [LKB23, (3.11)]
Bound_HRTR = floor(160*max(NormGrad_HRTR.^(-2), (-SmallestEig_HRTR).^(-3)));
figure
box
hold on
plot(0:i_HRTR-1, log10(1:i_HRTR), 'b.-', 'MarkerSize', 8);
plot(0:i_HRTR-1, log10(Bound_HRTR), 'g.-', 'MarkerSize', 8);
xlim([0 360])
xticks(60*(0:6))
ylim([0 15])
yticks(3*(0:5))
xlabel('i');
legend('log_{10}(i+1)', 'log_{10}(k_i)', 'location', 'northwest');
set(gca, 'FontSize', 11);
%% Plot distance between empirically and analytically computed P2GD or RFD iterates
figure
box
hold on
plot(log10(X_diff_P2GD), 'b.-')
plot(log10(X_diff_RFD), 'g.-')
xlabel('i');
ylabel('log_{10}(||X_i^{empirical}-X_i^{analytical}||)');
legend('P2GD', 'RFD', 'location', 'east');
set(gca, 'FontSize', 11);
figure
box
hold on
plot(log10(sigma_diff_P2GD), 'b.-')
plot(log10(sigma_diff_RFD), 'g.-')
xlabel('i');
ylabel('log_{10}(||\Sigma_i^{empirical}-\Sigma_i^{analytical}||)');
legend('P2GD', 'RFD', 'location', 'east');
set(gca, 'FontSize', 11);
figure
box
hold on
plot(log10(sigma_r_diff_P2GD), 'b.-')
plot(log10(sigma_r_diff_RFD), 'g.-')
xlabel('i');
ylabel('log_{10}(|\sigma_{r, i}^{empirical}-\sigma_{r, i}^{analytical}|)');
legend('P2GD', 'RFD', 'location', 'east');
set(gca, 'FontSize', 11);
