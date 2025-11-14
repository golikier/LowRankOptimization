%% Running time on a weighted low-rank approximation (WLRA) problem
% Author: Guillaume Olikier (2025-11-01)
% This script computes, for N randomly generated instances of a WLRA
% problem, the running time and number of iterations needed by each of the
% following methods to bring the objective function below f_tol:
%   1. monotone PGD [OW25, Algorithm 4.2 with l = 0 or p = 1];
%   2. P2GD [SU15, Algorithm 3];
%   3. P2GDR [OGA25, Definition 6.1];
%   4. P2GD-PGD [OGA25, Definition 7.1];
%   5. RFD [SU15, Algorithm 4];
%   6. RFDR [OA23, Algorithm 3].
% All methods are stopped after one minute. The final iterate, function
% value, B-stationarity measure, smallest singular value, and measure of
% the lack of orthonormality are also computed.
N = 100;
f_tol = 1e-15;
%% Problem parameters
m = 600;
n = 400;
r = 15;
% Weight matrix
W = cell([N 1]);
for i = 1:N
    W{i} = rand(m, n);
end
% Matrix from [OGA25, section 8.2]
r_ = 10;
U = [eye(r+r_); zeros(m-r-r_, r+r_)];%[U, ~] = qr(randn(m, r+r_), 'econ');
V = [eye(r+r_); zeros(n-r-r_, r+r_)];%[V, ~] = qr(randn(n, r+r_), 'econ');
a2 = cell([N 1]);
a3 = cell([N 1]);
for i = 1:N
    a2{i} = randn(r-r_);
    a3{i} = randn(r_);
end
% Initial iterate from [OGA25, section 8.2]
U0 = U(:, 1:r);
V0 = V(:, 1:r);
s0 = rand(N, r);
s0 = sort(s0, 2, 'descend');
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
% Minimum between i_apo and the smallest index where the rth singular
% value of the apocalyptic sequence becomes smaller than eps(0)
i_apo = 1e4;
i_ZeroSingularValue = zeros(N, 1);
% Apocalyptic sequence
S_apo = cell([N i_apo]);
s_apo = cell([N i_apo]);
U_apo = cell([N i_apo]);
V_apo = cell([N i_apo]);
% Factorization of the last iterate
s = cell([N 9]);
Ur = cell([N 9]);
Vr = cell([N 9]);
%% Running time
for i = 1:N
    % Objective function and gradient
    A = U(:, (r_+1):r)*a2{i}*V(:, (r_+1):r)' + U(:, (r+1):(r+r_))*a3{i}*V(:, (r+1):(r+r_))';
    f0 = @(X) 0.5*norm(sqrt(W{i}).*(X-A), 'fro')^2;
    f1 = @(X) W{i}.*(X-A);
    g0 = @(L, R) f0(L*R');
    g1 = @(L, R) f1(L*R');
    % P2GD and RFD iterates based on [OGA25, Proposition 8.1]
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
    % PGD
    [s{i, 1}, Ur{i, 1}, Vr{i, 1}, obj(i, 1), time(i, 1), iter(i, 1)] = PGDtime(r, s0(i, :), U0, V0, f0, f1, a, b, c, f_tol);
    % P2GD
    [s{i, 2}, Ur{i, 2}, Vr{i, 2}, obj(i, 2), time(i, 2), iter(i, 2)] = P2GDtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, f_tol);
    % P2GDR with Delta
    [s{i, 3}, Ur{i, 3}, Vr{i, 3}, obj(i, 3), time(i, 3), iter(i, 3)] = P2GDRtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, Delta, f_tol);
    % P2GDR with Delta_
    [s{i, 4}, Ur{i, 4}, Vr{i, 4}, obj(i, 4), time(i, 4), iter(i, 4)] = P2GDRtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, Delta_, f_tol);
    % P2GD-PGD with Delta
    [s{i, 5}, Ur{i, 5}, Vr{i, 5}, obj(i, 5), time(i, 5), iter(i, 5)] = P2GDPGDtime(r, s0(i, :), U0, V0, f0, f1, g0, g1, a, b, c, Delta, f_tol);
    % P2GD-PGD with Delta_
    [s{i, 6}, Ur{i, 6}, Vr{i, 6}, obj(i, 6), time(i, 6), iter(i, 6)] = P2GDPGDtime(r, s0(i, :), U0, V0, f0, f1, g0, g1, a, b, c, Delta_, f_tol);
    % RFD
    [s{i, 7}, Ur{i, 7}, Vr{i, 7}, obj(i, 7), time(i, 7), iter(i, 7)] = ERFDtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, 0, f_tol);
    % RFDR with Delta
    [s{i, 8}, Ur{i, 8}, Vr{i, 8}, obj(i, 8), time(i, 8), iter(i, 8)] = ERFDRtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, 0, Delta, f_tol);
    % RFDR with Delta_
    [s{i, 9}, Ur{i, 9}, Vr{i, 9}, obj(i, 9), time(i, 9), iter(i, 9)] = ERFDRtime(r, s0(i, :), U0, V0, g0, g1, a, b, c, 0, Delta_, f_tol);
end
clear i_apo U_hat V_hat
%% Smallest singular value and lack of orthogonality
for i = 1:N
    A = U(:, (r_+1):r)*a2{i}*V(:, (r_+1):r)' + U(:, (r+1):(r+r_))*a3{i}*V(:, (r+1):(r+r_))';
    f0 = @(X) 0.5*norm(sqrt(W{i}).*(X-A), 'fro')^2;
    f1 = @(X) W{i}.*(X-A);
    g0 = @(L, R) f0(L*R');
    g1 = @(L, R) f1(L*R');
    for j = 1:9
        [~, ~, ~, ~, B(i, j)] = P2GDmap(r, length(s{i, j}), s{i, j}, Ur{i, j}, Vr{i, j}, g0, g1, a, b, c);
        sigma_r(i, j) = s{i, j}(r);
        orthU(i, j) = norm(Ur{i, j}'*Ur{i, j}-eye(r), 'fro');
        orthV(i, j) = norm(Vr{i, j}'*Vr{i, j}-eye(r), 'fro');
    end
end
clear A f0 f1 g0 g1
%% Generate tables
TimeTable = [min(time)' median(time)' max(time)'];
IterTable = [min(iter)' median(iter)' max(iter)'];
UnsolvedInstances = cell([9 1]);
ObjTable = zeros(9, 2);
BTable = zeros(9, 2);
sigma_rTable = zeros(9, 2);
for i = 1:9
    UnsolvedInstances{i} = find(obj(:, i) > f_tol);
    ObjTable(i, :) = [min(obj(UnsolvedInstances{i}, i)) max(obj(UnsolvedInstances{i}, i))];
    BTable(i, :) = [min(B(UnsolvedInstances{i}, i)) max(B(UnsolvedInstances{i}, i))];
    sigma_rTable(i, :) = [min(sigma_r(UnsolvedInstances{i}, i)) max(sigma_r(UnsolvedInstances{i}, i))];
end
% [~, fastestN] = min(time, [], 2);
% fastest = zeros(9, 1);
% for i = 1:9
%     fastest(i) = sum(fastestN == i);
% end
%% Generate performance profile
% Linear time
time_max = 60;
time_sample_length = 100;
time_sample = linspace(0, time_max, time_sample_length);
perf = zeros(9, time_sample_length);
for i = 1:9
    for j = 1:time_sample_length
        perf(i, j) = sum(time(:, i) <= time_sample(j));
    end
end
col = [0 0 0; 0 1 1; 0 0.5 1; 0 0 1; 0 1 0; 0 0.7 0.5; 1 0 1; 1 0 0; hex2rgb('#A2142F')];
figure
box
hold on
for i = 1:9
    plot(time_sample, perf(i, :), '.-', 'MarkerSize', 8, 'Color', col(i, :));
end
xlim([0 time_max])
xticks(5*(0:time_max/5))
ylim([0 N])
yticks(10*(0:N/10))
xlabel('t (seconds)');
ylabel('number of instances solved within t seconds');
legend('PGD', 'P^2GD', 'P^2GDR 0.01', 'P^2GDR 0.1', 'P^2GD-PGD 0.01', 'P^2GD-PGD 0.1', 'RFD', 'RFDR 0.01', 'RFDR 0.1', 'location', 'southeast');
set(gca, 'FontSize', 10);
clear i j col
% Logarithmic time
% time_min = 0.3;
% time_max = 60;
% time_sample_length = 100;
% log_time_min = log10(time_min);
% log_time_max = log10(time_max);
% log_time_sample = linspace(log_time_min, log_time_max, time_sample_length);
% time_sample = 10.^log_time_sample;
% perf = zeros(9, time_sample_length);
% for i = 1:9
%     for j = 1:time_sample_length
%         perf(i, j) = sum(time(:, i) <= time_sample(j));
%     end
% end
% col = [0 0 0; 0 1 1; 0 0.5 1; 0 0 1; 0 1 0; 0 0.7 0.5; 1 0 1; 1 0 0; hex2rgb('#A2142F')];
% figure
% box
% hold on
% for i = 1:9
%     plot(log_time_sample, perf(i, :), '.-', 'MarkerSize', 8, 'Color', col(i, :));
% end
% xlim([log_time_min log_time_max])
% xticks(log10([0.3 0.4 0.5 1:5 10 15 20:10:60]))
% xticklabels({'0.3', '0.4', '0.5', '1', '2', '3', '4', '5', '10', '15', '20', '30', '40', '50', '60'})
% ylim([0 N])
% yticks(10*(0:N/10))
% xlabel('t (seconds)');
% ylabel('number of instances solved within t seconds');
% legend('PGD', 'P^2GD', 'P^2GDR 0.01', 'P^2GDR 0.1', 'P^2GD-PGD 0.01', 'P^2GD-PGD 0.1', 'RFD', 'RFDR 0.01', 'RFDR 0.1', 'location', 'northwest');
% set(gca, 'FontSize', 10);
% clear i j col log_time_min log_time_max log_time_sample
%% Minimum singular value generated by RFD
s_iter_RFD = cell([N 1]);
s_r_iter_min_RFD = ones(N, 1);
for i = 1:N
    A = U(:, (r_+1):r)*a2{i}*V(:, (r_+1):r)' + U(:, (r+1):(r+r_))*a3{i}*V(:, (r+1):(r+r_))';
    f0 = @(X) 0.5*norm(sqrt(W{i}).*(X-A), 'fro')^2;
    f1 = @(X) W{i}.*(X-A);
    g0 = @(L, R) f0(L*R');
    g1 = @(L, R) f1(L*R');
    [s_iter_RFD{i}, ~, ~, ~, ~] = ERFDiterinfo(r, s0(i, :), U0, V0, g0, g1, a, b, c, 0, iter(i, 7));
    for j = 1:iter(i, 7)+1
        s_r_iter_min_RFD(i) = min(s_r_iter_min_RFD(i), s_iter_RFD{i}{j}(r));
    end
end
clear i j A f0 f1 g0 g1
