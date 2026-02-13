%% Plots for a matrix completion problem
% Author: Guillaume Olikier (2025-11-01)
% This script plots, for each of the following methods applied to a matrix
% completion problem, the objective function and a B-stationarity measure
% as functions of the running time until a function value smaller than or
% equal to f_tol is reached:
%   1. monotone PGD [OW25, Algorithm 4.2 with l = 0 or p = 1];
%   2. P2GD [SU15, Algorithm 3];
%   3. P2GDR [OGA25, Definition 6.1];
%   4. P2GD-PGD [OGA25, Definition 7.1];
%   5. RFD [SU15, Algorithm 4];
%   6. RFDR [OA23, Algorithm 3].
%load('TimeMatrixCompletion.mat')
i = 1;
%% Objective function and gradient
A = (U_A{i}.*s_A(i, :))*V_A{i}';
f0 = @(X) 0.5*norm(sqrt(W{i}).*(X-A), 'fro')^2;
f1 = @(X) W{i}.*(X-A);
g0 = @(L, R) f0(L*R');
g1 = @(L, R) f1(L*R');
%% Iterates, function values, and running time
s_iter = cell([9 1]);
U_iter = cell([9 1]);
V_iter = cell([9 1]);
obj_iter = cell([9 1]);
time_iter = cell([9 1]);
[s_iter{1}, U_iter{1}, V_iter{1}, obj_iter{1}, time_iter{1}] = PGDiterinfo(r, s0(i, :), U0{i}, V0{i}, f0, f1, a, b, c, iter(i, 1));
%[s_iter{1}, U_iter{1}, V_iter{1}, obj_iter{1}, time_iter{1}] = PGDiterinfo(r, s0(i, :), U0{i}, V0{i}, f0, f1, a, b, c, iter(i, 1));
[s_iter{2}, U_iter{2}, V_iter{2}, obj_iter{2}, time_iter{2}] = P2GDiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, iter(i, 2));
%[s_iter{2}, U_iter{2}, V_iter{2}, obj_iter{2}, time_iter{2}] = P2GDiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, iter(i, 2));
[s_iter{3}, U_iter{3}, V_iter{3}, obj_iter{3}, time_iter{3}, R_P2GDR] = P2GDRiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, Delta, iter(i, 3));
%[s_iter{3}, U_iter{3}, V_iter{3}, obj_iter{3}, time_iter{3}, R_P2GDR] = P2GDRiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, Delta, iter(i, 3));
[s_iter{4}, U_iter{4}, V_iter{4}, obj_iter{4}, time_iter{4}, R_P2GDR_] = P2GDRiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, Delta_, iter(i, 4));
%[s_iter{4}, U_iter{4}, V_iter{4}, obj_iter{4}, time_iter{4}, R_P2GDR_] = P2GDRiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, Delta_, iter(i, 4));
[s_iter{5}, U_iter{5}, V_iter{5}, obj_iter{5}, time_iter{5}, R_P2GDPGD] = P2GDPGDiterinfo(r, s0(i, :), U0{i}, V0{i}, f0, f1, g0, g1, a, b, c, Delta, iter(i, 5));
%[s_iter{5}, U_iter{5}, V_iter{5}, obj_iter{5}, time_iter{5}, R_P2GDPGD] = P2GDPGDiterinfo(r, s0(i, :), U0{i}, V0{i}, f0, f1, g0, g1, a, b, c, Delta, iter(i, 5));
[s_iter{6}, U_iter{6}, V_iter{6}, obj_iter{6}, time_iter{6}, R_P2GDPGD_] = P2GDPGDiterinfo(r, s0(i, :), U0{i}, V0{i}, f0, f1, g0, g1, a, b, c, Delta_, iter(i, 6));
%[s_iter{6}, U_iter{6}, V_iter{6}, obj_iter{6}, time_iter{6}, R_P2GDPGD_] = P2GDPGDiterinfo(r, s0(i, :), U0{i}, V0{i}, f0, f1, g0, g1, a, b, c, Delta_, iter(i, 6));
[s_iter{7}, U_iter{7}, V_iter{7}, obj_iter{7}, time_iter{7}] = ERFDiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, 0, iter(i, 7));
%[s_iter{7}, U_iter{7}, V_iter{7}, obj_iter{7}, time_iter{7}] = ERFDiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, 0, iter(i, 7));
[s_iter{8}, U_iter{8}, V_iter{8}, obj_iter{8}, time_iter{8}, R_RFDR] = ERFDRiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, 0, Delta, iter(i, 8));
%[s_iter{8}, U_iter{8}, V_iter{8}, obj_iter{8}, time_iter{8}, R_RFDR] = ERFDRiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, 0, Delta, iter(i, 8));
[s_iter{9}, U_iter{9}, V_iter{9}, obj_iter{9}, time_iter{9}, R_RFDR_] = ERFDRiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, 0, Delta_, iter(i, 9));
%[s_iter{9}, U_iter{9}, V_iter{9}, obj_iter{9}, time_iter{9}, R_RFDR_] = ERFDRiterinfo(r, s0(i, :), U0{i}, V0{i}, g0, g1, a, b, c, 0, Delta_, iter(i, 9));
%% B-stationarity measure
B_iter = cell([9 1]);
for j = 1:9
    B_iter{j} = zeros(iter(i, j)+1, 1);
    for k = 1:iter(i, j)+1
        [~, ~, ~, ~, B_iter{j}(k)] = P2GDmap(r, length(s_iter{j}{k}), s_iter{j}{k}, U_iter{j}{k}, V_iter{j}{k}, g0, g1, a, b, c);
    end
end
clear j k f0 f1 g0 g1
%% Plot objective function and B-stationarity measure as functions of time
col = [0 0 0; 0 1 1; 0 0.5 1; 0 0 1; 0 1 0; 0 0.7 0.5; 1 0 1; 1 0 0; hex2rgb('#A2142F')];
figure
% Objective function as a function of time
subplot(1, 2, 1)
box
hold on
for j = 1:9
    plot(time_iter{j}, log10(obj_iter{j}), '.-', 'MarkerSize', 26, 'Color', col(j, :));
end
xlim([0 ceil(time_iter{1}(end))])
xticks(0:ceil(time_iter{1}(end)))
ylim([-16 0])
yticks(-16:0)
xlabel('t_i (seconds)');
ylabel('log_{10}(f(X_i))');
legend('PGD', 'P^2GD', 'P^2GDR 0.01', 'P^2GDR 0.1', 'P^2GD-PGD 0.01', 'P^2GD-PGD 0.1', 'RFD', 'RFDR 0.01', 'RFDR 0.1', 'location', 'northeast');
set(gca, 'FontSize', 26);
% B-stationarity measure as a function of time
subplot(1, 2, 2)
box
hold on
for j = 1:9
    plot(time_iter{j}, log10(B_iter{j}), '.-', 'MarkerSize', 26, 'Color', col(j, :));
end
xlim([0 ceil(time_iter{1}(end))])
xticks(0:ceil(time_iter{1}(end)))
ylim([-9 0])
yticks(-9:0)
xlabel('t_i (seconds)');
ylabel('log_{10}(s(X_i; f, R_{\leq r}^{m\times n}))');
legend('PGD', 'P^2GD', 'P^2GDR 0.01', 'P^2GDR 0.1', 'P^2GD-PGD 0.01', 'P^2GD-PGD 0.1', 'RFD', 'RFDR 0.01', 'RFDR 0.1', 'location', 'northeast');
set(gca, 'FontSize', 26);
clear col