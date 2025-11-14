function [L, R, f, time, i] = HRTRtime(L, R, f0, f1, f2, gamma, gamma_c, eta, f_tol)
%% Description
% Author: Guillaume Olikier (2025-11-12)
% This function implements HRTR [LKB23, Algorithm 1] with the first lift
% phi of [LKB23, (1.1)], namely phi(L, R) := L*R', the hook from
% [LKB23, Example 3.11], and the Cauchy step.
% Input:
%   - L and R, hooked m-by-r and n-by-r real matrices, respectively;
%   - f0, a real-valued function of a matrix variable;
%   - f1, the gradient of f0;
%   - f2, a function that returns the directional derivative of f1 at its
%     first argument along its second argument;
%   - a positive real number gamma;
%   - gamma_c and eta in (0, 1);
%   - a positive real number f_tol.
% Output:
%   - the first (L, R) such that f0(L*R') <= f_tol;
%   - the value f of f0 at L*R';
%   - the running time required to generate (L, R);
%   - the number of iterations required to generate (L, R).
% This function computes only what is necessary to generate L and R.
g = @(L, R) f0(L*R');
g_1_1 = @(L, R) f1(L*R')*R;
g_1_2 = @(L, R) f1(L*R')'*L;
g_2_1 = @(L, R, dL, dR) f2(L*R', dL*R'+L*dR')*R + f1(L*R')*dR;
g_2_2 = @(L, R, dL, dR) f2(L*R', dL*R'+L*dR')'*L + f1(L*R')'*dL;
[m, r] = size(L);
[n, ~] = size(R);
dim1 = m*r;
dim2 = n*r;
dim = dim1+dim2;
tic
f = g(L, R);
i = 0;
while f > f_tol && toc < 1800
    G_1_1 = g_1_1(L, R);
    G_1_2 = g_1_2(L, R);
    SquaredNormGrad = norm(G_1_1, 'fro')^2+norm(G_1_2, 'fro')^2;
    NormGrad = sqrt(SquaredNormGrad);
    %% Smallest eigenvalue and associated eigenvector of the Hessian of the lifted cost function
    H = zeros(dim);
    for j = 1:m
        for k = 1:r
            dL_basis = zeros(m, r);
            dL_basis(j, k) = 1;
            H(:, (j-1)*r+k) = [reshape(g_2_1(L, R, dL_basis, zeros(n, r))', [dim1, 1]) ; reshape(g_2_2(L, R, dL_basis, zeros(n, r))', [dim2, 1])];
        end
    end
    for j = 1:n
        for k = 1:r
            dR_basis = zeros(n, r);
            dR_basis(j, k) = 1;
            H(:, dim1+(j-1)*r+k) = [reshape(g_2_1(L, R, zeros(m, r), dR_basis)', [dim1, 1]) ; reshape(g_2_2(L, R, zeros(m, r), dR_basis)', [dim2, 1])];
        end
    end
    [e, lambda] = eigs(H, 1, 'smallestreal', 'Tolerance', 1e-14, 'MaxIterations', 30);
    e = e/norm(e);
    E_1 = reshape(e(1:dim1), [r m])';
    E_2 = reshape(e(dim1+1:dim), [r n])';
    %% Trust-region step
    if trace(G_1_1'*E_1) + trace(G_1_2'*E_2) > 0
        E_1 = -E_1;
        E_2 = -E_2;
    end
    if lambda < 0 && SquaredNormGrad < -lambda^3
        delta = -gamma*lambda;
        U_1 = E_1;
        U_2 = E_2;
    else
        delta = gamma*NormGrad;
        U_1 = -G_1_1;
        U_2 = -G_1_2;
    end
    normU = sqrt(norm(U_1, 'fro')^2+norm(U_2, 'fro')^2);
    a = trace(g_2_1(L, R, U_1, U_2)'*U_1)+trace(g_2_2(L, R, U_1, U_2)'*U_2);
    b = trace(G_1_1'*U_1)+trace(G_1_2'*U_2);
    t = delta/normU;
    if a > 0
        t = min([t -b/a]);
    end
    L_new = L+t*U_1;
    R_new = R+t*U_2;
    decreaseModel = -b*t-0.5*t*t*a;
    if decreaseModel == 0
        rho = 1;
    else
        f_new = g(L_new, R_new);
        rho = (f-f_new)/decreaseModel;
    end
    while rho < eta && toc < 1800
        delta = gamma_c*delta;
        t = delta/normU;
        if a > 0
            t = min([t -b/a]);
        end
        L_new = L+t*U_1;
        R_new = R+t*U_2;
        decreaseModel = -b*t-0.5*t*t*a;
        if decreaseModel == 0
            rho = 1;
        else
            f_new = g(L_new, R_new);
            rho = (f-f_new)/decreaseModel;
        end
    end
    f = f_new;
    %% Hook
    [U_L, R_L] = qr(L_new, 'econ');
    [V, s, U] = svd(R_new*R_L', 'econ');
    s = sqrt(diag(s)');
    U = U_L*U;
    L = U.*s;
    R = V.*s;
    i = i+1;
end
time = toc;
end
