%FAST_SPACE_ORDER Estimate spatial convergence from nested single-grid
% solution differences.
clear;
format short e;
L = 1; T = 1; alpha0 = 0.5; r = 4; nu = 0.01; N = 1024;
epsilon_alpha = 1e-10; epsilon_kappa = 1e-8;
M_values = [16;32;64;128]; % 4;8;16;32;64;128;256;512
%M_values = [8;16;32;64];
all_M = [M_values;2 * M_values(end)];
final_solution = cell(numel(all_M),1);
cpu_all = zeros(numel(all_M),1);
for level = 1:numel(all_M)
    M = all_M(level);
    fprintf('\n=== Fast scheme: M=%d, N=%d ===\n',M,N);
    [U,~,~,cpu_all(level)]  =  Fast_Main(...
        L,T,M,N,alpha0,r,nu,epsilon_alpha,epsilon_kappa,false);
    final_solution{level} = U(:,end);
end

level_count = numel(M_values);
error_inf = zeros(level_count,1);
error_L2 = zeros(level_count,1);
rate_inf = nan(level_count,1);
rate_L2 = nan(level_count,1);
for level = 1:level_count
    M = M_values(level);
    h = L/M;
    difference = final_solution{level}-final_solution{level+1}(2:2:end);
    error_inf(level) = norm(difference,inf);
    error_L2(level) = sqrt(h * sum(difference.^2));
end
for level = 2:level_count
    rate_inf(level) = log2(error_inf(level-1)/error_inf(level));
    rate_L2(level) = log2(error_L2(level-1)/error_L2(level));
end
cpu_time = cpu_all(1:level_count);
fprintf('\n=== Spatial convergence results: fast scheme, exact solution known ===\n');
% results = table(M_values,error_inf,rate_inf,error_L2,rate_L2,cpu_time);
results = table(M_values,error_L2,rate_L2,cpu_time);
disp(results);
fprintf('Parameters: alpha0=%.2f, nu=%.2f, r=%.2f, N=%d\n',alpha0,nu,r,N);
fprintf('==================================================\n');
