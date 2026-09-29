clear;
format short e;

% Parameter settings
L = 1;
T = 1;
alpha0 = 0.5;
r = 2/alpha0;
nu = 0.01;

N = 512;
M_star= 16;

all_levels = 5;


M_values = zeros(all_levels,1);
error_L2 = zeros(all_levels,1);
rate_L2 = nan(all_levels,1);
cpu_time = zeros(all_levels,1);

% Estimate the spatial error using differences between nested-grid
% single-grid solutions.
M=M_star;
fprintf('\n=== Direct scheme: spatial convergence order ===\n');
for level = 1:all_levels
    fprintf('\nTest %d/%d: M=%d, N=%d\n',level,all_levels,M,N);

    [U,~,~,cpu_time(level)] = Direct_Main(...
        L,T,M,N,alpha0,r,nu,true);
    if level > 1
        h = L/M;
        difference = coarse_solution(:,end)-U(2:2:M-2,end);
        error_L2(level-1) = sqrt(h*sum(difference.^2));
    end
    coarse_solution =U;
    M_values(level) = M;
    M = 2*M;
end

for level = 2:all_levels
    rate_L2(level) = log2(error_L2(level-1)/error_L2(level));
end

% Display the results.  This format is easier to customize than a table.
fprintf('\n=== Spatial convergence results: direct scheme, exact solution known ===\n');

fprintf('%-8s %-12s %-8s %-10s\n', 'M_value', 'Error', 'Rate', 'Time(s)');
fprintf('%-8s %-12s %-8s %-10s\n', '------', '-----', '----', '-------');
for i = 1:all_levels-1
    fprintf('%-8d %-12.4e %-8.4f %-10.2f\n', ...
            M_values(i), error_L2(i), rate_L2(i), cpu_time(i));
end

fprintf('Parameters: alpha0=%.2f, nu=%.2f, r=%.2f, N=%d\n',alpha0,nu,r,N);
fprintf('==================================================\n');
