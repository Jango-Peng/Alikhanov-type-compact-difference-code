clear;
format short e;

% Parameter settings
L = 1;
T = 1;
alpha0 = 0.5; 
r = 2/alpha0; % Graded-mesh parameter
nu = 0.01; % Viscosity coefficient

M = 1024;
epsilon_alpha = 1e-10;
epsilon_kappa = 1e-8;
N0 = 8;
number_of_levels = 4;

% Preallocate errors, convergence rates, CPU times, and average Newton counts.
N_values = zeros(number_of_levels,1);
error_L2 = zeros(number_of_levels,1);
rate_L2 = nan(number_of_levels,1);
cpu_time = zeros(number_of_levels,1);
num_iter = zeros(number_of_levels,1);

N = N0;
fprintf('\n=== Fast scheme: temporal convergence order ===\n');
for level = 1:number_of_levels+1
    fprintf('\nTest %d/%d: M=%d, N=%d\n',...
        level,number_of_levels+1,M,N);

    [U,h,~,time,newton_iterations] = Fast_Main(...
        L,T,M,N,alpha0,r,nu,epsilon_alpha,epsilon_kappa,0);

    if level > 1
        % coarse_solution and U correspond to adjacent N and 2N time grids.
        error_vector = coarse_solution(:,end)-U(:,end);
        error_L2(level-1) = sqrt(h*sum(error_vector.^2));
    end
    coarse_solution = U;

    if level <= number_of_levels
        num_iter(level) = mean(newton_iterations);
        N_values(level) = N;
        cpu_time(level) = time;
    end
    N = 2*N;
end

for level = 2:number_of_levels
    rate_L2(level) = log2(error_L2(level-1)/error_L2(level));
end

fprintf('\n=== Temporal convergence results: fast scheme, exact solution known ===\n');
results = table(N_values,error_L2,rate_L2,cpu_time,num_iter);
disp(results);
fprintf('Parameters: alpha0=%.2f, nu=%.2f, r=%.2f, M=%d\n',...
    alpha0,nu,r,M);
fprintf('==================================================\n');
