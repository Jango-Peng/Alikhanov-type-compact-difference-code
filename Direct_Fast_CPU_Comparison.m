% DIRECT_FAST_CPU_COMPARISON
% Compare CPU times of the fast-SOE and direct schemes for Example 1.
% Before running, set the current MATLAB folder to this directory or add
% this directory to the MATLAB path.

clear;
format short e;

%% Parameter settings
L = 1;
T = 1;
alpha0 = 0.5;
r = 2/alpha0;          % Graded-mesh parameter
nu = 0.01;             % Viscosity coefficient
M = 128;               % Number of spatial subintervals

epsilon_alpha = 1e-10; % Caputo SOE tolerance
epsilon_kappa = 1e-8;  % signed-SOE tolerance

N = 256;               % Initial number of time levels
S = 6;                 % Number of tested N values

% Preallocate CPU-time and time-level arrays.
N_values = zeros(S,1);
fast_time = zeros(S,1);
direct_time = zeros(S,1);

%% Run both schemes for each parameter set
for i = 1:S
    fprintf('---- Group %d: M = %d, N = %d ----\n', i, M, N);

    % Fast scheme: the measured time includes SOE construction and
    % fast-history recurrence.
    [~,~,~,fast_time(i)] = Fast_Main( ...
        L,T,M,N,alpha0,r,nu,epsilon_alpha,epsilon_kappa,false);

    % Direct scheme: accumulate the history explicitly at every level.
    [~,~,~,direct_time(i)] = Direct_Main( ...
        L,T,M,N,alpha0,r,nu,false);

    N_values(i) = N;
    fprintf('    fast   CPU time = %.6e s\n', fast_time(i));
    fprintf('    direct CPU time = %.6e s\n', direct_time(i));

    % Increase the number of time levels after each group.
    N = N + 256;
end

%% Plot the CPU-time comparison
figure;
plot(N_values,fast_time,'r-o','LineWidth',1.2);
hold on;
plot(N_values,direct_time,'b-.s','LineWidth',1.2);
grid on;
legend('Fast scheme','Direct scheme','Location','northwest');
xlabel('N');
ylabel('CPU time (s)');
% title(sprintf('CPU time comparison, M = %d',M));
