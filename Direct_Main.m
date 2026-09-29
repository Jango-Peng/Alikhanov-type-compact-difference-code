function [U,h,t,cpu_time,newton_iterations,newton_residuals] = ...
    Direct_Main(L,T,M,N,alpha0,r,nu,show_progress)
%DIRECT_MAIN Direct compact-difference scheme for the full nonlinear
% fractional Burgers problem.
%
% U(:,n+1) stores the numerical solution at t_n on all interior spatial
% nodes.  The implementation follows the order
% grid generation -> current-level weights -> history accumulation ->
% standard Newton solve.  No structures are used, and all time-level
% weights are generated only when they are needed.

% Use progress output by default when it is not supplied.
if nargin < 8 
    show_progress = true;
end

if alpha0 <= 0 || alpha0 >= 1
    error('Direct_Main:OrderRange','alpha0 must satisfy 0<alpha0<1.');
end
if alpha0+0.2*T^(1+alpha0) >= 1
    error('Direct_Main:OrderRange',...
        'The current parameters give alpha(T)>=1; reduce T or modify alpha(t).');
end

start_time = tic;

%% 1. Time and spatial grids
theta = alpha0/2;
sigma = 1-theta;

t = zeros(1,N+1);
for n = 0:N
    t(n+1) = T*(n/N)^r;
end

tau = zeros(1,N);
for n = 1:N
    tau(n) = t(n+1)-t(n);
end

t_offset = zeros(1,N);
for n = 1:N
    t_offset(n) = theta*t(n)+sigma*t(n+1);
end

h = L/M;
x = (1:M-1)'*h;
in_count = M-1;

%% 2. Construct the compact matrix H, second-order matrix D,
% and first-order centered-difference matrix C.
e = ones(in_count,1);
H = spdiags([e,10*e,e],[-1,0,1],in_count,in_count)/12;
D = spdiags([e,-2*e,e],[-1,0,1],in_count,in_count)/h^2;
C = spdiags([-e,e],[-1,1],in_count,in_count)/(2*h);

%% 3. Initial data and Newton parameters
U = zeros(in_count,N+1);
U(:,1) = Exact_Solution(x,0,L,alpha0);

newton_tolerance = 1e-10;
max_newton_iterations = 30;
newton_iterations = zeros(N,1);
newton_residuals = zeros(N,1);

if show_progress
    fprintf('\n=== Direct scheme: compact nonlinear discretization ===\n');
    fprintf('M=%d, N=%d\n',M,N);
end

%% 4. Advance from time level 1 to time level N
progress_step = max(1,floor(N/10));
for n = 1:N
    % A_k_n(k) multiplies U^k-U^(k-1) and is stored using the actual
    % time-level index k.  Likewise, chi_j_n(j) multiplies U^(j-theta).
    [A_k_n,chi_j_n] = Direct_Time_Weights(...
        n,t,t_offset,alpha0,theta,sigma);

    F_n_theta = Source_Term(...
        x,t_offset(n),L,alpha0,nu);

    caputo_history = zeros(in_count,1);
    perturbation_history = zeros(in_count,1);

    for k = 1:n-1
        delta_U_k = U(:,k+1)-U(:,k);
        caputo_history = caputo_history+A_k_n(k)*delta_U_k;

        U_k_theta = theta*U(:,k)+sigma*U(:,k+1);
        perturbation_history = perturbation_history+chi_j_n(k)*U_k_theta;
    end

    % The current unknown is w=U^(n-theta).  A_k_n(n) and chi_j_n(n)
    % are the coefficients associated with the current level.
    current_A = A_k_n(n);
    current_chi = chi_j_n(n);
    linear_coefficient = current_A/sigma+current_chi;

    % The one-level nonlinear equation is
    % linear_coefficient*w+B_h(w,z)-nu*z+known_vector=0.
    known_vector = caputo_history+perturbation_history ...
        -(current_A/sigma)*U(:,n)-F_n_theta;

    % Use the previous solution or a linear extrapolation as the Newton
    % initial guess.  This does not change the final discrete equation.
    if n == 1
        U_predictor = U(:,n);
    else
        U_predictor = U(:,n)+(U(:,n)-U(:,n-1));
    end
    w_initial = theta*U(:,n)+sigma*U_predictor;

    [w,~,iteration_count,residual_norm] = Direct_Newton(...
        linear_coefficient,known_vector,H,D,C,h,nu,w_initial,...
        newton_tolerance,max_newton_iterations,n);

    U(:,n+1) = (w-theta*U(:,n))/sigma;
    newton_iterations(n) = iteration_count;
    newton_residuals(n) = residual_norm;

    if show_progress && (mod(n,progress_step)==0 || n==N)
        fprintf('Time advancement: %3d%%, level %d, Newton iterations %d\n',...
            round(100*n/N),n,iteration_count);
    end
end

cpu_time = toc(start_time);

if show_progress
    fprintf('Completed in %.4f seconds of CPU time.\n',cpu_time);
    fprintf('Average Newton iterations %.2f; maximum %d.\n',...
        mean(newton_iterations),max(newton_iterations));
end
end
