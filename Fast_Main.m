function [U,h,t,cpu_time,newton_iterations,newton_residuals,...
    Q_alpha,Q_kappa,caputo_soe_error,kappa_soe_error,...
    fast_alikhanov_bound,fast_alikhanov_ok] = Fast_Main(...
    L,T,M,N,alpha0,r,nu,epsilon_alpha,epsilon_kappa,show_progress,...
    alpha_function,dalpha_function)
%FAST_MAIN Fast scheme based on positive Caputo SOE and signed-SOE
% compression of the perturbation history.

% Optional alpha_function and dalpha_function inputs are accepted after
% show_progress for the general variable-order-kernel test.  Existing
% calls with the original ten inputs retain the example's default profile.
if nargin < 10 || isempty(show_progress)
    show_progress=true;
end
if nargin < 11 || isempty(alpha_function)
    alpha_function=@(q) alpha0+0.2*q.^(1+alpha0);
end
if nargin < 12 || isempty(dalpha_function)
    dalpha_function=@(q) 0.2*(1+alpha0)*q.^alpha0;
end
if ~isa(alpha_function,'function_handle') || ...
        ~isa(dalpha_function,'function_handle')
    error('Fast_Main:FunctionHandle',...
        'alpha_function and dalpha_function must be function handles.');
end

start_time = tic;

%% 1. Generate the time grid, the two SOE representations, and recurrence coefficients
[theta,sigma,t,tau,rho,t_offset,delta_j,source_scalar,...
    g_tilde_delta,s_alpha,omega_alpha,a_alpha,b_alpha,tilde_a_0,...
    s_kappa,omega_kappa,Q_alpha,Q_kappa,caputo_soe_error,...
    kappa_soe_error,fast_alikhanov_bound,fast_alikhanov_ok]  =  ...
    Fast_Prepare_Time(T,N,alpha0,r,epsilon_alpha,...
    epsilon_kappa,show_progress,alpha_function,dalpha_function);

%% 2. Construct one spatial grid and the compact-difference matrices
h = L/M;
x = (1:M-1)'*h;
interior_count = M-1;
e = ones(interior_count,1);
H = spdiags([e,10*e,e],[-1,0,1],interior_count,interior_count)/12;
D = spdiags([e,-2*e,e],[-1,0,1],interior_count,interior_count)/h^2;
C = spdiags([-e,e],[-1,1],interior_count,interior_count)/(2*h);

wave_number  =  pi/L;
sin_x = sin(wave_number*x);
cos_x = cos(wave_number*x);

%% 3. Initial data, SOE history states, and Newton parameters
U = zeros(interior_count,N+1);
U(:,1) = sin_x;
H_alpha_state = zeros(interior_count,Q_alpha);
Y_kappa_state = zeros(interior_count,Q_kappa);

newton_tolerance = 1e-10;
max_newton_iterations = 30;
newton_iterations = zeros(N,1);
newton_residuals = zeros(N,1);

%% 4. Advance from time level 1 to time level N
progress_step = max(1,floor(N/10));
for n = 1:N
    time_amplitude = 1+t_offset(n)^alpha0;
    F_current = source_scalar(n)*sin_x...
        +wave_number*time_amplitude^2*(sin_x.*cos_x)...
        +nu*wave_number^2*time_amplitude*sin_x;

    [linear_coefficient,known_vector,H_alpha_known,...
        H_alpha_current_shape,Y_kappa_decay,U_hat_previous]  =  ...
        Fast_History_Begin(n,U,H_alpha_state,Y_kappa_state,...
        F_current,theta,sigma,tau,rho,delta_j,g_tilde_delta,...
        s_alpha,omega_alpha,a_alpha,b_alpha,tilde_a_0,...
        s_kappa,omega_kappa);

    if n==1
        U_predictor = U(:,n);
    else
        U_predictor = 2*U(:,n)-U(:,n-1);
    end
    w_initial = theta*U(:,n)+sigma*U_predictor;
    [w,~,iteration_count,residual_norm]  =  Fast_Newton(...
        linear_coefficient,known_vector,H,D,C,h,nu,w_initial,...
        newton_tolerance,max_newton_iterations,n);
    U_new = (w-theta*U(:,n))/sigma;

    [H_alpha_state,Y_kappa_state] = Fast_History_Commit(...
        n,U(:,n),U_new,w,H_alpha_state,Y_kappa_state,...
        H_alpha_known,H_alpha_current_shape,Y_kappa_decay,...
        U_hat_previous,delta_j,s_kappa);
    U(:,n+1) = U_new;
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
    fprintf('=========================================\n')
end
end
