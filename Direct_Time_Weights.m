function [A_k_n,chi_j_n] = Direct_Time_Weights(...
    n,t,t_offset,alpha0,theta,sigma)
%DIRECT_TIME_WEIGHTS Compute A_k^(n) and chi_j^(n) directly at level n.
%
% To avoid reversing the convolution indices used in the manuscript, the
% arrays use the actual time-level indices:
% A_k_n(k)   * (U^k-U^(k-1)), k=1,...,n;
% chi_j_n(j) * U^(j-theta),   j=1,...,n.

tau = diff(t);
t_n_theta = t_offset(n);

%% 1. Compute the basic L2-1_sigma weights a_k^(n)
a_k_n = zeros(n,1);
for k = 1:n
    left_endpoint = t(k);
    if k < n
        right_endpoint = t(k+1);
    else
        right_endpoint = t_n_theta;
    end

    left_lag = t_n_theta-left_endpoint;
    right_lag = t_n_theta-right_endpoint;

    % The stable form based on expm1 and log1p avoids cancellation.
    a_k_n(k) = stable_power_difference(left_lag,right_lag,1-alpha0)... 
            /(tau(k)*gamma(2-alpha0)); % Stable difference of nearby powers.
end

%% 2. Compute the quadratic-interpolation correction weights b_k^(n)
b_k_n = zeros(max(n-1,0),1);
for k = 1:n-1
    midpoint = (t(k)+t(k+1))/2;

  % Direct adaptive quadrature follows the defining integral but is slow.
    integrand = @(s) (s-midpoint).*(t_n_theta-s).^(-alpha0) ...
        /gamma(1-alpha0);
    integral_value = integral(integrand,t(k),t(k+1),...
        'AbsTol',1e-13,'RelTol',1e-11);
    b_k_n(k) = 2*integral_value/(tau(k)*(tau(k)+tau(k+1)));
end

%% 3. Combine a_k^(n) and b_k^(n) to obtain A_k^(n)
A_k_n = zeros(n,1);
if n == 1
    A_k_n(1) = a_k_n(1);
else
    rho = zeros(1,n-1);
    for k = 1:n-1
        rho(k) = tau(k)/tau(k+1);
    end

    A_k_n(1) = a_k_n(1)-b_k_n(1);
    for k = 2:n-1
        A_k_n(k) = a_k_n(k)+rho(k-1)*b_k_n(k-1)-b_k_n(k);
    end
    A_k_n(n) = a_k_n(n)+rho(n-1)*b_k_n(n-1);
end

%% 4. Compute omega_j^(n) from endpoint differences of g_tilde
eta = zeros(1,n+1);
eta(1) = 0;
for j = 1:n
    eta(j+1) = t_offset(j);
end

omega_j_n = zeros(n,1);
for j = 1:n
    left_value = G_Tilde(t_n_theta-eta(j),alpha0);
    right_value = G_Tilde(t_n_theta-eta(j+1),alpha0);
    omega_j_n(j) = left_value-right_value;
end

%% 5. Combine omega_j^(n) to obtain chi_j^(n)
chi_j_n = zeros(n,1);
if n == 1
    chi_j_n(1) = omega_j_n(1);
else
    chi_j_n(1) = omega_j_n(1)+omega_j_n(2)/2;
    for j = 2:n-1
        chi_j_n(j) = (omega_j_n(j)+omega_j_n(j+1))/2;
    end
    chi_j_n(n) = omega_j_n(n)/2;
end

% theta and sigma remain explicit inputs so that this interface matches
% the manuscript notation.  eta(j+1)=t_(j-theta), where t_offset is
% generated from theta and sigma.
if abs(theta+sigma-1) > 100*eps
    error('Direct_Time_Weights:OffsetParameters',...
        'theta and sigma must satisfy theta+sigma=1.');
end

end

function difference = stable_power_difference(x,y,power_value)
% Compute the difference of two nearby powers in a cancellation-resistant
% form.
if y <= 0 
    difference = x^power_value; 
else 
    difference = -x^power_value*expm1(power_value*log1p((y-x)/x)); 
end 

end 
