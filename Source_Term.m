function F = Source_Term(x,t,L,alpha0,nu)
%SOURCE_TERM Evaluate the reconstructed-equation source term associated
% with the manufactured solution.
%
% The solution is u=(1+t^alpha0)sin(pi*x/L). The source term is obtained
% by substituting this solution into
% D_t^alpha0 u+(g_tilde'*u)+u*u_x-nu*u_xx.

wave_number = pi/L;
time_amplitude = 1+t^alpha0;

caputo_part = gamma(alpha0+1);

% After integration by parts, the perturbation term contributes
% g_tilde(t)u(0)+(g_tilde*u_t)(t).
g_at_t = G_Tilde(t,alpha0);

if t == 0
    convolution_part = 0;
else
    % The substitution s=t*z^(1/alpha0) removes the endpoint factor
    % s^(alpha0-1) from the transformed integrand.
    integrand = @(z) G_Tilde(...
        t*(1-z.^(1/alpha0)),alpha0);
    convolution_part = t^alpha0*integral(integrand,0,1,...
        'AbsTol',1e-11,'RelTol',1e-9);
end

time_fractional_part = caputo_part+g_at_t+convolution_part;
sin_x = sin(wave_number*x);
cos_x = cos(wave_number*x);

F = time_fractional_part*sin_x ...
    +wave_number*time_amplitude^2*(sin_x.*cos_x) ...
    +nu*wave_number^2*time_amplitude*sin_x;
end
