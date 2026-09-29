function [theta,sigma,t,tau,rho,t_offset,delta_j,source_scalar,...
    g_tilde_delta,s_alpha,omega_alpha,a_alpha,b_alpha,tilde_a_0,...
    s_kappa,omega_kappa,Q_alpha,Q_kappa,caputo_error,kappa_error,...
    admissible_error_bound,admissible_ok] = Fast_Prepare_Time(...
    T,N,alpha0,r,epsilon_alpha,epsilon_kappa,show_progress,...
    alpha_function,dalpha_function)
%FAST_PREPARE_TIME Generate the time grid, the two independent SOE
% representations, and the recurrence arrays used by the fast scheme.

if nargin<7 || isempty(show_progress)
    show_progress=false;
end
provided_alpha=nargin>=8 && ~isempty(alpha_function);
provided_dalpha=nargin>=9 && ~isempty(dalpha_function);
if xor(provided_alpha,provided_dalpha)
    error('Fast_Prepare_Time:FunctionHandlePair',...
        'alpha_function and dalpha_function must be supplied together.');
end
if ~provided_alpha
    alpha_function=@(q) alpha0+0.2*q.^(1+alpha0);
    dalpha_function=@(q) 0.2*(1+alpha0)*q.^alpha0;
end
if ~isa(alpha_function,'function_handle') || ...
        ~isa(dalpha_function,'function_handle')
    error('Fast_Prepare_Time:FunctionHandle',...
        'alpha_function and dalpha_function must both be function handles.');
end
validate_alpha_data(T,alpha0,alpha_function,dalpha_function);

theta=alpha0/2;
sigma=1-theta;
t=T*((0:N)/N).^r;
tau=diff(t);
rho=tau(1:end-1)./tau(2:end);
t_offset=theta*t(1:N)+sigma*t(2:N+1);
delta_j=diff([0,t_offset]);

% 1. Positive-weight SOE for the constant-order Caputo kernel.
Delta_t=max([sigma*tau(1),1e-11]);
[s_alpha,omega_alpha,caputo_error]=Fixed_SOE(...
    alpha0,Delta_t,T,epsilon_alpha);

beta_T=T^(-alpha0)/gamma(1-alpha0);
if N>=2
    mesh_factor=min((1+5*rho).*tau(1:end-1));
    condition_mesh=alpha0*mesh_factor/(105*T);
else
    condition_mesh=inf;
end
admissible_error_bound=beta_T*min(...
    [alpha0/(2*(1-alpha0)),1/26,condition_mesh]);
% This flag retains the historical public meaning: requested SOE tolerance.
admissible_ok=caputo_error<=epsilon_alpha;

Q_alpha=numel(s_alpha);
a_alpha=zeros(max(N-1,0),Q_alpha);
b_alpha=zeros(max(N-1,0),Q_alpha);
for k=1:N-1
    z=s_alpha.'*tau(k);
    phi0=zeros(size(z));
    phi1=zeros(size(z));
    small=abs(z)<1e-4;
    large=~small;
    phi0(large)=-expm1(-z(large))./z(large);
    phi1(large)=(1-(1+z(large)).*exp(-z(large)))./z(large).^2;
    zs=z(small);
    phi0(small)=1-zs/2+zs.^2/6-zs.^3/24+zs.^4/120;
    phi1(small)=1/2-zs/3+zs.^2/8-zs.^3/30+zs.^4/144;
    a_alpha(k,:)=phi0;
    b_alpha(k,:)=tau(k)/(tau(k)+tau(k+1))*(phi0-2*phi1);
end
tilde_a_0=sigma^(1-alpha0)/gamma(2-alpha0)*tau.^(-alpha0);

% 2. Build the perturbation kernel from variable-order alpha(t), then
% fit it with the general signed-SOE routine.  These approximations are
% independent: Caputo weights are positive, while perturbation weights
% may have either sign.
g_tilde_function=@(s) select_alpha_kernel(...
    s,alpha0,alpha_function,dalpha_function,'g');
kappa_function=@(s) select_alpha_kernel(...
    s,alpha0,alpha_function,dalpha_function,'kappa');
delta_min=min(delta_j);
[s_kappa,omega_kappa,kappa_error]=Signed_SOE(...
    kappa_function,delta_min,T,epsilon_kappa,show_progress,...
    struct());
s_kappa=s_kappa(:);
omega_kappa=omega_kappa(:);
Q_kappa=numel(s_kappa);

source_scalar=zeros(1,N);
for n=1:N
    source_scalar(n)=local_source_scalar(...
        t_offset(n),alpha0,g_tilde_function);
end
g_tilde_delta=g_tilde_function(delta_j);

if show_progress
    fprintf('Caputo SOE: Q_alpha=%d, maximum absolute error %.3e\n',...
        Q_alpha,caputo_error);
    fprintf('Fast-Alikhanov admissible bound %.3e, status %d\n',...
        admissible_error_bound,admissible_ok);
    fprintf('Perturbation-kernel signed-SOE: Q_kappa=%d, maximum absolute error %.3e\n',...
        Q_kappa,kappa_error);
end
end

function validate_alpha_data(T,alpha0,alpha_function,dalpha_function)
if ~isscalar(T) || ~isreal(T) || ~isfinite(T) || T<=0
    error('Fast_Prepare_Time:TimeInterval',...
        'T must be a positive finite scalar.');
end
sample=[0,linspace(T/200,T,200),logspace(log10(max(T*1e-12,realmin)),...
    log10(T),200)];
sample=unique(sample(:));
alpha_values=alpha_function(sample);
if ~isnumeric(alpha_values) || ~isreal(alpha_values) ||...
        numel(alpha_values)~=numel(sample)
    error('Fast_Prepare_Time:AlphaValue',...
        'alpha_function must return a real array with the same length as its input.');
end
alpha_values=reshape(alpha_values,[],1);
if any(~isfinite(alpha_values)) || any(alpha_values<=0) ||...
        any(alpha_values>=1)
    error('Fast_Prepare_Time:AlphaRange',...
        'alpha_function must satisfy 0<alpha(t)<1 at all check points.');
end
if abs(alpha_values(1)-alpha0)>100*eps(max([1,abs(alpha0)]))
    error('Fast_Prepare_Time:Alpha0Mismatch',...
        'alpha_function(0) must equal alpha0.');
end
positive_sample=sample(2:end);
dalpha_values=dalpha_function(positive_sample);
if ~isnumeric(dalpha_values) || ~isreal(dalpha_values) ||...
        numel(dalpha_values)~=numel(positive_sample) ||...
        any(~isfinite(dalpha_values(:)))
    error('Fast_Prepare_Time:DerivativeValue',...
        'dalpha_function must return a finite real array with the same length as its input.');
end
end

function value=select_alpha_kernel(...
    s,alpha0,alpha_function,dalpha_function,quantity)
[g_value,kappa_value]=Alpha_Kernels(...
    s,alpha0,alpha_function,dalpha_function);
if strcmp(quantity,'g')
    value=g_value;
else
    value=kappa_value;
end
end

function value=local_source_scalar(t,alpha0,g_tilde_function)
if t<=0
    value=0;
    return;
end
convolution_part=t^alpha0*quadgk(@(z) ...
    g_tilde_function(t*(1-z.^(1/alpha0))),0,1,...
    'AbsTol',1e-13,'RelTol',1e-11,'MaxIntervalCount',2000);
value=gamma(alpha0+1)+g_tilde_function(t)+convolution_part;
end
