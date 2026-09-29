function [g_tilde,kappa] = Alpha_Kernels( ...
    s,alpha0,alpha_function,dalpha_function)
%ALPHA_KERNELS Evaluate g_tilde and kappa for a general alpha.
%
%   K_alpha(s)=s^(-alpha(s))/Gamma(1-alpha(s)),
%   K_0(s)=s^(-alpha0)/Gamma(1-alpha0),
%   g_tilde(s)=K_alpha(s)-K_0(s), kappa(s)=g_tilde'(s).
%
% Only positive s are used for kappa.  At s=0, g_tilde is defined as its
% continuous value 0 under the variable-order assumptions, while kappa is
% returned as NaN because it has a logarithmic weak singularity.

provided_alpha=nargin>=3 && ~isempty(alpha_function);
provided_dalpha=nargin>=4 && ~isempty(dalpha_function);
if xor(provided_alpha,provided_dalpha)
    error('Alpha_Kernels:FunctionHandlePair',...
        'alpha_function and dalpha_function must be supplied together.');
end
if ~provided_alpha
    alpha_function=@(q) alpha0+0.2*q.^(1+alpha0);
    dalpha_function=@(q) 0.2*(1+alpha0)*q.^alpha0;
end
validateattributes(alpha0,{'numeric'},{'scalar','real','finite','>',0,'<',1});
if ~isa(alpha_function,'function_handle') || ...
        ~isa(dalpha_function,'function_handle')
    error('Alpha_Kernels:FunctionHandle',...
        'alpha_function and dalpha_function must be function handles.');
end
if ~isnumeric(s) || ~isreal(s) || any(~isfinite(s(:))) || any(s(:)<0)
    error('Alpha_Kernels:Lag',...
        'The lag variable must be finite and nonnegative.');
end

g_tilde=zeros(size(s));
kappa=nan(size(s));
positive=s>0;
if ~any(positive(:))
    return;
end
sp=s(positive);
direct_alpha=evaluate_order(alpha_function,sp,'alpha_function');
dalpha_s=evaluate_order(dalpha_function,sp,'dalpha_function');
if any(direct_alpha<=0) || any(direct_alpha>=1)
    error('Alpha_Kernels:OrderRange',...
        'alpha_function must satisfy 0<alpha(s)<1 at kernel evaluation points.');
end
if any(~isfinite(dalpha_s))
    error('Alpha_Kernels:Derivative',...
        'dalpha_function returned a non-finite value.');
end

% Form alpha(s)-alpha0 by integrating dalpha_function rather than
% subtracting two nearly equal floating-point numbers.  This preserves the
% initial layer on very fine graded meshes.
direct_delta=direct_alpha-alpha0;
integrated_delta=stable_order_increment(sp,alpha0,dalpha_function);
reliable=abs(direct_delta)>sqrt(eps)*max(1,abs(alpha0));
if any(reliable)
    mismatch=abs(integrated_delta(reliable)-direct_delta(reliable));
    scale=max(1,abs(direct_delta(reliable)));
    if any(mismatch>1e-7*scale)
        error('Alpha_Kernels:InconsistentDerivatives',...
            'alpha_function and dalpha_function are numerically inconsistent.');
    end
end
delta_alpha=integrated_delta;
delta_alpha(reliable)=direct_delta(reliable);
alpha_s=alpha0+delta_alpha;
if any(alpha_s<=0) || any(alpha_s>=1)
    error('Alpha_Kernels:OrderRange',...
        'The alpha(s) recovered from dalpha_function must satisfy 0<alpha(s)<1.');
end

log_s=log(sp);
K0=exp(-alpha0*log_s-gammaln(1-alpha0));
delta_log=stable_kernel_log_ratio(...
    log_s,delta_alpha,alpha_s,alpha0);
g_tilde(positive)=K0.*expm1(delta_log);

% d/ds log(K_alpha/K_0)
delta_prime=-dalpha_s.*log_s-delta_alpha./sp...
    +psi(1-alpha_s).*dalpha_s;
kappa(positive)=(-alpha0*K0./sp).*expm1(delta_log)...
    +K0.*exp(delta_log).*delta_prime;
end

function values=evaluate_order(function_handle,points,name)
values=function_handle(points);
if ~isnumeric(values) || ~isreal(values) || numel(values)~=numel(points)
    error('Alpha_Kernels:FunctionValue',...
        '%s must return a real array with the same length as its input.',name);
end
values=reshape(values,size(points));
if any(~isfinite(values))
    error('Alpha_Kernels:FunctionValue',...
        '%s returned a non-finite value.',name);
end
end

function delta_alpha=stable_order_increment(points,alpha0,dalpha_function)
% Fixed Gauss--Legendre quadrature after u=s*z^(1/(1+alpha0)).
% The transformation removes the initial s^alpha0 cusp allowed by the
% variable-order assumptions and keeps the increment accurate for tiny s.
persistent z weights transformed_weights cached_alpha0
if isempty(z) || isempty(cached_alpha0) || cached_alpha0~=alpha0
    quadrature_order=32;
    jacobi=diag((1:quadrature_order-1)./...
        sqrt(4*(1:quadrature_order-1).^2-1),1);
    jacobi=jacobi+jacobi.';
    [vectors,nodes]=eig(jacobi);
    nodes=diag(nodes);
    [nodes,ordering]=sort(nodes);
    weights=2*(vectors(1,ordering).').^2;
    z=(nodes+1)/2;
    weights=weights/2;
    p=1/(1+alpha0);
    transformed_weights=weights.*p.*z.^(p-1);
    cached_alpha0=alpha0;
end
input_size=size(points);
points=points(:);
p=1/(1+alpha0);
u=points.*(z.'.^p);
derivative_values=dalpha_function(u(:));
if ~isnumeric(derivative_values) || ~isreal(derivative_values) ||...
        numel(derivative_values)~=numel(u)
    error('Alpha_Kernels:DerivativeValue',...
        'dalpha_function must accept vector input and return an array of the same length.');
end
derivative_values=reshape(derivative_values,size(u));
delta_alpha=reshape(points.*(derivative_values*transformed_weights),input_size);
end

function delta_log=stable_kernel_log_ratio(...
    log_s,delta_alpha,alpha_s,alpha0)
delta_log=zeros(size(log_s));
small=abs(delta_alpha)<1e-4;
if any(small(:))
    d=delta_alpha(small);
    q=1-alpha0;
    gamma_difference_negative=psi(q)*d-0.5*psi(1,q)*d.^2 ...
        +(psi(2,q)/6)*d.^3-(psi(3,q)/24)*d.^4;
    delta_log(small)=-d.*log_s(small)+gamma_difference_negative;
end
large=~small;
if any(large(:))
    delta_log(large)=-delta_alpha(large).*log_s(large)...
        -gammaln(1-alpha_s(large))+gammaln(1-alpha0);
end
end
