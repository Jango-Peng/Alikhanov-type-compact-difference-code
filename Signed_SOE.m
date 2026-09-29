function [s_kappa,omega_kappa,validated_error,fit_info] = ...
    Signed_SOE(kernel_function,delta_min,T,tolerance,...
    show_progress,options)
%SIGNED_SOE Fit a signed sum-of-exponentials representation of a real kernel.
%
%   kappa(q) ~= sum_l omega_kappa(l)*exp(-s_kappa(l)*q),
%   delta_min <= q <= T,
%
% where s_kappa is nonnegative and omega_kappa is unrestricted in sign.
% Unlike the positive Caputo SOE, this signed representation has no
% coefficient-positivity property and must not be assigned the energy or
% monotonicity conclusions that rely on positive weights.
%
% The fit treats kernel_function as a black-box kernel.  An optional known
% logarithmic component may be supplied in options.  The approximation is
% validated numerically on an independent point set; this finite-grid
% check is not, by itself, a proof of a uniform approximation theorem or
% of unconditional stability and convergence of the resulting solver.

if nargin<5 || isempty(show_progress)
    show_progress=false;
end
if nargin<6 || isempty(options)
    options=struct();
end
if ~isstruct(options) || numel(options)~=1
    error('Signed_SOE:Options',...
        'options must be a scalar structure.');
end
if ~isa(kernel_function,'function_handle')
    error('Signed_SOE:KernelHandle',...
        'kernel_function must be a function handle.');
end
validateattributes(delta_min,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(T,{'numeric'},{'scalar','real','finite','>=',delta_min});
validateattributes(tolerance,{'numeric'},{'scalar','real','finite','positive'});

known_log_coefficient=isfield(options,'log_coefficient') && ...
    ~isempty(options.log_coefficient);
if known_log_coefficient
    validateattributes(options.log_coefficient,{'numeric'},...
        {'scalar','real','finite'});
    log_coefficient=options.log_coefficient;
else
    log_coefficient=nan;
end
if ~isfield(options,'candidate_counts') || isempty(options.candidate_counts)
    candidate_counts=[24,36,48,64,80,104,132,168,216,272,344];
else
    candidate_counts=options.candidate_counts(:).';
    if any(candidate_counts<2) || any(candidate_counts~=floor(candidate_counts))
        error('Signed_SOE:CandidateCounts',...
            'candidate_counts must contain integers greater than or equal to two.');
    end
end

% Use a mixed logarithmic/linear training grid to resolve both endpoint
% behavior and the smooth interior part of the kernel.
training_points=unique([logspace(log10(delta_min),log10(T),1200),...
    linspace(delta_min,T,300)]).';
% Keep the validation points independent from the least-squares training grid.
validation_points=logspace(log10(delta_min),log10(T),4000).';
training_values=evaluate_kernel(kernel_function,training_points);
validation_values=evaluate_kernel(kernel_function,validation_points);

% Represent a weak logarithmic component by a separate fixed-rate SOE.
% The remaining smooth residual is fitted by signed exponentials.
if ~known_log_coefficient || log_coefficient~=0
    log_tolerance=max(1e-13,tolerance/20);
    [s_log,omega_log,log_error]=logarithm_soe(...
        delta_min,T,log_tolerance);
    log_training=exp(-training_points*s_log.')*omega_log;
    log_validation=exp(-validation_points*s_log.')*omega_log;
else
    s_log=[];
    omega_log=[];
    log_error=0;
    log_training=zeros(size(training_values));
    log_validation=zeros(size(validation_values));
end
if known_log_coefficient
    base_training=log_coefficient*log_training;
    base_validation=log_coefficient*log_validation;
    residual_training=training_values-base_training;
else
    base_validation=zeros(size(validation_values));
    residual_training=training_values;
end

% Search over candidate exponential counts.  Each candidate is solved by
% a truncated SVD of the design matrix, which avoids amplifying nearly
% dependent exponential columns.
validated_error=inf;
s_kappa=[];
omega_kappa=[];
best_rank=0;
best_candidate=0;
best_log_coefficient=log_coefficient;
if show_progress
    fprintf('signed-SOE fit: ');
end
for candidate=candidate_counts
    % Include the zero rate explicitly.  The remaining rates are positive,
    % while all fitted amplitudes remain unrestricted in sign.
    positive_rates=logspace(log10(1e-3/T),...
        log10(20/delta_min),candidate-1).';
    trial_s=[0;positive_rates];
    exponential_design=exp(-training_points*trial_s.');
    if known_log_coefficient
        design=exponential_design;
    else
        design=[log_training,exponential_design];
    end
    [left_vectors,S,right_vectors]=svd(design,'econ');
    singular_values=diag(S);
    if isempty(singular_values)
        numerical_rank=0;
        trial_coefficients=zeros(size(design,2),1);
    else
        numerical_rank=sum(singular_values>...
            1e-13*singular_values(1));
        if numerical_rank==0
            trial_coefficients=zeros(size(design,2),1);
        else
            trial_coefficients=right_vectors(:,1:numerical_rank)*...
                ((left_vectors(:,1:numerical_rank).'*residual_training)...
                ./singular_values(1:numerical_rank));
        end
    end
    if known_log_coefficient
        trial_log_coefficient=log_coefficient;
        trial_omega=trial_coefficients;
    else
        trial_log_coefficient=trial_coefficients(1);
        trial_omega=trial_coefficients(2:end);
    end
    trial_approximation=base_validation...
        +(~known_log_coefficient)*trial_log_coefficient*log_validation...
        +exp(-validation_points*trial_s.')*trial_omega;
    % Use the maximum absolute error on the independent validation grid,
    % rather than the least-squares training error, as the acceptance metric.
    trial_error=max(abs(trial_approximation-validation_values));
    if trial_error<validated_error
        validated_error=trial_error;
        if isempty(s_log)
            s_kappa=trial_s;
            omega_kappa=trial_omega;
        else
            s_kappa=[s_log;trial_s];
            omega_kappa=[trial_log_coefficient*omega_log;trial_omega];
        end
        best_rank=numerical_rank;
        best_candidate=candidate;
        best_log_coefficient=trial_log_coefficient;
    end
    if show_progress
        fprintf('Q=%d(err %.1e) ',candidate,trial_error);
    end
    if trial_error<=tolerance
        break;
    end
end
if show_progress
    fprintf('\n');
end

s_kappa=real(s_kappa(:));
omega_kappa=real(omega_kappa(:));
fit_info=struct('requested_tolerance',tolerance,...
    'validated_max_abs_error',validated_error,...
    'Q_kappa',numel(s_kappa),'numerical_rank',best_rank,...
    'best_candidate_count',best_candidate,'delta_min',delta_min,...
    'T',T,'log_coefficient',best_log_coefficient,...
    'log_coefficient_was_supplied',known_log_coefficient,...
    'log_soe_error',log_error,'log_soe_terms',numel(s_log),...
    'reached_tolerance',validated_error<=tolerance,...
    'candidate_counts',candidate_counts);
if validated_error>20*tolerance
    warning('Signed_SOE:ToleranceNotReached',...
        ['The independent signed-SOE validation error is %.3e, above the ',...
        'requested tolerance %.3e. Increase the candidate count or ',...
        'relax the fitting interval.'],validated_error,tolerance);
end
end

function values=evaluate_kernel(kernel_function,points)
values=kernel_function(points);
if ~isnumeric(values) || ~isreal(values) || numel(values)~=numel(points)
    error('Signed_SOE:KernelValue',...
        'kernel_function must return a real array with the same length as its input.');
end
values=reshape(values,[],1);
if any(~isfinite(values))
    error('Signed_SOE:KernelValue',...
        'kernel_function returned a non-finite value on the fitting interval.');
end
end

function [s_log,omega_log,validated_error]=logarithm_soe(...
    delta_min,T,tolerance)
% Build a fixed-rate SOE for log(q) on [delta_min,T].  This component is
% useful when the perturbation kernel contains a known logarithmic weak
% singularity near the initial time.  Its error is checked separately
% before it is combined with the fitted residual representation.
step=0.38;
y_min=log(max(tolerance/20,realmin));
y_max=log((T/delta_min)*log(20/tolerance));
indices=floor(y_min/step):ceil(y_max/step);
y=step*indices;
positive_rates=exp(y).'/T;
positive_amplitudes=-step*ones(numel(positive_rates),1);
constant_amplitude=log(T)+step*sum(exp(-exp(y)));
s_log=[0;positive_rates];
omega_log=[constant_amplitude;positive_amplitudes];
points=logspace(log10(delta_min),log10(T),3000).';
approximation=exp(-points*s_log.')*omega_log;
validated_error=max(abs(approximation-log(points)));
end
