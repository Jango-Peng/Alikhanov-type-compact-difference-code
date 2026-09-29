function [linear_coefficient,known_vector,H_alpha_known,...
    H_alpha_current_shape,Y_kappa_decay,U_hat_previous] = ...
    Fast_History_Begin(n,U,H_alpha_state,Y_kappa_state,...
    F_current,theta,sigma,tau,rho,delta_j,g_tilde_delta,...
    s_alpha,omega_alpha,a_alpha,b_alpha,tilde_a_0,s_kappa,omega_kappa)
%FAST_HISTORY_BEGIN Assemble the known fast-history contribution at the
% current time level.
%
% The Caputo history is split into a positive-weight SOE recurrence.  The
% perturbation history is represented separately by a signed-SOE.  The
% most recent cell is retained in the local, exact treatment; only the
% completed far history is propagated through exponential states.

U_previous=U(:,n);
if n==1
    H_alpha_known=zeros(size(H_alpha_state));
    A0_fast=tilde_a_0(n);
    caputo_history=zeros(size(U_previous));
    H_alpha_current_shape=zeros(1,numel(s_alpha));
else
    k=n-1;
    decay_previous=exp(-s_alpha.'*tau(k));
    delta_U_previous=U(:,n)-U(:,n-1);
    H_alpha_known=bsxfun(@times,H_alpha_state,decay_previous)...
        +delta_U_previous*(a_alpha(k,:)-b_alpha(k,:));
    current_decay=exp(-s_alpha.'*(sigma*tau(n)));
    H_alpha_current_shape=rho(k)*b_alpha(k,:);
    A0_fast=tilde_a_0(n)+sum(...
        omega_alpha.'.*current_decay.*H_alpha_current_shape);
    caputo_history=H_alpha_known*(omega_alpha.*current_decay.');
end

% Propagate the signed-SOE perturbation states to the current offset time.
Y_kappa_decay=exp(-s_kappa.'*delta_j(n));
far_history=Y_kappa_state*(omega_kappa.*Y_kappa_decay.');
if n==1
    % At the first level the complete first-cell contribution is local.
    local_coefficient=g_tilde_delta(n);
    U_hat_previous=zeros(size(U_previous));
    perturbation_known=far_history;
else
    % At later levels, the current cell is represented by its endpoint
    % contribution, while the signed-SOE state supplies the far history.
    local_coefficient=0.5*g_tilde_delta(n);
    U_hat_previous=theta*U(:,n-1)+sigma*U(:,n);
    perturbation_known=far_history...
        +0.5*g_tilde_delta(n)*U_hat_previous;
end

linear_coefficient=A0_fast/sigma+local_coefficient;
known_vector=caputo_history+perturbation_known...
    -(A0_fast/sigma)*U_previous-F_current;
end
