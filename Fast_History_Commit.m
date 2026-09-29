function [H_alpha_state,Y_kappa_state] = Fast_History_Commit(...
    n,U_previous,U_new,w_current,H_alpha_state,Y_kappa_state,...
    H_alpha_known,H_alpha_current_shape,Y_kappa_decay,...
    U_hat_previous,delta_j,s_kappa)
%FAST_HISTORY_COMMIT Update the positive Caputo-SOE and signed-SOE states
% after the current solution has been accepted.

if n>=2
    % Add the newly completed Caputo cell to the far-history state.
    H_alpha_state=H_alpha_known...
        +(U_new-U_previous)*H_alpha_current_shape;
end
if n==1
    U_hat=w_current;
else
    U_hat=0.5*(U_hat_previous+w_current);
end
delta_n=delta_j(n);
% Integrate exp(-s_kappa*q) over the current offset interval.  The
% expm1 form is stable for small positive rates; the zero-rate term is
% handled by its exact limiting value delta_n.
phi_kappa=zeros(size(s_kappa));
positive=s_kappa>0;
phi_kappa(positive)=-expm1(-s_kappa(positive)*delta_n)...
    ./s_kappa(positive);
phi_kappa(~positive)=delta_n;
% The signed coefficients may have either sign; no positivity test is
% appropriate for this state.
Y_kappa_state=bsxfun(@times,Y_kappa_state,Y_kappa_decay)...
    +U_hat*phi_kappa.';
end
