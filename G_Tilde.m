function value = G_Tilde(s,alpha0)
%G_TILDE Evaluate the perturbation kernel g_tilde(s)=K(s)-K_0(s).
%
% This example uses alpha(s)=alpha0+0.2*s^(1+alpha0).

value = zeros(size(s));
positive_index = s>0; % Identify entries for which the lag is strictly positive.

if any(positive_index(:))
    positive_s = s(positive_index);
    alpha_s = alpha0+0.2*positive_s.^(1+alpha0);

    if any(alpha_s>=1)
        error('G_Tilde:OrderRange',...
            'The variable order alpha(s) must remain strictly below one.');
    end

    variable_kernel = positive_s.^(-alpha_s)./gamma(1-alpha_s);
    constant_kernel = positive_s.^(-alpha0)/gamma(1-alpha0);
    value(positive_index) = variable_kernel-constant_kernel;
end

end
