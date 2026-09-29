function value = Exact_Solution(x,T,L,alpha0)
%EXACT_SOLUTION Manufactured solution used in convergence experiments.

value = (1+T.^alpha0).*sin(pi*x/L); % Evaluate the requested time level.
end
