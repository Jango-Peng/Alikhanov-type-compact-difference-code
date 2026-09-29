function [w,z,iteration_count,residual_norm] = Fast_Newton(...
    linear_coefficient,known_vector,H,D,C,h,nu,w_initial,...
    newton_tolerance,max_newton_iterations,n)
%FAST_NEWTON Solve the complete nonlinear system at one time level with
% the standard, undamped Newton method.

interior_count=size(H,1);
w=w_initial;
z=H\(D*w);
Y=[w;z];
converged=false;
residual_norm=inf;
for iteration_count=1:max_newton_iterations
    [residual,jacobian]=Fast_Residual_Jacobian(...
        Y,linear_coefficient,known_vector,H,D,C,h,nu);
    residual_norm=norm(residual,inf);
    residual_scale=1+norm(known_vector,inf)...
        +abs(linear_coefficient)*max(1,norm(Y,inf));
    if residual_norm<=newton_tolerance*residual_scale
        converged=true;
        break;
    end
    newton_correction=jacobian\residual;
    Y=Y-newton_correction;
end
if ~converged
    residual=Fast_Residual_Jacobian(...
        Y,linear_coefficient,known_vector,H,D,C,h,nu);
    residual_norm=norm(residual,inf);
    residual_scale=1+norm(known_vector,inf)...
        +abs(linear_coefficient)*max(1,norm(Y,inf));
    converged=residual_norm<=newton_tolerance*residual_scale;
end
if ~converged
    error('Fast_Newton:NoConvergence',...
        'Newton failed at time level %d after %d iterations; residual %.3e.',...
        n,max_newton_iterations,residual_norm);
end
w=Y(1:interior_count);
z=Y(interior_count+1:end);
end
