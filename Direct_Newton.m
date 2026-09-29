function [w,z,iteration_count,residual_norm] = Direct_Newton(...
    linear_coefficient,known_vector,H,D,C,h,nu,w_initial,...
    newton_tolerance,max_newton_iterations,n)
%DIRECT_NEWTON Solve the nonlinear algebraic system at one time level
% with the standard, undamped Newton method.
%
% No damping, backtracking, or line search is used. Each iteration takes
% a full Newton step:
% J(Y^m)*DeltaY^m=F(Y^m), Y^(m+1)=Y^m-DeltaY^m.

interior_count = size(H,1);
w = w_initial;
z = H\(D*w);
Y = [w;z];

converged = false;
residual_norm = inf;

for iteration_count = 1:max_newton_iterations
    [residual,jacobian] = Direct_Residual_Jacobian(...
        Y,linear_coefficient,known_vector,H,D,C,h,nu);
    residual_norm = norm(residual,inf);

    % The discrete matrices contain h^(-2) on fine grids, so use a
    % scale-aware residual criterion.
    residual_scale = 1+norm(known_vector,inf) ...
        +abs(linear_coefficient)*max(1,norm(Y,inf));

    if residual_norm < newton_tolerance*residual_scale
        converged = true;
        break;
    end

    newton_correction = jacobian\residual;
    Y = Y-newton_correction;
end

if ~converged
    [residual,~] = Direct_Residual_Jacobian(...
        Y,linear_coefficient,known_vector,H,D,C,h,nu);
    residual_norm = norm(residual,inf);
    residual_scale = 1+norm(known_vector,inf) ...
        +abs(linear_coefficient)*max(1,norm(Y,inf));
    if residual_norm < newton_tolerance*residual_scale
        converged = true;
    end
end

if ~converged
    error('Direct_Newton:NoConvergence',...
        'Newton failed at time level %d after %d iterations; residual %.3e.',...
        n,max_newton_iterations,residual_norm);
end

w = Y(1:interior_count);
z = Y(interior_count+1:2*interior_count);
end
