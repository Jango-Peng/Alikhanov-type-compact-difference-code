function [residual,jacobian] = Direct_Residual_Jacobian(...
    Y,linear_coefficient,known_vector,H,D,C,h,nu)
%DIRECT_RESIDUAL_JACOBIAN Evaluate the one-level nonlinear residual F(Y)
% and its exact Jacobian J(Y).
%
% Y=[w;z], where w=U^(n-theta) and z approximates
% u_xx(t_(n-theta)).

interior_count = size(H,1);
w = Y(1:interior_count);
z = Y(interior_count+1:2*interior_count);

%% 1. Nonlinear compact Burgers term B_h(w,z)
Cw = C*w;
psi_ww = (w.*Cw+C*(w.^2))/3;
psi_zw = (z.*Cw+C*(z.*w))/3;
Burgers_term = psi_ww-h^2*psi_zw/2;

%% 2. Residual F(Y) formed by the two algebraic equations
residual_main = linear_coefficient*w+Burgers_term-nu*z+known_vector;
residual_compact = H*z-D*w;
residual = [residual_main;residual_compact];

%% 3. Exact Jacobian of the residual with respect to w and z
I = speye(interior_count);
diag_w = spdiags(w,0,interior_count,interior_count);
diag_z = spdiags(z,0,interior_count,interior_count);
diag_Cw = spdiags(Cw,0,interior_count,interior_count);

derivative_psi_ww = (diag_Cw+diag_w*C+2*C*diag_w)/3;
derivative_psi_zw_w = (diag_z*C+C*diag_z)/3;
derivative_psi_zw_z = (diag_Cw+C*diag_w)/3;

J11 = linear_coefficient*I+derivative_psi_ww ...
    -h^2*derivative_psi_zw_w/2;
J12 = -nu*I-h^2*derivative_psi_zw_z/2;
J21 = -D;
J22 = H;

jacobian = [J11,J12;J21,J22];
end
