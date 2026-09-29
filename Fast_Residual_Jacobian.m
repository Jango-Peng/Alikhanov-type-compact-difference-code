function [residual,jacobian,nonlinear_term] = ...
    Fast_Residual_Jacobian(...
    Y,linear_coefficient,known_vector,H,D,C,h,nu)
%FAST_RESIDUAL_JACOBIAN Evaluate the compact nonlinear residual and its
% exact Jacobian for the fast scheme.

interior_count=size(H,1);
w=Y(1:interior_count);
z=Y(interior_count+1:end);
Cw=C*w;
psi_ww=(w.*Cw+C*(w.^2))/3;
psi_zw=(z.*Cw+C*(z.*w))/3;
nonlinear_term=psi_ww-0.5*h^2*psi_zw;
residual=[linear_coefficient*w+nonlinear_term-nu*z+known_vector;...
    H*z-D*w];
if nargout>1
    I=speye(interior_count);
    diag_w=spdiags(w,0,interior_count,interior_count);
    diag_z=spdiags(z,0,interior_count,interior_count);
    diag_Cw=spdiags(Cw,0,interior_count,interior_count);
    Jww=(diag_Cw+diag_w*C+2*C*diag_w)/3;
    Jzw_w=(diag_z*C+C*diag_z)/3;
    Jzw_z=(diag_Cw+C*diag_w)/3;
    J11=linear_coefficient*I+Jww-0.5*h^2*Jzw_w;
    J12=-nu*I-0.5*h^2*Jzw_z;
    jacobian=[J11,J12;-D,H];
end
end
