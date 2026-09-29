function [s_alpha,omega_alpha,caputo_error,relative_tolerance] = ...
    Fixed_SOE(alpha0,delta,T,tolerance)
%FIXED_SOE Construct a positive-weight SOE for the constant-order Caputo kernel.
%
%   [s_alpha,omega_alpha,caputo_error,relative_tolerance] = ...
%       Fixed_SOE(alpha0,delta,T,tolerance)
%
% The returned coefficients satisfy
%   beta_(1-alpha0)(q) ~= sum_l omega_alpha(l)*exp(-s_alpha(l)*q)
% on [delta,T].  This function contains no perturbation-kernel or
% variable-order-specific code.

validateattributes(alpha0,{'numeric'},{'scalar','real','finite','>',0,'<',1});
validateattributes(delta,{'numeric'},{'scalar','real','finite','positive'});
validateattributes(T,{'numeric'},{'scalar','real','finite','>=',delta});
validateattributes(tolerance,{'numeric'},{'scalar','real','finite','positive'});

relative_tolerance=max(5e-15,min(1e-3,tolerance*delta^alpha0));
[s_alpha,raw_weights]=sum_of_exponentials(...
    alpha0,relative_tolerance,delta,T);
s_alpha=real(s_alpha(:));
omega_alpha=real(raw_weights(:))/gamma(1-alpha0);

validation_lags=logspace(log10(delta),log10(T),3000).';
exact_kernel=validation_lags.^(-alpha0)/gamma(1-alpha0);
approximation=exp(-validation_lags*s_alpha.')*omega_alpha;
caputo_error=max(abs(exact_kernel-approximation));
end

function [nodes,weights]=sum_of_exponentials( ...
    alpha,relative_tolerance,delta,Tfinal)
scaled_delta=delta/Tfinal;
h=2*pi/(log(3)+alpha*log(1/cos(1))+...
    log(1/relative_tolerance));
lower=log(relative_tolerance*gamma(1+alpha))/alpha;
if alpha>=1
    upper=log(1/scaled_delta)+log(log(1/relative_tolerance))...
        +log(alpha)+0.5;
else
    upper=log(1/scaled_delta)+log(log(1/relative_tolerance));
end
left_index=floor(lower/h);
right_index=ceil(upper/h);
negative_grid=left_index:-1;
raw_nodes_1=-exp(h*negative_grid);
raw_weights_1=h/gamma(alpha)*exp(alpha*h*negative_grid);
[weights_1,nodes_1]=prony_compress(raw_nodes_1,raw_weights_1);
positive_grid=0:right_index;
raw_nodes_2=-exp(h*positive_grid);
raw_weights_2=h/gamma(alpha)*exp(alpha*h*positive_grid);
nodes=[-real(nodes_1);-real(raw_nodes_2.')]/Tfinal;
weights=[real(weights_1);real(raw_weights_2.')]/Tfinal^alpha;
end

function [new_weights,new_nodes]=prony_compress(nodes,weights)
number_of_nodes=length(nodes);
moments=zeros(2*number_of_nodes,1);
for index=1:2*number_of_nodes
    moments(index)=nodes.^(index-1)*weights.';
end
H=hankel(moments(1:number_of_nodes),...
    moments(number_of_nodes:2*number_of_nodes-1));
recurrence=rank_deficient_qr(H,-moments,1e-12);
reduced_count=length(recurrence);
V=zeros(2*number_of_nodes,reduced_count);
new_nodes=roots([1;flipud(recurrence)]);
for index=1:2*number_of_nodes
    V(index,:)=new_nodes.^(index-1);
end
new_weights=rank_deficient_svd(V,moments,1e-12);
keep=real(new_nodes)<0;
new_nodes=new_nodes(keep);
new_weights=new_weights(keep);
end

function solution=rank_deficient_qr(A,b,tolerance)
[row_count,~]=size(A);
[Q,R]=qr(A,0);
numerical_rank=sum(abs(diag(R))>tolerance);
if numerical_rank==0
    solution=zeros(0,1);
    return;
end
Q=Q(:,1:numerical_rank);
R=R(1:numerical_rank,1:numerical_rank);
shifted_rhs=b(numerical_rank+1:row_count+numerical_rank);
solution=R\(Q.'*shifted_rhs);
end

function solution=rank_deficient_svd(A,b,tolerance)
[U,S,V]=svd(A,0);
singular_values=diag(S);
if isempty(singular_values)
    solution=zeros(size(A,2),1);
    return;
end
numerical_rank=sum(singular_values>tolerance*singular_values(1));
solution=zeros(size(A,2),1);
for index=1:numerical_rank
    solution=solution+(U(:,index)'*b)/singular_values(index)*V(:,index);
end
end
