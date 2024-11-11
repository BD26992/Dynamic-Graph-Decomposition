function x = my_proj_symmetric_verify(y)
%% Projection of y onto set of element-wise positive and trace zero and symmetric matrices
N=size(y,1);
%% Initialize
mu=randn;Lambda=randn(N);Theta=randn(N);
for n=1:1000
%% Primal Update
%x=y+Lambda-mu*eye(N);
%primal_loss(n)=0.5*norm(x-y,2)^2-trace(Lambda'*x)+mu*trace(x);
%% Dual Update
Lambda=mu*eye(N)+Theta-Theta'-y;Lambda(Lambda<0)=0;
mu=(1/N)*(trace(Lambda+Theta'-Theta+y));
Theta=0.5*(y+Lambda);
end
x=y+Lambda-mu*eye(N)+Theta'-Theta;
end