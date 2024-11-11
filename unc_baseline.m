function out = unc_baseline(X,Data,prms)
K = prms.max_iters;
R = size(X,3);
Ni2 = 10;

%gamma = prms.gamma;
%beta = prms.beta;
%delta = prms.delta;
%alpha = prms.alpha;%smooth

%% Molene dataset 10
%alpha = 1e-6;
%gamma = 1e-8;
%beta = 1e-4;
%delta = 1e-3;
%% Molene dataset 50
alpha = 0;
gamma = 0;
beta = 0;
delta = 0;
%%
[N,~,T] = size(X);
A=rand(N^2,R);C=rand(T,R);

XX=[];
for t=1:T
    XX=[XX,vec(X(:,:,t))];
end

%% Proximal subroutine to update A over K steps
N=sqrt(size(XX,1));T=size(C,1);
Dt=temporal_difference_matrix(T);
% we store the updated A's in column format in a matrix of size N^2 by R
for k=1:K
for rep=1:Ni2
for r=1:R
        m=ones(R,1);m(r)=0;
        Z=0.5*(mode_n_matricization(Data,1)*kron(C(:,r),eye(N)))';%XI_r in the draft
        S=zeros(size(XX));
        for n=1:R
            S=S+m(n)*A(:,n)*C(:,n)';
        end
        Y=XX-S;%This evaluates Y_r
        ab=(Y*C(:,r)-gamma*ones(N^2,1)-beta*A*m-delta/T*vec(Z));%This is the vectorized version of D_R in the draft
        Ab=reshape(ab,N,N);%reshaping it to get D_r
        Ab=(1/norm(C(:,r))^2)*Ab;% A update in the draft
        %Ab=Ab/max(max(Ab));
        A(:,r)=vec(Ab);
end
error_int(rep)=norm(XX-A*C','fro')^2/norm(XX,'fro')^2;%error within the A updates
end
%% C update
%% First make the smoothness matrix
for t=1:T
    for r=1:R
        M(t,r)=0.5*trace(reshape(A(:,r),N,N)*Data(:,:,t));
    end
end
c=inv(kron(A'*A,eye(T))+alpha*kron(eye(R),Dt'*Dt))*vec(XX'*A-delta*M);%C update
%c(c<0) = 0;
C=reshape(c,T,R);
error(k)=norm(XX-A*C','fro')^2/norm(XX,'fro')^2;%error at kth global iterate
end
X_rec=fac2tens(A,C);%function that reconstructs tensor from A and C

out.A =reshape(A,[N,N*R]);
out.C = C;
out.T_hat = reshape(X_rec,[N,N*T]);
out.error = error;
out.np = N^2*R+R*T;
end

