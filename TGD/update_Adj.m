function [A,C,X_rec,error] = update_Adj(XX,A,C,Data,K,R,Ni2,gamma,beta,delta,smooth)
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
        ab=(Y*C(:,r)-gamma*ones(N^2,1)-beta*A*m-delta*vec(Z));%This is the vectorized version of D_R in the draft
        Ab=reshape(ab,N,N);%reshaping it to get D_r
        Ab=(1/norm(C(:,r))^2)*my_proj_symmetric_verify(Ab);% A update in the draft
        Ab=Ab/max(max(Ab));
        A(:,r)=vec(Ab);
end
error_int(rep)=norm(XX-A*C','fro')^2/norm(XX,'fro')^2;%error within the A updates
end
%% C update
%% FIrst make the smoothness matrix
for t=1:T
    for r=1:R
        M(t,r)=0.5*trace(reshape(A(:,r),N,N)*Data(:,:,t));
    end
end
c=inv(kron(A'*A,eye(T))+smooth*kron(eye(R),Dt'*Dt))*vec(XX'*A-delta*M);%C update
C=reshape(c,T,R);
%making c nonegative
%C(C<0) = 0;

error(k)=norm(XX-A*C','fro')^2/norm(XX,'fro')^2;%error at kth global iterate
end
X_rec=fac2tens(A,C);%function that reconstructs tensor from A and C
end

