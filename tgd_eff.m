function out = tgd_eff(X,A_true,prms)

[N,~,T] = size(X);
K = prms.max_iters;
R = prms.R;
Ni2 = 10;

beta = prms.beta; %orthogonality on A
gamma = prms.gamma; %sparsity on A
eta = prms.eta; %Frobenius norm on A
mu = prms.mu; % temporal difference on C
rho = prms.rho; %Frobenius norm on C

A=rand(N^2,R);C=rand(T,R); %initialization

XX=[];
AA = [];
for t=1:T
    XX=[XX,vec(X(:,:,t))]; % X0 matrix form in the paper
    AA=[AA,vec(A_true(:,:,t))];
end

%% Proximal subroutine to update A over K steps
N=sqrt(size(XX,1));T=size(C,1);
Dt=temporal_difference_matrix(T);
% we store the updated A's in column format in a matrix of size N^2 by R
for k=1:K
for rep=1:Ni2 %several updates for A0
for r=1:R
        m=ones(R,1);m(r)=0;
        S=zeros(size(XX));
        for n=1:R
            S=S+m(n)*A(:,n)*C(:,n)';
        end
        Y=XX-S;%This evaluates Y_r
        ab=(Y*C(:,r)-gamma*ones(N^2,1)-beta*A*m);%This is the vectorized version of D_R in the draft
        Ab=reshape(ab,N,N);%reshaping it to get D_r
        Ab=(1/norm(C(:,r))^2+eta)*my_proj_symmetric_verify(Ab);% A update in the draft
        A(:,r)=vec(Ab);
end
error_int(rep)=norm(AA-A*C','fro')^2/norm(AA,'fro')^2;%error within the A updates
end
%% C update
c=inv(kron(A'*A,eye(T))+ mu*kron(eye(R),Dt'*Dt) + rho*kron(eye(R),eye(T)))*vec(XX'*A);%C update
c(c<0) = 0;
C=reshape(c,T,R);
error(k)=norm(AA-A*C','fro')^2/norm(AA,'fro')^2;%error at kth global iterate
error2 = norm(AA-AA.*(A*C'),"fro");
Azt = AA == 0;
error3 = norm(Azt.*(AA-A*C'),"fro");
fsc = fscore(AA,A*C');

figure(1)
subplot(312)
imagesc(reshape(A*C',[N,N*T]))
colorbar()
subplot(311)
imagesc(reshape(A_true,[N,N*T]))
colorbar()
subplot(313)
imagesc(reshape(XX,[N,N*T]))
colorbar()
title(['Iteration: ' num2str(k) ' error: ' num2str(error(k)) ' error ones: ' num2str(error2) ' error zeros: ' num2str(error3) ' fsc: ' num2str(fsc) ])
%obj_fun(k,:) = objective_funtion(AA,A,C,Data,Dt,alpha,beta,delta,gamma,prms.M).a_n;

end
X_rec=fac2tens(A,C);%function that reconstructs tensor from A and C

out.A =reshape(A,[N,N*R]);
out.C = C;
out.T_hat = reshape(X_rec,[N,N*T]);
out.it_error = zeros(K,5);%obj_fun;
out.np = N^2*R+R*T;
end

