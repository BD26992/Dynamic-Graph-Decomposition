function out = tgd_data_eff(X,Data,true_A,prms)

[N,~,T] = size(X);
K = prms.max_iters;
R = prms.R;
M = prms.M;
Ni2 = 10;

gamma = prms.gamma;%/(N*R);
beta = prms.beta;%/(N*R);
delta = prms.delta;%/(N*M);%smoothness
alpha = prms.alpha;%/(N*T);
eta = 1e-4;
rho = 1e-4;

A=rand(N^2,R);C=rand(T,R); %initialization

XX=[];
AA = [];
for t=1:T
    XX=[XX,vec(X(:,:,t))]; % X0 matrix form in the paper
    AA=[AA,vec(true_A(:,:,t))];
end

%% Proximal subroutine to update A over K steps
N=sqrt(size(XX,1));T=size(C,1);
Dt=temporal_difference_matrix(T);
% we store the updated A's in column format in a matrix of size N^2 by R
for k=1:K
for rep=1:Ni2 %several updates for A0
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
        Ab=(1/norm(C(:,r))^2+eta)*my_proj_symmetric_verify(Ab);% A update in the draft
        %Ab=Ab/max(max(Ab));
        A(:,r)=vec(Ab);
end
%A = double(A > 0);
error_int(rep)=norm(AA-A*C','fro')^2/norm(AA,'fro')^2;%error within the A updates
end
%% C update
%% First make the smoothness matrix
for t=1:T
    for r=1:R
        M(t,r)=0.5*trace(reshape(A(:,r),N,N)*Data(:,:,t));
    end
end
c=inv(kron(A'*A,eye(T))+alpha*kron(eye(R),Dt'*Dt) + kron(eye(R),rho*eye(T)))*vec(XX'*A-delta*M);%C update
figure(2);
subplot(311);plot(c);grid on;
subplot(312);imagesc(reshape(c,T,R)');colorbar();
c(c<0) = 0;
C=reshape(c,T,R);
subplot(313);imagesc(C');colorbar();
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
imagesc(reshape(true_A,[N,N*T]))
colorbar()
subplot(313)
imagesc(reshape(XX,[N,N*T]))
colorbar()
title(['Iteration: ' num2str(k) ' error: ' num2str(error(k)) ' error ones: ' num2str(error2) ' error zeros: ' num2str(error3) ' fsc: ' num2str(fsc) ])
obj_fun(k,:) = objective_funtion(AA,A,C,Data,Dt,alpha,beta,delta,gamma,prms.M).a_n;

end
X_rec=fac2tens(A,C);%function that reconstructs tensor from A and C

out.A =reshape(A,[N,N*R]);
out.C = C;
out.T_hat = reshape(X_rec,[N,N*T]);
out.it_error = obj_fun;
out.np = N^2*R+R*T;
end

