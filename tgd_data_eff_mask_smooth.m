function out = tgd_data_eff_mask_smooth(X,A_true,Mask,W,prms)

[N,~,T] = size(X);
max_iters = prms.max_iters;
R = prms.R;
M = prms.M;
Ni2 = 10;
W=W/(N*M);

beta = 0;%prms.beta; %orthogonality on A
gamma = prms.gamma; %sparsity on A
eta = prms.eta; %Frobenius norm on A
delta = prms.delta; %smoothness penalty
mu = prms.mu; % temporal difference on C
rho = prms.rho; %Frobenius norm on C
alpha = prms.alpha;
cita = 0.01;

A=rand(N^2,R);C=rand(R,T); %initialization


XX = reshape(X,[N^2,T]);
AA = reshape(A_true,[N^2,T]);
WW = reshape(W,[N^2,T]);
Ma = reshape(Mask,size(XX));
%% Proximal subroutine to update A over K steps
Dt=temporal_difference_matrix(T);
% we store the updated A's in column format in a matrix of size N^2 by R
for k=1:max_iters
for rep=1:Ni2 %several updates for A0
for r=1:R
        m=ones(R,1);m(r)=0;Cr = C(r,:)';
        %Z1 = (0.5*WW*kron(Cr,eye(N)))'; 
        %Z3 = reshape(reshape(W,[T,N^2])'*Cr,[N,N]);
        Z = reshape(WW*Cr,[N,N]);
        
        S=zeros(size(XX));
        for n=1:R
            S=S+m(n)*A(:,n)*C(n,:);
        end
        Y = XX-S;%This evaluates Y_r
        B = Y'.*Ma';
        qr = alpha*diag(Ma*diag(Cr)*B); % this is the new qr with the mask
        %antiguo
            %ab=(qr-gamma*ones(N^2,1)-beta*A*m-delta*vec(Z));%This is the vectorized version of D_R in the draft
            %Ab=reshape(ab,N,N);%reshaping it to get D_r
        %nuevo
            Qr = reshape(qr,[N,N])';
            Ab =(Qr-gamma*ones(N)-beta*reshape(A*m,[N,N])-delta*Z);
        H = Ma*diag(Cr)*diag(Cr)*Ma';
        J = reshape(diag(H),[N,N]);
        Ab = my_proj_symmetric_verify2(Ab,A,C,T,R,r,cita,eta)./(eta+alpha*J);
        A(:,r)=vec(Ab);
end
%A = double(A > 0);
%error_int(rep)=norm(AA-A*C','fro')^2/norm(AA,'fro')^2;%error within the A updates
end
%% C update
%% First make the smoothness matrix
% S = A'*reshape(W,[N^2,T]); % this is the same as previous one
% mx = Ma(:).*XX(:);
% B = Ma(:).*kron(eye(T),A);
% F = alpha*(B'*B)+mu*kron(Dt'*Dt,eye(R)) + rho*kron(eye(T),eye(R));
% c = F\(alpha*B'*mx-delta*vec(S));%C update
%%%%%%%%%%%
S = WW'*A; % this is the same as previous one
mt = vec(Ma');xt = vec(XX');
mxt = mt.*xt;
Bt = mt.*kron(A,eye(T));
F = alpha*(Bt'*Bt)+2*mu*kron(eye(R),Dt'*Dt) + rho*kron(eye(R),eye(T));

%(49)
La1 = rand(T,R);La2 = rand(T,N); 
IR = eye(R);IT = eye(T);vone = ones(N,1);Ips = reshape(A,[N,N*R])*kron(IR,vone);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%update La1 (50,51)
Theta = inv(alpha*(Bt'*Bt)+2*mu*kron(eye(R),Dt'*Dt) + rho*kron(eye(R),eye(T)));
IMa = pinv(kron(Ips,IT)*Theta'*kron(Ips',IT));
prms_La1 = struct('ZTA',S,'Ips',Ips,'La2',La2,'B',Bt,'a',mxt,'delta',delta,'alpha',alpha);
prms_La2 = struct('Theta',Theta,'ZTA',S,'Ma',IMa,'La1',La1,'Ips',Ips,'B',Bt,'a',mxt,'delta',delta,'cita',cita,'alpha',alpha);
for ni = 1:5
    La1 = update_La1(prms_La1).La1;
    prms_La2.La1 = La1;
    %update La2 (52,53)
    La2 = update_La2(prms_La2).La2;
    prms_La1.La2 = La2;
    figure(12)
    subplot(121)
    imagesc(La1)
    title('La1')
    colorbar()
    subplot(122)
    imagesc(La2)
    colorbar()
    title(['La2' num2str(ni)])
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%La1 = zeros(T,R);%La2 = zeros(T,N); 
%(49)
c = F\(alpha*2*Bt'*mxt-delta*vec(S) + vec(La1) + vec(La2*Ips));%C update

%Remember to change the reshape for the second formulation
c(c<0) = 0;
C=reshape(c,[T,R])';

figure(6)
subplot(221)
imagesc(F);colorbar()
title(num2str(rank(F)))
subplot(222)
imagesc(B'*B);colorbar()
subplot(223)
imagesc(mu*kron(eye(R),Dt'*Dt));colorbar()
subplot(224)
imagesc(rho*kron(eye(R),eye(T)));colorbar()


figure(2);
subplot(211);plot(c);grid on;
subplot(212);imagesc(C);colorbar();
        %Error of the estimate tensor vs the original one
        errors(k,1) = (norm(AA-A*C,"fro")/norm(AA,"fro"))^2;
        %Error of the estimate tensor vs the observed one 
        errors(k,2) = (norm(XX-A*C,"fro")/norm(XX,"fro"))^2;
        %Error of the estimated ones vs the true ones 
        errors(k,3) = norm(AA-AA.*(A*C),"fro");
        Azt = AA == 0;
        %Error of the estimated zeros vs the true zeros
        errors(k,4) = norm(Azt.*(AA-A*C),"fro");
        fsc = fscore(AA,A*C);
        figure(1)
        subplot(312)
        imagesc(reshape(A*C,[N,N*T]))
        title('Estimated')
        colorbar()
        subplot(313)
        imagesc(reshape(X,[N,N*T]))
        colorbar()
        title('Observed')
        subplot(311)
        imagesc(reshape(AA,[N,N*T]))
        title(['True: Iteration: ' num2str(k) ' error true: ' num2str(errors(k,1)) ' error observed: ' num2str(errors(k,2)) ' fsc: ' num2str(fsc)])
        colorbar()
%obj_fun(k,:) = objective_funtion(AA,A,C,W,Dt,alpha,beta,delta,gamma,prms.M).a_n;

end
X_rec=fac2tens(A,C');%function that reconstructs tensor from A and C

out.A =reshape(A,[N,N*R]);
out.C = C;
out.T_hat = reshape(A*C,[N,N*T]);
out.it_error = zeros(max_iters,5);
out.np = N^2*R+R*T;
figure(3)
plot(errors(:,1))
grid on
end

