function out = tgd_dems_andrei(X,A_true,Mask,W,prms)

[N,~,T] = size(X);
max_iters = prms.max_iters;
R = prms.R;
M = prms.M;
W=W/(N*M);

beta = prms.beta; %orthogonality on A
gamma = prms.gamma; %sparsity on A
eta = prms.eta; %Frobenius norm on A
delta = prms.delta; %smoothness penalty
mu = prms.mu; % temporal difference on C
rho = prms.rho; %Frobenius norm on C
alpha = prms.alpha;%Fitting term
cita = 1;
prms.cita = cita;
prms.la1 = 1e-2;
la2 = 1e-2;
c_iters = 10;

A=rand(N^2,R);C=rand(R,T);La1 = rand(T,N,R);La2 = rand(N,T);c = rand(T*R,1); %initialization

XX = reshape(X,[N^2,T]);
AA = reshape(A_true,[N^2,T]);
WW = reshape(W,[N^2,T]);
Ma = reshape(Mask,size(XX));
% Proximal subroutine to update A over K steps
Dt=temporal_difference_matrix(T);
obj_fun = zeros(max_iters,7);
elaps_time = zeros(max_iters,1);
it_err = zeros(max_iters,1);
it_fsc = zeros(max_iters,1);
% we store the updated A's in column format in a matrix of size N^2 by R

IR = eye(R);IT = eye(T);On = ones(N,1);Otn = ones(T,N);
t1 = tic;
for k=1:max_iters
    % A update

    [A,La1] = update_A(A,C,XX,WW,Ma,La1,prms);

    % C update
    S = WW'*A; % this is the same as previous one
    V4 = diag(sum(Ma));

    %mt = vec(Ma');xt = vec(XX');
    %mxt = mt.*xt;
    %Bt = mt.*kron(A,eye(T));
    Ips = reshape(A,[N,N*R])*kron(IR,On);
    %F = alpha*(Bt'*Bt)+mu*kron(IR,Dt'*Dt) + rho*kron(IR,IT) + la2*kron(Ips'*Ips,IT);
    F = alpha*kron(A'*A,V4) + mu*kron(IR,Dt'*Dt) + rho*kron(IR,IT) + la2*kron(Ips'*Ips,IT);
    
    Q = rand(size(Otn));
    for i = 1:c_iters
        c = F\vec(alpha*V4*XX'*A - delta*S - La2'*Ips + la2*(cita*Otn + Q)*Ips);%C update
        %c = c - 1e-5*(F*c-vec(alpha*V4*XX'*A - delta*S - La2'*Ips + la2*(cita*Otn + Q)*Ips));
        %Remember to change the reshape for the second formulation
        %Projections on C
        c(c<0) = 0;
  
        C=reshape(c,[T,R])';
        %C = diag(1./sum(C,2))*C; 
        %Update Q
        Q = 1/la2*La2' + C'*Ips' - cita*Otn;
        Q(Q<=0) = 0;
        %Update La2
        La2 = La2 + la2*(C'*Ips'-cita*Otn-Q)';
        % figure(6)
        % subplot(231)
        % imagesc(alpha*V4*XX'*A)
        % colorbar()
        % title('V4*XX*A')
        % subplot(232)
        % imagesc(-delta*S)
        % colorbar()
        % title('delta*S')
        % subplot(233)
        % imagesc(-La2'*Ips)
        % colorbar()
        % title('LA2*Ips')
        % subplot(234)
        % imagesc(la2*cita*Otn*Ips)
        % colorbar()
        % title('la2*cita*Otn*Ips')
        %  subplot(235)
        % imagesc(la2*Q*Ips)
        % colorbar()
        % title('la2*cita*Q*Ips')
    end


    % figure(4)
    % subplot(121)
    % imagesc(reshape(A(:,1),[N,N]))
    % colorbar()
    % title('A1')
    % subplot(122)
    % imagesc(reshape(A(:,2),[N,N]))
    % colorbar()
    % title('A2')
    %Xir = trace(reshape(A(:,1),[N,N])*reshape(WW,[N,N*T])*kron(C(1,:)',eye(N)))...
    %    + trace(reshape(A(:,2),[N,N])*reshape(WW,[N,N*T])*kron(C(2,:)',eye(N)));
    
    % figure(2);
    % subplot(211);plot(c);grid on;
    % subplot(212);imagesc(C);colorbar();
    % %Error of the estimate tensor vs the original one
    errors(k,1) = (norm(AA-A*C,"fro")/norm(AA,"fro"))^2;
    % %Error of the estimate tensor vs the observed one 
    % errors(k,2) = (norm(XX-A*C,"fro")/norm(XX,"fro"))^2;
    % %Error of the estimated ones vs the true ones 
    % errors(k,3) = norm(AA-AA.*(A*C),"fro");
    % Azt = AA == 0;
    % %Error of the estimated zeros vs the true zeros
    % errors(k,4) = norm(Azt.*(AA-A*C),"fro");
    % fsc = fscore(AA,A*C);
    % figure(1)
    % subplot(312)
    % imagesc(reshape(A*C,[N,N*T]))
    % title('Estimated')
    % colorbar()
    % subplot(313)
    % imagesc(reshape(X,[N,N*T]))
    % colorbar()
    % title('Observed')
    % subplot(311)
    % imagesc(reshape(AA,[N,N*T]))
    % title(['True: Iteration: ' num2str(k) ' error true: ' num2str(errors(k,1)) ' error observed: ' num2str(errors(k,2)) ' fsc: ' num2str(fsc)])
    % colorbar()
    obj_fun(k,:) = objective_funtion(AA,A,C,W,Dt,prms).a_n;
    it_err(k) = errors(k,1);
    it_fsc(k) = fscore(AA,A*C);
    t2 = toc(t1);
    elaps_time(k) = t2; 
end
X_rec=fac2tens(A,C');%function that reconstructs tensor from A and C

out.A =reshape(A,[N,N*R]);
out.C = C;
out.T_hat = reshape(A*C,[N,N*T]);
out.it_obj = obj_fun;
out.it_err = it_err;
out.it_fsc = it_fsc;
out.it_time = elaps_time;
out.np = N^2*R+R*T;
figure(3)
plot(errors(:,1))
grid on
end

