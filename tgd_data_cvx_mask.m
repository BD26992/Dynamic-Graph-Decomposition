function out = tgd_data_cvx_mask(X0,A_true,Mask,W,prms)
    [N,~,T] = size(X0);
    R = prms.R;
    M = prms.M;
    X = reshape(X0,[N^2,T]);
    Ma = reshape(Mask,[N^2,T]);
    A_t = reshape(A_true,[N^2,T]);
    C = rand(R,T);

    W=W/(M*N); %2*M

    alpha = prms.alpha; %fitting term
    %beta = prms.beta; %orthogonality on A
    delta = prms.delta; %smoothness penalty
    gamma = prms.gamma; %sparsity on A
    eta = prms.eta; %Frobenius norm on A
    mu = prms.mu; % temporal difference on C
    rho = prms.rho; %Frobenius norm on C
   
    P = kron(eye(N),ones(N,1));
    B = ones(T,N);
    Dt=temporal_difference_matrix(T);

    max_iters = prms.max_iters;

    WI = reshape(W,[N*T,N])*eye(N);
    wi = WI(:)';
    errors = zeros(max_iters,4);
    for i = 1:max_iters
        %Estimate A
        cvx_begin quiet
            variable A(N^2,R)
            f0 = 0;
            for r = 1:R
                %f0 = f0 + gamma*sum(A(:,r)) - 5*eta*sum(log(sum(reshape(A(:,r),[N,N]))));
                f0 = f0 + gamma*sum(A(:,r)) + eta*A(:,r)'*A(:,r);
            end
            minimize(alpha*norm(Ma.*(X-A*C),'fro') + f0 + delta*(wi*vec(A*C)))
            subject to  
            for r = 1:R
                %sum(A(:,r)) >= N;
                %sum(reshape(A(:,1),[N,N])*c1 + reshape(A(:,2),[N,N])*c2 + reshape(A(:,3),[N,N])*c3) >= ones(N,1)*2;
                reshape(A(:,r),[N,N]) == reshape(A(:,r),[N,N])';
                trace(reshape(A(:,r),[N,N])) == 0;
                %add orthgonality constraints
            end
            A>=0;
            C'*A'*P >= B;
        cvx_end

        %%% Estimate C fixing A
        cvx_begin quiet
            variable C(R,T)
            minimize(alpha*norm(Ma.*(X-A*C),'fro')+ delta*(wi*vec(A*C))+ rho*norm(C,'fro') + mu*norm(Dt*C','fro'))
            subject to
                C>=0;
                sum(C)>=1;
                C'*A'*P >= B;
        cvx_end
        %Error of the estimate tensor vs the original one
        errors(i,1) = (norm(A_t-A*C,"fro")/norm(A_t,"fro"))^2;
        %Error of the estimate tensor vs the observed one 
        errors(i,2) = (norm(X-A*C,"fro")/norm(X,"fro"))^2;
        %Error of the estimated ones vs the true ones 
        errors(i,3) = norm(A_t-A_t.*(A*C),"fro");
        Azt = A_t == 0;
        %Error of the estimated zeros vs the true zeros
        errors(i,4) = norm(Azt.*(A_t-A*C),"fro");
        fsc = fscore(A_t,A*C);
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
        imagesc(reshape(A_t,[N,N*T]))
        title(['True: Iteration: ' num2str(i) ' error true: ' num2str(errors(i,1)) ' error observed: ' num2str(errors(i,2)) ' fsc: ' num2str(fsc)])
        colorbar()

        figure(5);imagesc(C);colorbar()
    end
    out.T_hat = reshape(A*C,[N,N*T]);
    out.A = reshape(A,[N,N*R]);
    out.C = C;
    out.np = N^2*R+R*T;
end







% function out = tgd_data_cvx_mask(X0,A_true,Mask,W,prms)
%     [N,~,T] = size(X0);
%     R = prms.R;
%     M = prms.M;
%     X = reshape(X0,[N^2,T]);
%     Ma = reshape(Mask,[N^2,T]);
%     A_t = reshape(A_true,[N^2,T]);
%     C = rand(R,T);
% 
%     W=W/(M*N); %2*M
% 
%     alpha = prms.alpha; %fitting term
%     %beta = prms.beta; %orthogonality on A
%     delta = prms.delta; %smoothness penalty
%     gamma = prms.gamma; %sparsity on A
%     eta = prms.eta; %Frobenius norm on A
%     mu = prms.mu; % temporal difference on C
%     rho = prms.rho; %Frobenius norm on C
%    
%     Dt=temporal_difference_matrix(T);
% 
%     max_iters = prms.max_iters;
% 
%     WI = reshape(W,[N*T,N])*eye(N);
%     wi = WI(:)';
%     errors = zeros(max_iters,4);
%     for i = 1:max_iters
%         %Estimate A
%         cvx_begin quiet
%             variable A(N^2,R)
%             f0 = 0;
%             for r = 1:R
%                 %f0 = f0 + gamma*sum(A(:,r)) - 5*eta*sum(log(sum(reshape(A(:,r),[N,N]))));
%                 f0 = f0 + gamma*sum(A(:,r)) + eta*A(:,r)'*A(:,r);
%             end
%             minimize(alpha*norm(Ma.*(X-A*C),'fro') + f0 + delta*(wi*vec(A*C)))
%             subject to  
%             for r = 1:R
%                 sum(A(:,r)) >= N;
%                 %sum(reshape(A(:,1),[N,N])*c1 + reshape(A(:,2),[N,N])*c2 + reshape(A(:,3),[N,N])*c3) >= ones(N,1)*2;
%                 reshape(A(:,r),[N,N]) == reshape(A(:,r),[N,N])';
%                 trace(reshape(A(:,r),[N,N])) == 0;
%                 %add orthgonality constraints
%             end
%             A>=0;
%         cvx_end
% 
%         %%% Estimate C fixing A
%         cvx_begin quiet
%             variable C(R,T)
%             minimize(alpha*norm(Ma.*(X-A*C),'fro')+ delta*(wi*vec(A*C))+ rho*norm(C,'fro') + mu*norm(Dt*C','fro'))
%             subject to
%                 C>=0;
%                 sum(C)>=1;
%         cvx_end
%         %Error of the estimate tensor vs the original one
%         errors(i,1) = (norm(A_t-A*C,"fro")/norm(A_t,"fro"))^2;
%         %Error of the estimate tensor vs the observed one 
%         errors(i,2) = (norm(X-A*C,"fro")/norm(X,"fro"))^2;
%         %Error of the estimated ones vs the true ones 
%         errors(i,3) = norm(A_t-A_t.*(A*C),"fro");
%         Azt = A_t == 0;
%         %Error of the estimated zeros vs the true zeros
%         errors(i,4) = norm(Azt.*(A_t-A*C),"fro");
%         fsc = fscore(A_t,A*C);
%         figure(1)
%         subplot(312)
%         imagesc(reshape(A*C,[N,N*T]))
%         title('Estimated')
%         colorbar()
%         subplot(313)
%         imagesc(reshape(X,[N,N*T]))
%         colorbar()
%         title('Observed')
%         subplot(311)
%         imagesc(reshape(A_t,[N,N*T]))
%         title(['True: Iteration: ' num2str(i) ' error true: ' num2str(errors(i,1)) ' error observed: ' num2str(errors(i,2)) ' fsc: ' num2str(fsc)])
%         colorbar()
% 
%         figure(5);imagesc(C);colorbar()
%     end
%     out.T_hat = reshape(A*C,[N,N*T]);
%     out.A = reshape(A,[N,N*R]);
%     out.C = C;
%     out.np = N^2*R+R*T;
% end