function out = tgd_cvx_mask(X0,A_true,Mask,prms)
    [N,~,T] = size(X0);
    R = prms.R;
    X = reshape(X0,[N^2,T]);
    Ma = reshape(Mask,[N^2,T]);
    A_t = reshape(A_true,[N^2,T]);
    C = rand(R,T);
    
    alpha = prms.alpha; %fitting term
    %beta = prms.beta; %orthogonality on A
    %delta = prms.delta; %smoothness penalty
    gamma = prms.gamma; %sparsity on A
    eta = prms.eta; %Frobenius norm on A
    mu = prms.mu; % temporal difference on C
    rho = prms.rho; %Frobenius norm on C

    Dt=temporal_difference_matrix(T);
    max_iters = prms.max_iters;
    errors = zeros(max_iters,4);

    for i = 1:max_iters
        %Estimate A
        cvx_begin quiet
            variable A(N^2,R)
            f0 = 0;
            for r = 1:R
                f0 = f0 + eta*A(:,r)'*A(:,r) + gamma*sum(A(:,r));
            end
            minimize(alpha*norm(Ma.*(X-A*C),'fro') + f0)
            subject to  
            for r = 1:R
                sum(A(:,r)) >= N;
                reshape(A(:,r),[N,N]) == reshape(A(:,r),[N,N])';
                trace(reshape(A(:,r),[N,N])) == 0;
                %add orthgonality constraints
            end
            A>=0;
        cvx_end

        %%% Estimate C fixing A

        cvx_begin quiet
            variable C(R,T)
            minimize(alpha*norm(Ma.*(X-A*C),'fro') + rho*norm(C,'fro') + mu*norm(Dt*C','fro'))
            subject to
                C>=0;
                sum(C)>=1;
        cvx_end
        T0 = A_t==0; %mask for the true zeros
        T1 = A_t == 1; %the true ones 
        %Error of the estimate tensor vs the original one
        errors(i,1) = (norm(A_t-A*C,"fro")/norm(A_t,"fro"))^2;
        %Error of the estimate tensor vs the observed one 
        errors(i,2) = (norm(Ma.*(X-A*C),"fro")/norm(X,"fro"))^2;
        %Error of the estimated ones vs the true ones 
        errors(i,3) = (norm(T1.*(A_t-A*C),"fro")/norm(A_t,"fro"))^2;
        %Error of the estimated zeros vs the true zeros
        errors(i,4) = (norm(T0.*(A_t-A*C),"fro")/norm(A_t,"fro"))^2;
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
        title(['True: Iteration: ' num2str(i) ' True err.: ' num2str(errors(i,1)) ' Obs. err.: ' num2str(errors(i,2)) ' Ones err.: ' num2str(errors(i,2)) ' Zeros err.: ' num2str(errors(i,4)) ' fsc: ' num2str(fsc)])
        colorbar()
    end
    out.T_hat = reshape(A*C,[N,N*T]);
    out.A = reshape(A,[N,N*R]);
    out.C = C;
    out.np = N^2*R+R*T;
    out.errors=errors;
end


%%
% function out = tgd_data(X0,W,prms)
%     [N,~,T] = size(X0);
%     R = prms.R;
%     RR = 1:R;
%     X = reshape(X0,[N^2,T]);
%     Wr = reshape(W,[N^2,T])';
%     C = rand(R,T);
%     At = rand(N^2,R);
%     alpha = 1e-1;%prms.alpha; 
%     gamma = prms.gamma;
%     beta = prms.beta;
%     rho = 0;
%     lambda = 1e-3;
%     max_iters = prms.max_iters;
%     ON = ones(N,1);
%     for i = 1:max_iters
%         %Estimate A
%         for r = RR
%             %compute residuals
%             idx = RR(RR~=r);
%             Z = X-At(:,idx)*C(idx,:);
%             Ar = sum(At(:,idx),2); 
%             Wc = reshape(C(r,:)*Wr,[N,N]);
%             cvx_begin quiet
%                 variable A(N^2,1)
%                 expressions L(N,N)
%                 L = diag(diag(reshape(A,[N,N])))-reshape(A,[N,N]);
%                 minimize(rho*norm(Z-A*C(r,:),'fro') + alpha*trace(B*Wc) + gamma*norm(A,1) + beta*A'*Ar)
%                 subject to  
%                     diag(B) <= 1e-6;
%                     sum(B) >=1;
%                     B == B';
%                     A>=0;
%             cvx_end
% 
%             %update At
%             At(:,r) = A;
%         end
%         %%% Estimate C fixing A
%         for m = 1:T
%             Zm = W(:,:,m);
%             for n = 1:R
%                 An = reshape(At(:,n),[N,N]);
%                 Q(n,m) = 1/2*trace(Zm*An);
%             end
%         end
% 
%         cvx_begin quiet
%             variable C(R,T)
%             minimize(rho*norm(X-At*C,'fro') + alpha*sum(sum(Q.*C)))
%             subject to
%                 C>=0;
%         cvx_end
%         error = norm(X-At*C,"fro");
%         figure(1)
%         imagesc(reshape(At,[N,N*R]))
%         colorbar()
%         disp(error)
%     end
%     out.T_hat = reshape(At*C,[N,N*T]);
%     out.A = reshape(At,[N,N*R]);
%     out.C = C;
%     out.np = N^2*R+R*T;
% end