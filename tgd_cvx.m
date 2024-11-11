function out = tgd_cvx(X0,A_true,prms)
    [N,~,T] = size(X0);
    R = prms.R;
    X = reshape(X0,[N^2,T]);
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
            minimize(alpha*norm(X-A*C,'fro') + f0)
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
            minimize(alpha*norm(X-A*C,'fro') + rho*norm(C,'fro') + mu*norm(Dt*C','fro'))
            subject to
                C>=0;
                sum(C)>=1;
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
    end
    out.T_hat = reshape(A*C,[N,N*T]);
    out.A = reshape(A,[N,N*R]);
    out.C = C;
    out.np = N^2*R+R*T;
    out.errors=errors;
end