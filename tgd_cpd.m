function out = tgd_cpd(X0,prms)
    [N,~,T] = size(X0);
    R = prms.R;
    RR = 1:R;
    X = reshape(X0,[N^2,T]);
    C = rand(R,T);
    E = zeros(N^2,T);
    At = rand(N^2,R);
    gamma = prms.gamma;
    beta = prms.beta;
    max_iters = prms.max_iters;
    for i = 1:max_iters
        %Estimate A
        for r = RR
            %compute residuals
            idx = RR(RR~=r);
            Z = X-At(:,idx)*C(idx,:)-0.01*E;
            Ar = sum(At(:,idx),2); 
            cvx_begin quiet
                variable A(N^2,1)
                minimize(norm(Z-A*C(r,:),'fro') + 1*gamma*norm(A,1) + 1*beta*A'*Ar)
                subject to  
                    diag(reshape(A,[N,N])) <= 1e-6;
                    A>=0;
            cvx_end

            %update At
            At(:,r) = A;
        end
        %%% Estimate C fixing A
        Z = X-0.01*E;
        cvx_begin quiet
            variable C(R,T)
            minimize(norm(Z-At*C,'fro'))
            subject to
                C>=0;
        cvx_end

        P = 20;
        E_est = tensor(reshape(X-At*C,[N,N,T]));
        E = cp_als(E_est,P,'printitn',0);
        E = reshape(double(E),[N^2,T]);
        error = norm(X-At*C-E,"fro");
        figure(1)
        subplot(211)
        imagesc(reshape(At,[N,N*R]))
        colorbar()
        subplot(212)
        imagesc(reshape(E,[N,N*T]))
        colorbar()
        disp(error)
    end
    out.T_hat = reshape(At*C,[N,N*T]);
    out.A = reshape(At,[N,N*R]);
    out.C = C;
    out.np = N^2*R+R*T + (2*N+T+1)*P;
end