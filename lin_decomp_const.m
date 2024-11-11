function out = lin_decomp_const(X0,prms)
    [N,~,T] = size(X0);
    R = prms.R;
    RR = 1:R;
    X = reshape(X0,[N^2,T]);
    C = rand(R,T);
    At = rand(N^2,R);
    gamma = prms.gamma;
    max_iters = prms.max_iters;
    for i = 1:max_iters
        %Estimate A
        for r = RR
            %compute residuals
            idx = RR(RR~=r);
            Z = X-At(:,idx)*C(idx,:);
            Ar = sum(At(:,idx),2); 
            cvx_begin quiet
                variable A(N^2,1)
                minimize(norm(Z-A*C(r,:),'fro') + gamma*norm(A,1))
                subject to  
                    diag(reshape(A,[N,N])) <= 1e-6;
                    A>=0;
                    A'*Ar == 0;
            cvx_end

            %update At
            At(:,r) = A;
        end
        %%% Estimate C fixing A
        cvx_begin quiet
            variable C(R,T)
            minimize(norm(X-At*C,'fro'))
            subject to
                C>=0;
        cvx_end
        error = norm(X-At*C,"fro");
        %disp(error)
    end
    out.T_hat = reshape(At*C,[N,N*T]);
    out.A = reshape(At,[N,N*R]);
    out.C = C;
    out.np = N^2*R+R*T;
end