function out = sgr_Ldata(X0,Y,W,prms)
    [N,~,T] = size(X0);
    R = prms.R;
    M = size(Y,2);
    %alpha = prms.alpha/(N*M); 
    %gamma = prms.gamma;
    %rho = prms.rho;

    alpha = 0.1/(N*M); 
    gamma = 1e-3;
    rho = 1;

    max_iters = 1;%prms.max_iters;
    E = reshape(repmat(eye(N),[1,T]),[N^2,T]);
    
    col1 = [1; zeros(T-1, 1)];  % First main Toeplitz column
    col2 = [0; -1; zeros(T-2, 1)];  % Second main Toeplitz column
    % Create the block Toeplitz matrix
    B = toeplitz(col1, col2);
    B = B(1:T-1,:);
    
    for i = 1:max_iters
        %Estimate A

        cvx_begin quiet
            variable L(N^2,T)
            f0 = 0;
            for t = 1:T
                Yt = Y(:,:,t);
                f0 = f0 + alpha*vec(Yt*Yt')'*L(:,t) + gamma*L(:,t)'*L(:,t);
            end
    
            minimize(f0 + rho*norm(B*L','fro'))
            subject to  
                for t = 1:T
                    trace(reshape(L(:,t),[N,N])) == N;
                    sum(reshape(L(:,t),[N,N])) == 0;
                    reshape(L(:,t),[N,N]) == reshape(L(:,t),[N,N])';
                    L(~E) <= 0;
                end
        cvx_end

        %%% Estimate C fixing A
    end
    for t =1:T
        Lt = reshape(L(:,t),[N,N]);
        A(:,t) = vec(diag(diag(Lt))-Lt);
    end
    figure(1)
    imagesc(reshape(A,[N,N*T]))
    colorbar()

    out.T_hat = reshape(A,[N,N*T]);
    %out.A = reshape(A,[N,N*R]);
    %out.C = C;
    out.np = N^2*R+R*T;
end