function out = sgr_Adata(X0,Y,W,prms)
    [N,~,T] = size(X0);
    M = size(Y,2);
    R = prms.R;
    alpha = 166;%prms.alpha/(2*N*M); 
    gamma = 2.8;%prms.gamma;
    beta = 1;
    max_iters = 1;%prms.max_iters;
    WI = reshape(W,[N*T,N])*eye(N);
    wi = WI(:)';

    col1 = [1; zeros(T-1, 1)];  % First main Toeplitz column
    col2 = [0; -1; zeros(T-2, 1)];  % Second main Toeplitz column
    % Create the block Toeplitz matrix
    B = toeplitz(col1, col2);
    B = B(1:T-1,:);

    for i = 1:max_iters
        %Estimate A

        cvx_begin quiet
            variable A(N^2,T)
            f0 = 0;
            for t = 1:T
                f0 = f0 + gamma*A(:,t)'*A(:,t);
            end
            
            f0 = f0 + alpha*wi*A(:) + beta*norm(B*A','fro');
            
            minimize(f0)
            subject to  
                for t = 1:T
                    trace(reshape(A(:,t),[N,N])) == 0;
                    reshape(A(:,t),[N,N]) == reshape(A(:,t),[N,N])';
                    sum(A(:,t)) >= N;
                end
                A >= 0;
        cvx_end

        %%% Estimate C fixing A
    end
    figure(1)
    imagesc(reshape(A,[N,N*T]))
    colorbar()

    out.T_hat = reshape(A,[N,N*T]);
    out.np = N^2*R+R*T;
end