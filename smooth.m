function out = smooth(W,prms)
    [N,~,T] = size(W);    
    M = prms.M;
    nn = N^2;
    nm = N*M;
    alpha = prms.alpha*0.5/nm;
    beta = prms.beta/nn;
    gamma = prms.gamma/nn;
    on = ones(1,N); %column vector
    
    T_hat = zeros(N,N,T);
    for t = 1:T
        Wt = W(:,:,t);
        cvx_begin quiet
            variable A_hat(N,N) symmetric
        
            minimize(3*alpha*on*(A_hat.*Wt)*on' - beta*log(on*A_hat)*on' + 20*gamma*norm(A_hat,'fro'))
            %minimize(alpha*on*(A_hat.*Wt)*on' + beta*on*A_hat*on' + gamma*norm(A_hat,'fro'))
            subject to 
                A_hat >= 0;
                diag(A_hat) == 0;

                %sum(A_hat) >= 1;
        cvx_end
        A_hat = A_hat/max(max(A_hat));
        figure(1)
        imagesc(A_hat)
        colorbar()
        T_hat(:,:,t) = A_hat;
    end
    out.np = N^2*T;
    out.T_hat = reshape(T_hat,[N,N*T]);
end