function out = decomp(X0,prms)
    [N,~,T] = size(X0);
    X = reshape(X0,[N,N*T]);
    C = zeros(N,N*T);
    max_iters = prms.max_iters;
    alpha = prms.alpha;
    R = prms.R;
    for i = 1:max_iters
        %%% Estimate A and C
        cvx_begin quiet
            variable A(N,N)
            expressions At(N,N*R) 
            At = repmat(A,1,T);
            minimize(square_pos(norm(X-At-C,'fro')) + alpha*norm(C,1))
            subject to 
                diag(A) <= 1e-6;
                A == A';
                A>=0;
                %A <= 1;
        cvx_end
        C_est = tensor(reshape(X-At,[N,N,T]));
        C = cp_als(C_est,R,'printitn',0);
        C = reshape(double(C),[N,N*T]);
        error = norm(X-At-C,"fro");
        %disp(error)
    end
    out.T_hat = At+C;
    out.At = At;
    out.A = A;
    out.C = C;
    out.np = N^2+(2*N+T+1)*R; 
end