function out = smooth_topoID(W,prms)
    a = prms.a;
    b = prms.b;
    [N,~,T] = size(W);
    W = W/(N*prms.M);
    T_hat = zeros(N,N,T);
    p = struct('verbosity',0);
    for t = 1:T
        [A_hat, ~] = gsp_learn_graph_log_degrees(W(:,:,t), a,b,p);
        %[A_hat, ~] = gsp_learn_graph_l2_degrees(W(:,:,t), a,p);
        T_hat(:,:,t) = A_hat/max(max(A_hat));
        figure(2)
        imagesc(A_hat)
    end

    out.np = N^2*T;
    out.T_hat = reshape(T_hat,[N,N*T]);
end
