function [A_out,La1] = update_A(A,C,XX,WW,Ma,La1,prms)
    [T,N,R] = size(La1);
    Onn = ones(N);
    On = ones(N,1);
    In = eye(N);
    Ont = ones(N,T);
    prms.delta = 100;
    alpha = prms.alpha;
    beta = prms.beta;
    eta = prms.eta;
    delta= prms.delta;
    gamma = prms.gamma;
    cita = prms.cita;
    la1 = 10;%prms.la1;
    K = 20;

    for r = 1:R % update each Ar
        La = La1(:,:,r);
        Ar = reshape(A(:,r),[N,N]);
        % Selecting Ar and Cr
        m=ones(R,1);m(r)=0;Cr = C(r,:)';m = logical(m);
        v1 = alpha*sum((Cr.^2).*(sum(Ma)'));
        V2 = zeros(N);
        for ti = 1:T
            Ap = zeros(N);
            for p = 1:R
                if p ~= r
                    Ap = Ap + C(p,ti)*reshape(A(:,p),[N,N]);
                end
            end
            V2 = V2 + C(r,ti)*sum(Ma(:,ti))*Ap;
        end
        V3 = reshape(sum(XX*diag(Cr.*sum(Ma)'),2),[N,N]);
        Qr = V3-V2;
        %S = A(:,m)*C(m,:); %check that this is correct
        %Yr = XX-S;B = Yr'.*Ma';
        %qr = alpha*diag(Ma*diag(Cr)*B);
        %Qr = reshape(qr,[N,N])';
        Fr = reshape(A*m,[N,N]);
        Xir = reshape(WW*Cr,[N,N]);

        Phir = kron(C(r,:)',On')'; %check if this is correct
        Gar = zeros(size(Phir));
        for i = 1:R
            if i ~= r
                %Ai = reshape(A(:,i),[N,N]);
                %Ci = C(i,:);
                %Gar = Gar + Ai*kron(Ci',On)';
                Gar = Gar + reshape(A(:,i),[N,N])*kron(C(i,:)',On')';
            end
        end
        Gar = Gar - cita*Ont;
        P = rand(size(Gar));
        for k = 1:K     % several iteration for upadate each Ar
            %G is the matrix including all the derivatives which do not depend on Ar
            G = alpha*Qr - gamma*Onn - delta*Xir - beta*Fr + la1*(P - 1/la1*La' - Gar)*Phir'; 
            %H is the matrix including all the derivatives which depend on Ar
            H = (v1 + eta)*In + la1*Phir*Phir';
            %print_matrices(Qr,delta*Xir,beta*Fr,Phir*La',la1*2*Gar*Phir');
            Ar = G*inv(H);
            %Ar = Ar - 1e-5*(Ar*H - G);
            
            %projections
            Ar = (Ar+Ar')/2;
            Ar(logical(In)) = 0;
            Ar(Ar<0) = 0;
            %Ar = Ar/max(max(Ar));

            %update P
            P = 1/la1*La' + Ar*Phir + Gar;
            P(P<0) = 0;
            %update La
            La = La + la1*(Ar*Phir + Gar - P)';
            % figure(4)
            % subplot(221)
            % imagesc(Ar)
            % title('Ar')
            % colorbar()
            % subplot(222)
            % imagesc(P)
            % title('P')
            % colorbar()
            % subplot(223)
            % imagesc(La)
            % title('La')
            % colorbar()
        end


        %updating A
        A(:,r) = vec(Ar);
        La1(:,:,R) = La;
    end
    A_out = A;
end