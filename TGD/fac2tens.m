function X_rec = fac2tens(A,C)
%% BTD reconstruction
%% Each column of A is the vectorized version of the rth adjacency matrix
[T,R]=size(C);N=sqrt(size(A,1));
AA=A*C';
X_rec=zeros(N,N,T);
for t=1:T
    X_rec(:,:,t)=reshape(AA(:,t),N,N);
end
end