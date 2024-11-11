%% Description

% K : number of iterations between A and C
% R : number of components recovered
% Ni2: number of rounds of iterations over all A_r s
% gamma: hyperparameter for sparsity
% beta: hyperparameter for non-overlapping support
% delta: hyperparameter for smoothness
% smooth: smoothness hyperparameter for the term ||Dc||_2^2

addpath(genpath('..\real_data\molene'))
addpath(genpath('..\utils'))
clear all
T=10;sigma=0.1;K=3;N=28;
load('..\real_data\molene\Molene_experiment.mat')
[X,Y]=generate_adj_matrices(T,0.1,3);

% data set should have Adj Tensor (X) and Data Tensor (Y)

% ZT generation from Y

ZT = compute_W(Y);

% extract X_0 (in the draft) from X



XX=[];
for t=1:T
XX=[XX,vec(X(:,:,t))];
end

% Hyperparameters

R=3;K=20;Ni2=10;

% Calling the algorithm

gamma=1e-1;
beta=0;%1e-1;
delta=1e-3;
smooth=1e-3;
% A: output modes in vector form
% C: temporal matrix
% X_rec: reconstructed tensor

A_init=rand(N^2,R);C_init=rand(T,R);%initialize at random
[A,C,X_rec,~] = update_Adj(XX,A_init,C_init,ZT,K,R,Ni2,gamma,beta,delta,smooth);

%You can reshape the A matrix. C is obtained in T by R size
nerr = norm(X_rec(:)-XX(:),2)^2/norm(XX(:),2)^2;

figure(1)
subplot(211)
imagesc(reshape(XX,[N,N*T]))
colorbar()
subplot(212)
imagesc(reshape(X_rec,[N,N*T]))
colorbar()
title(num2str(nerr))


%%
I_R = eye(R);
Z = reshape(ZT,[T,N^2]);
m = N^2;n = R;
Ac = reshape(1:m*n, m, n);
v = reshape(Ac', 1, []);
P = eye(m*n);
P = P(v,:);
K_CA = P;
f_CA = kron(A',C)*K_CA + kron(I_R,C*A'-XX'+delta*Z);


