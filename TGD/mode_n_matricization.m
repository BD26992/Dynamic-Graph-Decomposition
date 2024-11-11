function X = mode_n_matricization(X,n)
% MODE_K_MATRICIZATION takes a tensor X as input and a mode n such
% that n<ndims(X) and returns the mode-n matricization of X.
% INPUTS tensor X, mode n.
% OUPUT mode-n matricization of X.
dims=size(X);
N_mode=length(size(X));
p=[n,setdiff((1:N_mode),n)];
A=permute(X,p);
X=reshape(A,dims(n),prod(dims)/dims(n));
end