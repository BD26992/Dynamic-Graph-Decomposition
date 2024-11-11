function X = temporal_difference_matrix(T)
X=zeros(T-1,T);
for t=1:T-1
    X(t,:)=[zeros(1,t-1),1,-1,zeros(1,T-1-t)];
end
end