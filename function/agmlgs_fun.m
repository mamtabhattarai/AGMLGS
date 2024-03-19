
function [Pre_labels_train ,Pre_labels_test,time,obj] = agmlgs_fun(X1,Y, X2,Y2, par)

beta=par.beta;

 apha=par.alpha;
Q=par.K;
[N2,M1]=size(Y);% M1 classes
% N2 samples. D dim
[N2,D]=size(X1);%training
[N1,D]=size(X2);%testing

SI = exp(-squareform(0.5*pdist(X1)).^2);
delta = diag(sum(SI, 2));
delta = (delta)^(-1/2);
Ls = diag(sum(SI, 2)) - SI;
Ls = delta*Ls*delta;
%

U=zeros(N2,N2);
for i=1:N2
    if(Y(i,1)~=0)
        U(i,i)=100;
    end
end


W=randn(D,M1);
Dw=zeros(D,D);
for ii=1:D
    Dw(ii,ii)=1/(2*norm(W(ii,:)));
end


I=eye(N2,N2);

tic;
for i=1:10
A=(2*I + *Ls + *U)^-1;

B=(X1*W+*U*Y);


F=A*B;%Update F





%  ======Update S fix W======
% sig = 0.7 ;
sig=(1/N2^2)*(norm(X1)^2);
RI = (eye(N2)-evalkernel(X1,X1,'rbf',sig))+eye(N2)*10^100;% acc to eqn 5 diagonal are infinity
YY=X1*W;





% =======update W ============

M=I-S;
II=eye(M1,M1);
W=((X1'*X1+*Dw+beta*X1'*X1+*X1'*M*M'*X1)^-1)*X1'*F;
 time=toc;

Pre_labels_train = X1*W;
Pre_labels_train = sign(Pre_labels_train);
Pre_labels_test = X2*W;
Pre_labels_test = sign(Pre_labels_test);
temp1=(norm((X1*W-F),'FRO'))^2;
 temp3=alpha*sum(sqrt(sum(W.^2, 2)));
 temp4=trace(F'*Ls*F);
temp5=trace((F-Y)'*U*(F-Y));
 temp2=(norm(W'*(X1'-X1'*S)))^2;
 temp6=beta*(norm(RI.*S))^2;
 temp8=trace(W'*X1'*X1*W-II);
obj(i)=temp1+temp2+temp3+temp4+temp5+temp6+temp8;
end
row_sums_squared = sum(W.^2, 2);

% Sort the rows of W based on the sum of squares
[sorted_sums_squared, sorted_indices] = sort(row_sums_squared, 'descend');

% Select the top 100 rows
top_100_indices = sorted_indices(1:Q);

% Introduce zeros in the respective indices for the rest of the rows
W_rest = W;
W_rest(setdiff(1:D, top_100_indices), :) = 0;

time=toc;
Pre_labels_train = X1*W;
Pre_labels_train = sign(Pre_labels_train);
Pre_labels_test = X2*W;
Pre_labels_test = sign(Pre_labels_test);
%time=toc;
end


