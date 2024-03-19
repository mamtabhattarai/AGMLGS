function [w,b,gini_indx,time]=svmclassifier(D,Y,ker,C,rr,kerpara) 
%%rr represents the portion of the training data for buiding up the
%%classifier
[n1,n2]=size(D);
m=floor(rr*n1);
D1=D(1:m,:);
m2=length(find(Y==-1));
m1=length(find(Y==1));

A1=[];
B1=[];
A1=D(find(Y==1),:);
B1=D(find(Y==-1),:);

w=[];
b=0;
switch lower(ker)
case 'linear'
    A=A1;
    B=B1;
    D=D1;
case 'rbf'
    Z = zeros(n1,n1);
    for i=1:n1
        for j=1:n1
            Z(i,j) = svkernel(ker,D1(i,:),D1(j,:),kerpara);
        end
    end
    clear D
    D=Z;
    clear Z
    [n1,n2]=size(D);

case 'poly'
    Z = zeros(n1,n1);
    for i=1:n1
        for j=1:n1
            Z(i,j) = svkernel(ker,D1(i,:),D1(j,:),kerpara);
        end
    end
    clear D
    D=Z;
    clear Z
    [n1,n2]=size(D);
    
end

H=(diag(Y)*D)*(diag(Y)*D)';

H=(H+H')/2; %for making it hessian. 

f=-( ones(m1+m2,1) ) ;

Aeq = [Y']; beq = [0];
AA = []; bb = [];

lb=zeros(m1+m2,1);
ub=C*ones(m1+m2,1);

 x0 = zeros(m1+m2,1); % starting point

options = optimoptions('quadprog','display','off');

tic;
[alpha, fval, exitflag ] = quadprog( H, f, AA, bb, Aeq, beq, lb, ub, x0, options);
time=toc;

% if isempty(X)
%     continue;
% end

w = (alpha'*(diag(Y)*D))';
b = sum(Y - D*w)/(m1+m2);
wn = norm(w);
w=w/wn;
b=b/wn;
nsv=length(find(alpha>0.001))/length(alpha);

gini_indx = 0; %modified 02/07/2022
% [gini_indx] =calc_gini_indx(alpha);
return;