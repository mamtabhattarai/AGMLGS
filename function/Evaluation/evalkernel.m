function k = evalkernel(u,v,kernel,kernelparam)

%function K = evalkernel(samples1, samples2, kernel, kernelparam)
%  Evaluate kernel function
% Usage:
% Input:
%   samples1: n1 x d matrix, each row is a sample.
%   samples2: n2 x d matrix, each row is a sample.
%   kernel: kernel type
%   kernelparam: kernel parameter
% Outputs
%   K: n1 x n2 kernel matrix

p1=kernelparam;
%global p1 p2;
    %kernel='linear';
   % kernelparam = p1;
%     p1 = kernelparam.p1;
%     p2 = kernelparam.p2;
if (size(u,2)~=size(v,2))
    error('sample1 and sample2 differ in dimensionality!!');
end
[n1, dim] = size(u);
[n2, dim] = size(v);

switch kernel    
case 'linear'
    k = u*v';%/dim;
case 'poly'
    k = (1 + u*v'/dim).^p1;
    %K = (1 + u*v').^kernelparam;
case 'rbf'
    a = sum(u.*u,2);
    b = sum(v.*v,2);
    dist2 = a*ones(1,n2);
    dist2 = dist2 + ones(n1,1)*b';
    dist2 = dist2 - 2*u*v';
    k = exp(-dist2/kernelparam);
    %k = exp(dist2*(-kernelparam));
    %k = exp(-dist2/(2*(p1^2)));
  case 'rbfp'
      a = sum(u.*u,2);
    b = sum(v.*v,2);
    dist2 = a*ones(1,n2);
    dist2 = dist2 + ones(n1,1)*b';
    dist2 = dist2 - 2*u*v';
    %K = exp(-kernelparam*dist2);
    %k = exp(-dist2/(2*(p1^2)));
    k = exp(dist2./p1);
    case 'rbf2'
    a = sum(u.*u,2);
    b = sum(v.*v,2);
    dist2 = a*ones(1,n2);
    dist2 = dist2 + ones(n1,1)*b';
    dist2 = dist2 - 2*u*v';
    %K = exp(-kernelparam*dist2);
    %k = exp(-dist2/(2*(p1^2)));
    k = exp(-dist2./p1);
            
    case 'erbf'
        k = exp(-sqrt((u-v)*(u-v)')/(2*p1^2));
    case 'sigmoid'
        k = tanh(p1*u*v'/length(u) + p2);
    case 'fourier'
        z = sin(p1 + 1/2)*2*ones(length(u),1);
        i = find(u-v);
        z(i) = sin(p1 + 1/2)*(u(i)-v(i))./sin((u(i)-v(i))/2);
        k = prod(z);
    case 'spline'
        z = 1 + u.*v + (1/2)*u.*v.*min(u,v) - (1/6)*(min(u,v)).^3;
        k = prod(z);
    case 'bspline'
        z = 0;
        for r = 0: 2*(p1+1)
            z = z + (-1)^r*binomial(2*(p1+1),r)*(max(0,u-v + p1+1 - r)).^(2*p1 + 1);
        end
        k = prod(z);
    case 'anovaspline1'
        z = 1 + u.*v + u.*v.*min(u,v) - ((u+v)/2).*(min(u,v)).^2 + (1/3)*(min(u,v)).^3;
        k = prod(z);
    case 'anovaspline2'
        z = 1 + u.*v + (u.*v).^2 + (u.*v).^2.*min(u,v) - u.*v.*(u+v).*(min(u,v)).^2 + (1/3)*(u.^2 + 4*u.*v + v.^2).*(min(u,v)).^3 - (1/2)*(u+v).*(min(u,v)).^4 + (1/5)*(min(u,v)).^5;
        k = prod(z);
    case 'anovaspline3'
        z = 1 + u.*v + (u.*v).^2 + (u.*v).^3 + (u.*v).^3.*min(u,v) - (3/2)*(u.*v).^2.*(u+v).*(min(u,v)).^2 + u.*v.*(u.^2 + 3*u.*v + v.^2).*(min(u,v)).^3 - (1/4)*(u.^3 + 9*u.^2.*v + 9*u.*v.^2 + v.^3).*(min(u,v)).^4 + (3/5)*(u.^2 + 3*u.*v + v.^2).*(min(u,v)).^5 - (1/2)*(u+v).*(min(u,v)).^6 + (1/7)*(min(u,v)).^7;
        k = prod(z);
    case 'anovabspline'
        z = 0;
        for r = 0: 2*(p1+1)
            z = z + (-1)^r*binomial(2*(p1+1),r)*(max(0,u-v + p1+1 - r)).^(2*p1 + 1);
        end
        k = prod(1 + z);
    
otherwise
    error('Unknown kernel function');



end 

