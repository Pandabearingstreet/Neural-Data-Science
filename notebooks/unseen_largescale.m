function [histx,x] = unseen_largescale(f,cutoff)
% Input: fingerprint f, where f(i,1) is the number of domain elements that 
% have been observed exactly f(i,2) times 

% Output: approximation of 'histogram' of true distribution.  Specifically,
% histx(i) represents the number of domain elements that occur with
% probability x(i).   Thus sum_i x(i)*histx(i) = 1, as distributions have
% total probability mass 1.   
%
% An approximation of the entropy of the true distribution can be computed
% as:    Entropy = (-1)*sum(histx.*x.*log(x))


    cutoff = 500;

k=sum(f(:,1).*f(:,2));  %total sample size


%%%%%%% algorithm parameters %%%%%%%%%%%
gridFactor = 1.05;     % the grid of probabilities will be geometric, with this ratio.
% setting this smaller may slightly increase accuracy, at the cost of speed 
alpha = .5; %the allowable discrepancy between the returned solution and the "best" (overfit).
% 0.5 worked well in all examples we tried, though the results were nearly indistinguishable 
% for any alpha between 0.25 and 1.  Decreasing alpha increases the chances of overfitting. 
xLPmin = 1/(k*min(10000,max(10,k))); 

min_i=min(find(f>0));
if min_i > 1
    xLPmin = min_i/k;
end% minimum allowable probability. 
% a more aggressive bound like 1/k^1.5 would make the LP slightly faster,
% though at the cost of accuracy
maxLPIters = 1000;    % the 'MaxIter' parameter for Matlab's 'linprog' LP solver.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Split the fingerprint into the 'dense' portion for which we 
% solve an LP to yield the corresponding histogram, and 'sparse' 
% portion for which we simply use the empirical histogram
ind=find(f(:,2)>cutoff);
histx=f(ind,1);
x=f(ind,2)/k;
exp_f_emp = zeros(1,cutoff);
if min(size(ind))>0
    for i=1:max(size(ind))
        exp_f_emp = exp_f_emp+f(ind(i),1)*binopdf(1:cutoff,k,f(ind(i),2)/k);
    end
end

ind2=find(f(:,2)<=cutoff);
E_f = zeros(1,cutoff);
for i=1:max(size(ind2))
    E_f(f(ind2(i),2))=E_f(f(ind2(i),2))+f(ind2(i),1);
end
E_f = E_f-exp_f_emp;  


% Set up the first LP
LPmass = 1 - sum(x.*histx); %amount of probability mass in the LP region

E_f;
szLPf=max(size(E_f));

xLPmax = cutoff/k;
xLP=xLPmin*gridFactor.^(0:ceil(log(xLPmax/xLPmin)/log(gridFactor)));
szLPx=max(size(xLP));

objf=zeros(szLPx+2*szLPf,1);
objf(szLPx+1:2:end)=1./(sqrt(max(E_f+1,1)));  % discrepancy in ith fingerprint expectation
objf(szLPx+2:2:end)=1./(sqrt(max(E_f+1,1)));  % weighted by 1/sqrt(f(i) + 1)

A = zeros(2*szLPf,szLPx+2*szLPf);
b=zeros(2*szLPf,1);
for i=1:szLPf
    A(2*i-1,1:szLPx)=binopdf(i,k,xLP);
    A(2*i,1:szLPx)=(-1)*A(2*i-1,1:szLPx);
    A(2*i-1,szLPx+2*i-1)=-1;
    A(2*i,szLPx+2*i)=-1;
    b(2*i-1)=E_f(i);
    b(2*i)=-E_f(i);
end

Aeq = zeros(1,szLPx+2*szLPf);
Aeq(1:szLPx)=xLP;
beq = LPmass;


options = optimset('MaxIter', maxLPIters,'Display','off');
for i=1:szLPx
    A(:,i)=A(:,i)/xLP(i);   %rescaling for better conditioning
    Aeq(i)=Aeq(i)/xLP(i);
end
[sol, fval, exitflag, output] = linprog(objf, A, b, Aeq, beq, zeros(szLPx+2*szLPf,1), Inf*ones(szLPx+2*szLPf,1),[], options);
if exitflag==0
        'maximum number of iterations reached--try increasing maxLPIters'
end
if exitflag<0
    'LP1 solution was not found, still solving LP2 anyway...'
    exitflag
end

% Solve the 2nd LP, which minimizes support size subject to incurring at most
% alpha worse objective function value (of the objective function in the 
% previous LP). 
objf2=0*objf;
objf2(1:szLPx) = 1;
A2=[A;objf'];         % ensure at most alpha worse obj value
b2=[b; fval+alpha];   % than solution of previous LP
for i=1:szLPx
    objf2(i)=objf2(i)/xLP(i);   %rescaling for better conditioning
end
[sol2, fval2, exitflag2, output] = linprog(objf2, A2, b2, Aeq, beq, zeros(szLPx+2*szLPf,1), Inf*ones(szLPx+2*szLPf,1),[], options);

if not(exitflag2==1)
    'LP2 solution was not found'
    exitflag2
end


%append LP solution to empirical portion of histogram
sol2=sol2(1:szLPx)./xLP';   %removing the scaling
x=[x;xLP'];
histx=[histx;sol2];
[x,ind]=sort(x);
histx=histx(ind);
ind = find(histx>0);
x=x(ind);
histx=histx(ind);