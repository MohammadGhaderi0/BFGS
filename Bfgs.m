
clear all;
% min f(x) = exp(x(1)+ 3 * x(2)-0.1) + exp(x(1)-3*x(2)-0.1) + exp(-x(1)-0.1)
alpha = 0.1;
beta = 0.5;
D = eye(2);
x = [1,1];
for i = 1:100
    fx = exp(x(1)+ 3*x(2)-0.1) + exp(x(1)-3*x(2)-0.1)+ exp(-x(1)-0.1);
    if i > 1; 
    gxold = gx;
    end
    gx = [exp(x(1)+ 3*x(2)-0.1) + exp(x(1)-3*x(2)-0.1)- exp(-x(1)-0.1); 3* exp(x(1)+3*x(2)-0.1)-3*exp(x(1)-3*x(2)-0.1)];
    if i > 1;
        q = gx -gxold;
    end
    if i > 1;
        tau = q'*D*q;
        v = p/(p'*q)-D*q/tau;
        D = D -(D*(q*q')*D)/(q'*D*q)+p*p'/(p'*q)+tau*v*v';
    end
    t = 1;
    d = -D*gx;
    xp = x+t*d;
    fxp = exp(x(1)+ 3*x(2)-0.1) + exp(x(1)-3*x(2)-0.1)+ exp(-x(1)-0.1);
    while fxp> fx+ alpha*t*gx'*d;
        t = beta*t;
        xp = x+t*d;
        fxp = exp(xp(1)+ 3*xp(2)-0.1) + exp(xp(1)-3*xp(2)-0.1)+ exp(-xp(1)-0.1);
    end
    x = x+t*d;
    p = t*d;
end
