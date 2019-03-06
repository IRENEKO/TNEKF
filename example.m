close all; clear all

% flag = 1 implis generating a new random system; flag = 0 implies using
% previously saved models
flag = 1; 

% opt = 1 points to the use of normal (or truncated) TN rounding; opt = 2
% points to the use of DMRG rounding
opt = 1;

maxr = 1;


if flag == 1
    m = 1;                    % input dimension
    M = 4;                    % memory
    m1 = m*M + 1;             % extended input dimension
    l = 2;                    % output dimension
    d = 3;                    % polynomial degree
    tol = 1e-10;              % TN rounding tolerance
    fgf = 1;                  % forgetting factor, default is 1
    
    % generate the system and create IO-data
    Ltrain = 2000;           % the length of training sequence 
    Lval = 100;               %the length of validation sequence
    L = Ltrain + Lval;        % the length of the generated sequence 
    fac = randn(l,1);
    C = fac*randn(1,m1^d);
    C = C(:);
    u = randn(L,m);           % from 0 to L-1
    y = zeros(L,l);           % from 1 to L
    e = 1e-4*randn(1,l);
    sigma_e = 1e-8*eye(l);
    for i=M:L
        v1 = [1,reshape(u(i:-1:i-M+1,:)',[1,m*M])]'; % [1,m1]
        y(i,:) = (kron(mkron(v1,d)',eye(l))*C)' + e;
    end
    clear v1

elseif flag == 0
    load mdl.mat
end


% initialization
% for updating thetaC
TNd = d+1;                    % number of cores in a tensor network
thetaC_hat.n = [ones(TNd,1),[l;m1*ones(d,1)],ones(TNd,1)];  
sigma_thetaC.n = [[1;1*ones(TNd-1,1)],[l;m1*ones(d,1)],[l;m1*ones(d,1)],[1*ones(TNd-1,1);1]];
for i = 1:TNd
    thetaC_hat.core{i} = zeros(thetaC_hat.n(i,:));
    if i == 1
        sigma_thetaC.core{i} = reshape([1e1,6;6,1e1],sigma_thetaC.n(i,:));
    else
        sigma_thetaC.core{i} = reshape(eye(m1),sigma_thetaC.n(i,:));
    end
end


% param. estimation
for k = M+1:Ltrain+1
    v1.n = [ones(TNd,1),[l;ones(d,1)],[l;m1*ones(d,1)],ones(TNd,1)];
    v1.core{1} = reshape(eye(l),v1.n(1,:));
    for i = 2:TNd
        v1.core{i} = reshape([1,reshape(u(k-1:-1:k-M,:)',[1,m*M])],v1.n(i,:));
    end
    tic;
    [thetaC_hat,sigma_thetaC] = TNEKF(thetaC_hat,sigma_thetaC,v1,y(k-1,:),M,fgf,sigma_e,tol,opt,maxr);
    t(k-M) = toc;
    for i = Ltrain+1:L
        for p = 2:TNd
            v1.core{p} = reshape([1,reshape(u(i:-1:i-M+1,:)',[1,m*M])],v1.n(p,:));
        end
        ysim(i-Ltrain,:) = contract(contractab(v1,thetaC_hat,[3,2]))';
    end
    Cdiff(k-M) = norm(reshape(y(Ltrain+1:L,:)-ysim,[numel(ysim),1]))/norm(reshape(y(Ltrain+1:L,:),[numel(ysim),1]));
end


figure;
semilogy(abs(Cdiff));xlabel('Iterations');hold on;
if opt == 1
    fprintf('roundTN TNEKF costs: %4.2f seconds\n',sum(t))
elseif opt == 2
    fprintf('DMRGround TNEKF costs: %4.2f seconds\n',sum(t))
end


