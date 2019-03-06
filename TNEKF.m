function [thetaC_hat,sigma_thetaC] = TNEKF(thetaC_hat,sigma_thetaC,v1,y,M,fgf,sigma_e,tol,opt,varargin)
% [thetaC_hat,sigma_thetaC] = TNEKF(thetaC_hat,sigma_thetaC,v1,y,M,fgf,sigma_e,tol,opt,varargin)
% -------------------------------------------
% y_k = [(1,u_{k}^T,..., u_{k-M+1}^T)^d kron Il] * thetaC + w_{k}
%
% thetaC_hat         =	Tensor Networks for matrix of mean vectors thetaC 
%
% sigma_thetaC_hat   =	Tensor Networks for matrix of covariance matrices
%                       sigma_thetaC
%
% v1                 =	Tensor Network, containing input values for MIMO
%				        MIMO Volterra system
%
% y                  =	matrix, each row corresponds with l measured output
%
% M                  =  Memory
%
% fgf                =  forgetting factor
%
% sigma_e            =	matrix, variances of the Gaussian measurement noise
%
% tol	             =  scalar, relative approximation error in the TN-rounding
%
% opt                =  1 points to TN rounding, 2 points to DMRGrounding
%
%
% Reference
% ---------
%
% Extended Kalman filtering with low-rank Tensor Networks for MIMO Volterra system identification
%
% ---------
%
% 03/2019, Ching-Yun Ko

TNd = size(thetaC_hat.n,1);
d = TNd-1;
m1 = thetaC_hat.n(2,2);
m = (m1-1)/M;
l = size(y,2);

sigma_thetaC.core{1} = sigma_thetaC.core{1}./fgf;
y_hat = contract(contractab(v1,thetaC_hat,[3,2]))'; 
C_thetaC = v1;
Linv = reshape(contract(contractab(contractab(C_thetaC,sigma_thetaC,[3,2]),C_thetaC,[3,3])),[l,l]) + sigma_e;
L_thetaC = contractab(sigma_thetaC,C_thetaC,[3,3]);
L_thetaC = roundTN(L_thetaC,tol);
sz = L_thetaC.n(1,:); 
L_thetaC.core{1} = reshape(permute(L_thetaC.core{1},[1,2,4,3]),[sz(1)*sz(2)*sz(4),sz(3)])*inv(Linv); % true L_thetaC.core{1} = permute(reshape(L_thetaC.core{1},[sz(1),sz(2),sz(4),sz(3)]),[1,2,4,3]);
Ly = L_thetaC;
Ly.core{1} = reshape(L_thetaC.core{1}*(y-y_hat)',[sz(1),sz(2),1,sz(4)]); % don't need the permutation between the third and the fourth indices
Ly.n(1,3) = 1;
L_thetaC.core{1} = permute(reshape(L_thetaC.core{1},[sz(1),sz(2),sz(4),sz(3)]),[1,2,4,3]);
thetaC_hat = addTN(thetaC_hat,Ly);
clear sz Ly
thetaC_hat = roundTN(thetaC_hat,tol);
LSL.n = [ones(TNd,1),[l;m1*ones(d,1)]*ones(1,2),ones(TNd,1)];
sz = L_thetaC.n(1,2:end);
LSL.core{1} = reshape(permute(L_thetaC.core{1},[1,2,4,3]),[sz(1)*sz(3),sz(2)])*Linv*reshape(L_thetaC.core{1},[sz(1),sz(2)*sz(3)]);
LSL.core{1} = reshape(permute(reshape(LSL.core{1},[sz(1),sz(3),sz(2),sz(3)]),[1,3,2,4]),[sz(1),sz(2),sz(3)^2]);
LSL.n(1,end) = L_thetaC.n(1,end)^2;
LSL.n(2,1) = L_thetaC.n(1,end)^2;
for i=2:TNd
    sz = L_thetaC.n(i,:);
    LSL.core{i} = reshape(permute(L_thetaC.core{i},[1,2,4,3]),[sz(1)*sz(2)*sz(4),1])*reshape(permute(L_thetaC.core{i},[1,2,4,3]),[sz(1)*sz(2)*sz(4),1])';
    LSL.core{i} = reshape(permute(reshape(LSL.core{i},[sz(1),sz(2),sz(4),sz(1),sz(2),sz(4)]),[1,4,2,5,3,6]),[sz(1)^2,sz(2),sz(2),sz(4)^2]);
    LSL.n(i,end) = sz(4)^2;
    if i~= TNd
        LSL.n(i+1,1) = sz(4)^2;
    end
end
LSL = roundTN(LSL,tol);
LSL.core{1} = -LSL.core{1};
sigma_thetaC = addTN(sigma_thetaC,LSL);
clear L_thetaC C_thetaC LSL Linv
if opt == 1
    if isempty(varargin)
        sigma_thetaC = roundTN(sigma_thetaC,tol);
    else
        sigma_thetaC = roundTN(sigma_thetaC,tol,varargin{1});
    end
elseif opt == 2
    if isempty(varargin)
        sigma_thetaC = DMRGround(sigma_thetaC,tol);
    else 
        sigma_thetaC = DMRGround(sigma_thetaC,tol,varargin{1});
    end
end
end
