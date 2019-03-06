Extended Kalman filtering with low-rank Tensor Networks for MIMO Volterra system identification (Matlab&copy;/Octave&copy;)
-------------------------------------------------------------------------------------------------------------------------------


1. Functions
------------

* [thetaC_hat,sigma_thetaC] = TNEKF(thetaC_hat,sigma_thetaC,v1,y,M,fgf,sigma_e,tol,opt,varargin)

Computes the mean MPSs m and covariance MPOs for the identification of Volterra systems.

* c=addTN(a,b)

Adds two Tensor Networks *a* and *b* together.

* C = tkron(A,B)

Returns the Kronecker product of two input tensors A,B.

* b=contract(a)

Sums the Tensor Network *a* over all its auxiliary indices to obtain the underlying tensor.

* y=mkron(varargin)

Returns a Kronecker product of multiple input tensors.

* c=contractab(a,b,k)

Contracts the cores of Tensor Network a along mode k(1) with cores of Tensor Network b along mode k(2).

* [TN,err] = DMRGround(oTN,eps,varargin)

Returns an approximation of the Tensor Network *TN* using DMRG rounding.

* a=roundTN(a,tol,varargin)

Returns an approximation of the Tensor Network *a* such that the approximation has a relative error *tol*.


* example.m

Small demo that illustrates how to use the TNEKF for system identifcation of Volterra systems.


2. Reference
------------

"Extended Kalman filtering with low-rank Tensor Networks for MIMO Volterra system identification"


Authors: Kim Batselier, Ching-Yun Ko, Ngai Wong
