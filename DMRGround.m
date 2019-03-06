function [TN,err] = DMRGround(oTN,eps,varargin)
% function [TN,err] = DMRGround(oTN,eps,varargin)
% -------------------------------------------
% Returns an approximation of the Tensor Network with ideally lower ranks
%
% oTN          =   Tensor Network,
%
% eps 		   =   scalar, a given truncation tolerance
%
% Reference
% ---------
%
% Extended Kalman filtering with low-rank Tensor Networks for MIMO Volterra system identification
%
% ---------
%
% 03/2019, Ching-Yun Ko


d = size(oTN.n,1);  % d should be at least 3
legs = size(oTN.n,2)-2;  % TT when legs = 1, MPO when legs = 2
Itr = 1;


% initialization
TN.n = ones(size(oTN.n));
TN.n(:,2:legs+1) = oTN.n(:,2:legs+1);
for i = d:-1:3
    sz = TN.n(i,:);
    TN.core{i} = reshape(orth(ones(sz(1),prod(sz(2:legs+2)))')',sz);
end

osz = oTN.n(d,:);
sz = TN.n(d,:);
vp{d} = reshape(oTN.core{d},[osz(1),prod(osz(2:end))])*reshape(permute(TN.core{d},[2:legs+2,1]),[prod(sz(2:legs+2)),sz(1)]); % [R_d r_d] 
for i = d-1:-1:3
    osz = oTN.n(i,:);
    sz = TN.n(i,:);
    vp{i} = reshape(oTN.core{i},[prod(osz(1:end-1)),osz(end)])*vp{i+1}; % [R_in_in_i r_{i+1}]
    vp{i} = reshape(vp{i},[osz(1),prod(osz(2:legs+1))*1])*reshape(permute(TN.core{i},[2:legs+2,1]),[prod(sz(2:legs+2)),sz(1)]); % [R_i r_i]
end

% updates
for itr = 1:Itr
for i=1:d-2
    osz = oTN.n(i,:);
    sz = TN.n(i,:);
    if i == 1
        supercore = reshape(reshape(oTN.core{i+1},[prod(oTN.n(i+1,1:legs+1)),oTN.n(i+1,legs+2)])*vp{3},[oTN.n(i+1,1),prod(oTN.n(i+1,2:legs+1))*size(vp{3},2)]);
        supercore = reshape(reshape(oTN.core{i},[prod(osz(1:legs+1)),osz(legs+2)])*supercore,[prod(TN.n(i,1:legs+1)),prod(TN.n(i+1,2:legs+2))]);
        if isempty(varargin)
            [U,S,V] = svd(supercore,'econ');
        else
            if min(size(supercore))<varargin{1}
                [U,S,V] = svd(supercore,'econ');
            else
                [U,S,V]=svds(supercore,varargin{1});
            end
        end  
        delta = S(1,1)*eps*max(size(supercore));
        s = diag(S);
        rr = sum(s>delta);      
        TN.core{1} = reshape(U(:,1:rr),[TN.n(i,1:legs+1),rr]);
        TN.core{2} = reshape(S(1:rr,1:rr)*V(:,1:rr)',[rr,TN.n(i+1,2:legs+2)]);
        TN.n(1,end) = rr;
        TN.n(2,1) = rr;
        sz(end) = rr;
        vm{1} = reshape(permute(TN.core{1},[legs+2,1:legs+1]),[sz(legs+2),prod(sz(1:legs+1))])*reshape(oTN.core{1},[prod(osz(1:legs+1)),osz(legs+2)]); % [r_2 R_2]  
    else
        supercore = reshape(reshape(oTN.core{i+1},[prod(oTN.n(i+1,1:legs+1)),oTN.n(i+1,legs+2)])*vp{i+2},[oTN.n(i+1,1),prod(oTN.n(i+1,2:legs+1))*size(vp{i+2},2)]);
        supercore = reshape(reshape(oTN.core{i},[prod(osz(1:legs+1)),osz(legs+2)])*supercore,[osz(1),prod(TN.n(i,2:legs+1))*prod(TN.n(i+1,2:legs+2))]);
        supercore = reshape(vm{i-1}*supercore,[prod(TN.n(i,1:legs+1)),prod(TN.n(i+1,2:legs+2))]);
        if isempty(varargin)
            [U,S,V] = svd(supercore,'econ');
        else
            if min(size(supercore))<varargin{1}
                [U,S,V] = svd(supercore,'econ');
            else
                [U,S,V]=svds(supercore,varargin{1});
            end
        end  
        delta = S(1,1)*eps*max(size(supercore));
        s = diag(S);
        rr = sum(s>delta);      
        TN.core{i} = reshape(U(:,1:rr),[TN.n(i,1:legs+1),rr]);
        TN.core{i+1} = reshape(S(1:rr,1:rr)*V(:,1:rr)',[rr,TN.n(i+1,2:legs+2)]);
        TN.n(i,end) = rr;
        TN.n(i+1,1) = rr;
        sz(end) = rr;
        vm{i} = reshape(vm{i-1}*reshape(oTN.core{i},[osz(1),prod(osz(2:legs+2))]),[sz(1)*prod(osz(2:legs+1)),osz(legs+2)]);
        vm{i} = reshape(permute(TN.core{i},[legs+2,1:legs+1]),[sz(legs+2),prod(sz(1:legs+1))])*vm{i}; % [r_{i+1} R_{i+1}]  
    end
    err((d-2)*(2*itr-2)+i) = norm(reshape(contract(TN)-contract(oTN),[numel(contract(oTN)),1]))/norm(reshape(contract(oTN),[numel(contract(oTN)),1]));
end
for i=d-1:-1:2
    osz = oTN.n(i,:);
    sz = TN.n(i,:);
    if i == d-1
        supercore = reshape(oTN.core{i+1},[oTN.n(i+1,1),prod(oTN.n(i+1,2:legs+2))]);
        supercore = reshape(reshape(oTN.core{i},[prod(osz(1:legs+1)),osz(legs+2)])*supercore,[osz(1),prod(TN.n(i,2:legs+1))*prod(TN.n(i+1,2:legs+2))]);
        supercore = reshape(vm{i-1}*supercore,[prod(TN.n(i,1:legs+1)),prod(TN.n(i+1,2:legs+2))]);
        if isempty(varargin)
            [U,S,V] = svd(supercore,'econ');
        else
            if min(size(supercore))<varargin{1}
                [U,S,V] = svd(supercore,'econ');
            else
                [U,S,V]=svds(supercore,varargin{1});
            end
        end  
        delta = S(1,1)*eps*max(size(supercore));
        s = diag(S);
        rr = sum(s>delta);  
        TN.core{i+1} = reshape(V(:,1:rr)',[rr,TN.n(i+1,2:legs+2)]);
        TN.core{i} = reshape(U(:,1:rr)*S(1:rr,1:rr),[TN.n(i,1:legs+1),rr]);
        TN.n(i+1,1) = rr;
        TN.n(i,end) = rr;
        sz(end) = rr;
        vp{d} = reshape(oTN.core{d},[osz(legs+2),prod(oTN.n(i+1,2:end))])*reshape(permute(TN.core{d},[2:legs+2,1]),[prod(oTN.n(d,2:legs+1))*TN.n(d,end),TN.n(d,1)]); % [R_d r_d] 
    else
        supercore = reshape(reshape(oTN.core{i+1},[prod(oTN.n(i+1,1:legs+1)),oTN.n(i+1,legs+2)])*vp{i+2},[oTN.n(i+1,1),prod(oTN.n(i+1,2:legs+1))*size(vp{i+2},2)]);
        supercore = reshape(reshape(oTN.core{i},[prod(osz(1:legs+1)),osz(legs+2)])*supercore,[osz(1),prod(TN.n(i,2:legs+1))*prod(TN.n(i+1,2:legs+2))]);
        supercore = reshape(vm{i-1}*supercore,[prod(TN.n(i,1:legs+1)),prod(TN.n(i+1,2:legs+2))]);
        if isempty(varargin)
            [U,S,V] = svd(supercore,'econ');
        else
            if min(size(supercore))<varargin{1}
                [U,S,V] = svd(supercore,'econ');
            else
                [U,S,V]=svds(supercore,varargin{1});
            end
        end  
        delta = S(1,1)*eps*max(size(supercore));
        s = diag(S);
        rr = sum(s>delta);  
        TN.core{i+1} = reshape(V(:,1:rr)',[rr,TN.n(i+1,2:legs+2)]);
        TN.core{i} = reshape(U(:,1:rr)*S(1:rr,1:rr),[TN.n(i,1:legs+1),rr]);
        TN.n(i+1,1) = rr;
        TN.n(i,end) = rr;
        sz(end) = rr;
        vp{i+1} = reshape(oTN.core{i+1},[prod(oTN.n(i+1,1:end-1)),oTN.n(i+1,end)])*vp{i+2}; % [R_{i+1}n_{i+1}n_{i+1} r_{i+2}]
        vp{i+1} = reshape(vp{i+1},[oTN.n(i+1,1),prod(oTN.n(i+1,2:legs+1))*TN.n(i+1,legs+2)])*reshape(permute(TN.core{i+1},[2:legs+2,1]),[prod(oTN.n(i+1,2:legs+1))*TN.n(i+1,end),TN.n(i+1,1)]); % [R_{i+1} r_{i+1}]
    end
    err((d-2)*(2*itr-1)+d-i) = norm(reshape(contract(TN)-contract(oTN),[numel(contract(oTN)),1]))/norm(reshape(contract(oTN),[numel(contract(oTN)),1]));
end
end

end