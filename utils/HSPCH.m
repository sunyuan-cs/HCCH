function [ImgToTxt,TxtToImg] = HSPCH(trainLabel, param, dataset)
  
X1 = dataset.XDatabase;
X2 = dataset.YDatabase;
XTest = dataset.XTest;
YTest = dataset.YTest;
testL = dataset.testL;
databaseL = dataset.databaseL;

% top_K=1000;
[d1,~] = size(X1');
[d2,~] = size(X2');
bit = param.bit;
maxIter = param.maxIter;
lambda = param.lambda;
beta = param.beta;
alpha = param.alpha;
r = param.r;
numTrain = size(trainLabel, 1);

%--------------------------------initial------------------------------------

B1= ones(numTrain, bit); 
B1(randn(numTrain, bit) < 0) = -1;
B2=B1;
H=B1;

P1=randn(d1,r);
P2=randn(d2,r);
R1=randn(r,bit);
R2=randn(r,bit);

XTX1=X1'*X1;
XTX2=X2'*X2;
YYT=trainLabel*trainLabel';


for epoch = 1:maxIter
tic 
   %--------- B-step
   B1=sgn(X1*P1*R1+alpha*bit*YYT*H);
   B2=sgn(X2*P2*R2+beta*bit*YYT*H);
   H_C = (alpha*bit*YYT*B1+beta*bit*YYT*B2)';
   H=hash_const(H_C,bit,numTrain)';  
     
   %--------- PR-step  
   PR1 = P1*R1;
   PR2 = P2*R2;
   v1  = sqrt(sum(PR1.*PR1,2)+eps);
   v2  = sqrt(sum(PR2.*PR2,2)+eps);
   M1  = diag(1./(2*v1));
   M2  = diag(1./(2*v2)); 
   St1 = XTX1+lambda*M1;
   St2 = XTX2+lambda*M2;
   Sb1 = X1'*B1*B1'*X1;
   Sb2 = X2'*B2*B2'*X2;
   [P1, ~, ~]=eig1((St1\Sb1), r, 1);
   [P2, ~, ~]=eig1((St2\Sb2), r, 1);
   R1 = (P1'*St1*P1)\(P1'*X1'*B1);
   R2 = (P2'*St2*P2)\(P2'*X2'*B2); 
   
   %real-time evaluation
    tBX = sign(XTest * P1*R1);
    tBY = sign(YTest * P2*R2);
    sim_ti = B1 * tBX';
    sim_it = B2 * tBY';
    R = size(H,1); 
    ImgToTxt = mAP(sim_ti,databaseL,testL,R);%
    TxtToImg= mAP(sim_it,databaseL,testL,R);%
    fprintf('...iter:%d,   i2t:%.4f,   t2i:%.4f\n',epoch, ImgToTxt, TxtToImg)

toc
end


end




