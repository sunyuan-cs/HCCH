clear all
warning off
addpath(genpath(fullfile('utils/')));

seed = 0;
rng('default');
rng(seed);
param.seed = seed;

dataname = 'wikiData';
%% parameters setting
param.dataname = dataname;
param.method = 'HSPCH';

bits = [8,16,32,64];
nb = numel(bits);

param.bits = bits;
param.maxIter = 20;


alpha = [1];  %[1e-4,1e-3,1e-2,1e-1,1,10,100]; 
beta = [1e-3];%[1e-4,1e-3,1e-2,1e-1,1,10,100]; 
lambda =[1e-4];%[1e-5,1e-4,1e-3,1e-2,1e-1];
r = [400];

par_1 = numel(alpha);
par_2 = numel(beta);
par_3 = numel(lambda);
par_4 = numel(r);


%load dataset
dataset = load_data(dataname);
n_anchors1 = 1500;
n_anchors2 = 1500;



rbf2;
total_res=[];
% run algorithm
for i = 1: nb
    for ij=1:par_1
        for jj=1:par_2
            for jjj=1:par_3
                for jjjj=1:par_4
                fprintf('...method: %s\n', param.method);
                fprintf('...bit: %d\n', bits(i));
                param.bit = bits(i);
                param.alpha = alpha(ij);
                param.beta= beta(jj);
                param.lambda= lambda(jjj);
                param.r= r(jjjj)
                trainL = dataset.databaseL;
%                 trainL = normr(trainL);

                
                [ImgToTxt,TxtToIm] = HSPCH(trainL, param, dataset)
                total_res = [total_res;ImgToTxt,TxtToIm, param.bit, param.alpha, param.beta,param.lambda,param.r];
                end
            end
        end       
    end
end

save('result_wiki.mat','total_res');
