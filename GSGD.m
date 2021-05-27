% Guided Stochastic Gradient Descent (GSGD) 2.0
% Code has been simplified
% Copyright (c) 2018, Anuraganand Sharma - All rights reserved.
% 
% This is a simplified and more efficient Matlab code orignially proposed by Anuraganand Sharma in
% the paper:
% A. Sharma, “Guided Stochastic Gradient Descent Algorithm for inconsistent datasets,” 
% Applied Soft Computing, vol. 73, pp. 1068–1080, Dec. 2018 


function GSGD
clear all;
clc;
close all;

algos = {'Canonical','Momentum','Nesterov','Adagrad','Adadelta','RMSprop','Adam'}; 
bPlot = true;

[file,folder]=uigetfile('*.data');

%This is rho in the paper
ropeTeamSz = 10 %[1:10] %neighborhood size % May use Bayesian for parameter tuning

iAlgo = algos{1};
 
    bestFig = figure('Name','Best Figure');

    file_path = [];
    data = [];
    
    bestFig = clf(bestFig);


    fold = 3; %max(2,5); %minimum is 2
  
        [NC, data, file_path, x, y, N, d, inputVal, givenOut] = readData(data, file_path,file,folder);
        [~,file_used,~] = fileparts(file_path); %get only file name. Remove extension.
        if(NC == 2)
            NC = 1;
            activation = 'cross-entropy';
        elseif(NC>2)
            warning('Note: For multi-class classification, class labels MUST be from 0 to %d!\n',NC-1);
            activation='softmax';
        else
            warning('one class problem is not supported!\n');
            exit
        end
        range = ceil(N*(fold-1)/fold); %for one fold. 
        T = 2000;% 2*range*fold; %multiply by 2 is because of another "consistent run".
    

    eta = 0.2;
    
    NFC = 0;
    deleteWorstNoisePermanently = false; 

    W=zeros(d,NC);
    Whistory = cell(NC,2);
    gradHist = zeros(d, NC);
    r = zeros(d, NC);
    s = zeros(d, NC); 
    mAdam = zeros(d, NC);
    vAdam = zeros(d, NC);
    
    Wsgd = zeros(d,NC);
    WsgdHistory = cell(NC,2);
    gradHistSGD = zeros(d, NC);
    rSGD = zeros(d, NC);
    sSGD = zeros(d, NC);
    mAdamSGD = zeros(d, NC);
    vAdamSGD = zeros(d, NC);
%     Wmom = zeros(d,1); %momentum, currently working with Wsgd variable....
    Wprev = [];

    noiseIdx = []; 
    noiseScore = zeros(1,ropeTeamSz); % can range from 0-N in each sequence.
    noiseLevel = zeros(1,ropeTeamSz); % still not clear whether to take average or Max. currently it is average.

    PocketGoodWeights = cell(5,2); %for 51-60%, 61-70%, 71-80%, 81-90%, 91-100% SRs
    plotE = [];
    plotEgens = [];
    plotEout = [];
    PlotEoutSR = [];
    hold on;
    clr = 'k';
    %get current legend if exists:
    [bAddLegend, hPrev, prevLegendNames, curLegendName] = myUpdateLegend(iAlgo);

    pe = inf;  

%     ropeTeamSz = min(10, floor(0.75*N/fold));%rope-team: neighborhood size in paper
    % These are only rope-team contenders. Actual rope-team is consistentId
    
    t = 0;

    idx = randperm(N);
    e = 0;
    et = 0;
    fprintf('Please Wait!\n');
    while t <= T 
        t = t+1; 

    %<<< Ein error will be removed in future version. Kept for display perpose     
    % om scores/labes => other affected instances
        omPlusScore = zeros(1,ropeTeamSz);
        omPlusLevel = zeros(1,ropeTeamSz);

        omMinuScore = zeros(1,ropeTeamSz);
        omMinusLevel = zeros(1,ropeTeamSz);   
        
        et = et+1;
        if isempty(idx)
            idx = randperm(N);
            e = 0;
            et = 1;
        end
                
        er = randperm(N);
        er = er(1:2*ropeTeamSz); %for verification apporximate error.
        ve = 0;
        for k = er
            ve = ve + getError(k,x,y,W,activation);
        end
        ve = ve/size(er,2);
        
        %e = 0;
        tmpGuided = 0;
        for in =  idx % its never empty so tmpGuided > 0         
            if tmpGuided >= ropeTeamSz
                break;
            end
            [W, s, r, gradHist, Whistory, mAdam, vAdam] = ...
            SGDvariation(t, x(:,in), y(in), W, eta, activation, ...
            iAlgo, s, r, gradHist, Whistory, mAdam, vAdam);

            NFC = NFC+1;
            tmpGuided = tmpGuided + 1;
            nErr = getError(in,x,y,W,activation);
            e = e+nErr;
            
            %Collect Inconsistent Instances
            [omPlusScore, omPlusLevel, omMinuScore, omMinusLevel] = ...
            CollectInconsistentInstances (omPlusScore, omPlusLevel, omMinuScore,...
            omMinusLevel, idx,x,y,W,activation, ropeTeamSz, pe);
              
        end  %end for loop                

        %Its possible the last tmpGuided < ropeTeamSz.
        %Slowly the approximate error would become 'average error' 
        e = e/((et-1)*ropeTeamSz + tmpGuided); %An 'approximate' average error call it a verification error. OR incrementally add it? 
        e = (e+ve)/2;
        
        [consistentIdx, inconsistentIdx] = extractConsistentInstances ...
        (e, pe, omPlusScore, omPlusLevel, omMinuScore,...
        omMinusLevel, t, T, ropeTeamSz, N, tmpGuided, idx);

        pe = e; %Now previous error is used losely, however it ensures "ones upon a time" we had that error. 
        idx (1:tmpGuided) = [];
        
        for cI = consistentIdx
            [W, s, r, gradHist, Whistory, mAdam, vAdam] = ...
            SGDvariation(t, x(:,cI), y(cI), W, eta, activation, ...
            iAlgo, s, r, gradHist, Whistory, mAdam, vAdam);

            NFC = NFC+1;
        end
        
        if(mod(t,10)==0 || t == T)
                plotE = [plotE e]; %EsgdIII];
                plotEgens = [plotEgens t];

                % this is validation error 
                [PocketGoodWeights, doTerminate, SR, E] = validate(PocketGoodWeights,inputVal,W,givenOut,NFC, activation);

                if(doTerminate)
                    break;
                end
                plotEout = [plotEout E];
                PlotEoutSR = [PlotEoutSR SR];
            %%>>
        end
    end
    
    if(bPlot)       
        plot(plotEgens,plotE,'-', 'color', clr); %tGSGDplot
        hold on; 
        plot(plotEgens,plotEout,'--','color', 'm');
        plot(plotEgens,PlotEoutSR,'--','color', 'r');

        drawnow

        % range = sprintf('range*%d',range);
        xlabel('Selected Iterations','fontsize',10,'color','b')
        ylabel('Error (E_i_n/E_v)','fontsize',10,'color','b')
        title({['Overall Performance Digest: GSGD/SGD-' iAlgo], file_used});
        legend('Training','Validation');%,'Validation');
    end   
end

%% Collect Inconsistent Instances
%  omPlusScore = otherAffectedPointsScore;
%  omPlusLevel = otherAffectedPointsLevel
%  omMinuScore = otherAffectedPointsScore;
%  omMinusLevel = otherAffectedPointsLevel
%   
function [omPlusScore, omPlusLevel, omMinuScore, omMinusLevel] = ...
    CollectInconsistentInstances (omPlusScore, omPlusLevel, omMinuScore,...
    omMinusLevel, idx,x,y,W,activation, ropeTeamSz, pe)
    for j = 1:min(ropeTeamSz,size(idx,2))
        nErr = getError(idx(j),x,y,W,activation);
        %vector direction may be different but atleast getting lower error
        %check with avgError of previous iteration.
        %objective is to separate consistent with inconsistent
        if(nErr>pe)% of course, not applicable for the first index. 
            %It means, instance - not contributing
            omPlusScore(j) = 1;
            omPlusLevel(j) = nErr-pe;% how much not contributing .. om+
        else
            %the curIdx will fall here because it is always better
            %than pe. 
            omMinuScore(j) = 1;
            omMinusLevel(j) = pe-nErr;% how much not contributing ..om-
        end 
    end     
end

%%
function [consistentIdx, inconsistentIdx] = extractConsistentInstances ...
    (e, pe, omPlusScore, omPlusLevel, omMinuScore,...
    omMinusLevel, t, T, ropeTeamSz, N, tmpGuided, idx)
    noiseIdx = []; 
    noiseScore = zeros(1,ropeTeamSz); % can range from 0-N in each sequence.
    noiseLevel = zeros(1,ropeTeamSz); % still not clear whether to take average or Max. currently it is average.
    
    if(e<pe)%if good  
        inw = 1;
        noiseScore = noiseScore + omPlusScore*inw; %Few other points are bad
        noiseLevel = noiseLevel + omPlusLevel*inw;
    else
        %noise score update gives good results.
        inw = 1;
        noiseScore = noiseScore + omMinuScore*inw; %Few other points are bad
        noiseLevel = noiseLevel + omMinusLevel*inw;
        %curIdx is already catered above
    end


    %<<< Guided approach                     
    threshold = getNoiseScoreThreshold(t, T, ropeTeamSz);
    [noiseIdx, noiseScore, noiseLevel] = noiseSorting(noiseIdx, noiseScore, noiseLevel, ...
        'S->L', threshold);

    consistentIdx = [];
    inconsistentIdx = [];

    if(tmpGuided>0)
        consistentIdx = idx(setdiff(1:tmpGuided,noiseIdx));
        inconsistentIdx = idx(noiseIdx);
    end
end
%% Use before the algorithm starts to run in loops
function [bAddLegend, hPrev, prevLegend, curLegendName] = myUpdateLegend(algo)        
    curLegendName = strcat(algo, {' Ein'});
    [bAddLegend, hPrev, prevLegend, curLegendName] = updateLegend(curLegendName);
end

%%
% Threshold for noise score.
function [threshold] = getNoiseScoreThreshold(t, T, ropeTeamSz)
%     threshold = ceil(0.45*fold*(1/(1+exp(0.5*fold-v)))); 
 threshold = max(ceil(0.5*ropeTeamSz*t/T*(1/(1+exp(0.5*T-t)))),1);
end

%%
% Threshold for noise Idx only. It should be <= maxNoiseIdx
function [threshold] = getNoiseIdxThreshold(v, fold, maxNoiseIdx)
    threshold = ceil(0.45*maxNoiseIdx*(1/(1+exp(0.5*fold-v)))); 
end

%% check what noise level is tolerated.
function [noisy] = isNoisy(noiseIdx, threshold)
    noisy = false;
    if(noiseIdx >= threshold && noiseIdx > 0)
        noisy = true;
    end
end

%% noise sorting like Deb Ranking
% Sort by score if same then by level.
function [noiseIdx, noiseScore, noiseLevel] = noiseSorting(noiseIdx, noiseScore, ...
    noiseLevel, ranking, threshold)

    %noise ranking -> score->level 
    if(ranking == 'S->L')
        rank1 = noiseScore;
        rank2 = noiseLevel;
    elseif(ranking == 'L->S')
        rank1 = noiseLevel;
        rank2 = noiseScore;
        threshold = 0; %currently not catered.
    else
        error('Wrong ranking provided\n');
        exit;
    end

    [nval, nidx] = sort(rank1,'descend');
    
    for i = size(nval,2):-1:1 %fliplr(nidx)
        if(nval(i)<threshold) % you may change 0 to a small threathold as well :)
            nidx(i) = [];
        else
            break;
        end
    end
    
    firstLevelIdx = nidx; % size of nidx is less or equal to maxNoisePerGen
    maxNoisePerGen = size(firstLevelIdx,2);
    if(isempty(firstLevelIdx))
        return;
    end
                      
    lastHighIdx = firstLevelIdx(maxNoisePerGen);
    lastHighVal = rank1(lastHighIdx);
    finalIdx = find(rank1(firstLevelIdx)>lastHighVal);
    finalIdx = firstLevelIdx(finalIdx);
    szFinalIdx = size(finalIdx,2);                

    secondLevelIdx = find(rank1(firstLevelIdx) == lastHighVal);                     
    thirdLevelIdx =  rank2(firstLevelIdx(secondLevelIdx));
    [~, nidx] = sort(thirdLevelIdx,'descend');
                            
    nidx = nidx + szFinalIdx;

    secondLevelIdx = firstLevelIdx(nidx)  ;                          
    secondLevelIdx = secondLevelIdx(1:maxNoisePerGen-szFinalIdx);
    finalIdx = [finalIdx secondLevelIdx];
    
    noiseIdx = finalIdx;

    if(ranking == 'S->L')
        noiseScore = rank1; %remove
        noiseLevel = rank2; %remove
    elseif(ranking == 'L->S')
        noiseLevel = rank1; %remove
        noiseScore = rank2; %remove
    end
end

%% SGD gradient calculation
function gr = getSGD(xi, yi, W, type)
    if(strcmp(type,'cross-entropy'))
        s = W'*xi;
        gr = - yi*xi/(1+exp(yi*s)); %W'*xi)); % W at t i.e. W(t)
    elseif(strcmp(type,'softmax'))
        s = W'*xi; % W'(NCxd) * x(dx1) = s(NCx1); 
        [d, NC] = size(W);
        D = -max(s);
        s = exp(s+D)/sum(exp(s+D)); % s(NCx1)
        
        datasize = size(xi,2);      
        labels = full(sparse([ 1:datasize+1],[yi'+1;NC],1)); %groundTruth
        labels(end,:) = []; % labels (datasize x NC)
        labels = labels';
        try
            gr = (s-labels)*xi';
            gr = gr';
        catch ME
            d
            NC
            fprintf('s: [%d %d]\n',size(s));
            fprintf('xi: [%d %d]\n',size(xi));  
        end
    else
        error('Error! Activation function not supported.\n');
        exit
    end
end

%% BGD gradient calculation
function gr = getBGD(N, d, x, y, W)
    gr = zeros(d,1);
    for i = 1:N
        gri = -y(i)*x(:,i)/(1+exp(y(i)*W'*x(:,i))); % W at t i.e. W(t)
        gr = gr + gri;
    end 
    gr = (1/N)*gr;
end

%% get error
% for SGD idx is 1:N but for GSGD its random
function e = getError(idx,x,y,W,type)
    e = 0;
    if(strcmp(type,'cross-entropy'))
        for in = [idx] %take the full training sequence/data of this fold   
            e = e + log(1+exp(-y(in)*W'*x(:,in)));   
        end                  
    elseif(strcmp(type,'softmax')) % W has d dimenstion (same as input) and NC possible outputs
        datasize = size(idx,2);        
        [d, NC] = size(W);
        labels = full(sparse([ 1:datasize+1],[y(idx)'+1;NC],1)); %groundTruth
        labels(end,:) = []; % labels (datasize x NC)
        
        lb = 1;
        for in = [idx] 
            logits = W'*x(:,in); % W'(NCxd) * x(dx1) = logits(NCx1); %intermediate output is known as logits in machine learning
            D = -max(logits); %to cater large numbers                       
        
            tmp = exp(logits+D)/sum(exp(logits+D));
            tmp = labels(lb,:)*tmp;
            lb = lb+1;
            tmp = 1/tmp;
            e = e + log(tmp);
        end
    else
        error('Error! Activation function not supported.\n');
        exit;
    end
    
    sz = max(size(idx));
    e = e/sz; %average error         
end

%% all variations of the SGD algorithm
function [W, s, r, gradHist, Whistory, m, v] = SGDvariation(t, x, y, W, eta, activation, ...
    algo, s, r, gradHist, Whistory, m, v)
    beta = 0.9;
    epsilon = 1e-8;

    gr = getSGD(x, y, W, activation);

    if(strcmp(algo,'Canonical'))
        W = W-eta*gr;
    elseif(strcmp(algo,'Momentum'))
        bapplyMomentum = false;
        if(isempty(Whistory{1}) && isempty(Whistory{2})) 
            Whistory{1} = W;
        elseif(isempty(Whistory{2}))
            Whistory{2} = W;
        else % 2
            Whistory{1} = Whistory{2}; %Wt-2
            Whistory{2} = W; %Wt-1
            bapplyMomentum = true;
        end
        
        if(bapplyMomentum)
            rho = 0.9;
            W = Whistory{2}-eta*gr - rho*(Whistory{2}-Whistory{1});%momentum SGD
        else
            W = W-eta*gr;%simple SGD
        end
    elseif(strcmp(algo,'Nesterov'))
        bapplyNesterov = false;
        if(isempty(Whistory{1}) && isempty(Whistory{2})) 
            Whistory{1} = W;
        elseif(isempty(Whistory{2}))
            Whistory{2} = W;
        else % 2
            Whistory{1} = Whistory{2}; %Wt-2
            Whistory{2} = W; %Wt-1
            bapplyNesterov = true;
        end
        
        if(bapplyNesterov)
            rho = 0.9;
            gr = getSGD(x, y, W-rho*Whistory{2}, activation);
            W = Whistory{2}-eta*gr - rho*(Whistory{2}-Whistory{1});%nesterov SGD
        else
            gr = getSGD(x, y, W, activation);
            W = W-eta*gr;%simple SGD
        end
    elseif(strcmp(algo,'Adagrad'))
        gradHist = gradHist + gr.^2;
        W = W - eta * gr ./ (sqrt(gradHist) + epsilon); 
        
    elseif(strcmp(algo,'Adadelta'))
        if(t==1)
            r = gr.^2;
        else
            r = beta * r + (1-beta)* gr.^2;  
        end  
        
        v1 = - (sqrt(s + epsilon)./sqrt(r + epsilon)) .* gr;
        % update accumulated updates (deltas)
        %out
        s = beta * s + (1-beta)* v1.^2;
        %out
        W = W + v1; 

    elseif(strcmp(algo,'RMSprop'))
        if(t==1)
            r = gr.^2;
        else
            r = beta * r + (1-beta)* gr.^2;  
        end
        W = W - eta * gr ./ sqrt(r + epsilon); 
    elseif(strcmp(algo,'Adam'))
        beta1 = 0.9;
        beta2 = 0.999;
        
        m = beta1.*m + (1 - beta1).*gr;
        % Update biased 2nd raw moment estimate
        v = beta2.*v + (1 - beta2).*(gr.^2);

        % Compute bias-corrected 1st moment estimate
        mHat = m./(1 - beta1^t);
        % Compute bias-corrected 2nd raw moment estimate
        vHat = v./(1 - beta2^t);

        % Update decision variables
        W = W - eta.*mHat./(sqrt(vHat) + epsilon);
    else %error
        W = [];
        s = [];
        gradHist = [];
        warning('Incorrect choice of algorithm in SGDvariation!\n');
        exit
    end
            
end
