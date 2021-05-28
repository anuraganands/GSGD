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

    file_path = [];
    data = [];

  
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
    T = 2000;% 2*range*fold; %multiply by 2 is because of another "consistent run".
    

    eta = 0.2;    
    NFC = 0;

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
    % These are only rope-team contenders. Actual rope-team is consistentId
    
    t = 0;

    idx = randperm(N);
    e = 0;
    et = 0;
    fprintf('Please Wait!\n');
    while t <= T 
        t = t+1; 
  
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
        
%%  The core part of GSGD
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
            collectInconsistentInstances (omPlusScore, omPlusLevel, omMinuScore,...
            omMinusLevel, idx,x,y,W,activation, ropeTeamSz, pe);
              
        end  %end for loop                

        %Its possible the last tmpGuided < ropeTeamSz.
        %Slowly the approximate error would become 'average error' 
        e = e/((et-1)*ropeTeamSz + tmpGuided); %An 'approximate' average error call it a verification error. OR incrementally add it? 
        e = (e+ve)/2;
        
        %Extract consistent instances
        [consistentIdx, inconsistentIdx] = extractConsistentInstances ...
        (e, pe, omPlusScore, omPlusLevel, omMinuScore,...
        omMinusLevel, t, T, ropeTeamSz, N, tmpGuided, idx);

        pe = e; %Now previous error is used losely, however it ensures "ones upon a time" we had that error. 
        idx (1:tmpGuided) = [];
        
        %Further refinement
        for cI = consistentIdx
            [W, s, r, gradHist, Whistory, mAdam, vAdam] = ...
            SGDvariation(t, x(:,cI), y(cI), W, eta, activation, ...
            iAlgo, s, r, gradHist, Whistory, mAdam, vAdam);

            NFC = NFC+1;
        end
 
        %% Plot section
        if(mod(t,10)==0 || t == T)
            plotE = [plotE e]; 
            plotEgens = [plotEgens t];

            % this is validation error 
            [PocketGoodWeights, doTerminate, SR, E] = validate(PocketGoodWeights,inputVal,W,givenOut,NFC, activation);

            if(doTerminate)
                break;
            end
            plotEout = [plotEout E];
            PlotEoutSR = [PlotEoutSR SR];
        end
    end
    
    if(bPlot)       
        plot(plotEgens,plotE,'-', 'color', clr); %tGSGDplot
        hold on; 
        plot(plotEgens,plotEout,'--','color', 'm');
        plot(plotEgens,PlotEoutSR,'-.','color', 'r');

        drawnow

        % range = sprintf('range*%d',range);
        xlabel('Selected Iterations','fontsize',10,'color','b')
        ylabel('Error (E_i_n/E_v)','fontsize',10,'color','b')
        title({['Performance of GSGD-' iAlgo ' for '], file_used});
        legend('Training','Validation', 'SR');%,'Validation');
    end   
end

%% Use before the algorithm starts to run in loops
function [bAddLegend, hPrev, prevLegend, curLegendName] = myUpdateLegend(algo)        
    curLegendName = strcat(algo, {' Ein'});
    [bAddLegend, hPrev, prevLegend, curLegendName] = updateLegend(curLegendName);
end


