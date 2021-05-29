% Guided Stochastic Gradient Descent (GSGD) 2.0
% Code has been simplified
% Copyright (c) 2018, Anuraganand Sharma - All rights reserved.
% GSGDv2: This is a simplified Matlab code orignially proposed by Anuraganand Sharma in
% the paper:
% A. Sharma, “Guided Stochastic Gradient Descent Algorithm for inconsistent datasets,” 
% Applied Soft Computing, vol. 73, pp. 1068–1080, Dec. 2018
% The original version is GSGDv1.
% Please do report any bugs you find. 

function GSGDv2
    clear all;
    clc;
    close all;

    algos = {'Canonical','Momentum','Nesterov','Adagrad','Adadelta','RMSprop','Adam'}; 
    bPlot = true;

    [file,folder]=uigetfile('*.data');

    %This is rho in the paper
    ropeTeamSz = 10; %[1:10] %neighborhood size => increase rho value when the dataset is very noisy.

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
    T = 10000;
    

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

    pe = inf;  
    t = 0;

    idx = randperm(N);
    et = 0;
    E = inf;
    best_E = inf;
    fprintf('Please Wait!\n');
    
    while t <= T 
        t = t+1; 
        
        et = et+1;
        if isempty(idx)
            idx = randperm(N);
            et = 1;
        end
        curIdx = idx(et); %it can be noise...........................
        
        [W, s, r, gradHist, Whistory, mAdam, vAdam] = ...
            SGDvariation(t, x(:,curIdx), y(curIdx), W, eta, activation, ...
            iAlgo, s, r, gradHist, Whistory, mAdam, vAdam);
        NFC = NFC+1;
                   
        er = randperm(N);
        er = er(1:ceil(1.0*N)); %for verification apporximate error.
        ve = 0;
        for k = er
            ve = ve + getError(k,x,y,W,activation);
        end
        ve = ve/size(er,2);
        
        %Collect Inconsistent Instances
        [omPlusScore, omPlusLevel, omMinuScore, omMinusLevel, tmpGuided] = ...
        collectInconsistentInstances (idx,x,y,W,activation, ropeTeamSz, best_E);             

        %Extract consistent instances
        consistentIdx = extractConsistentInstances ...
        (ve, best_E, omPlusScore, omPlusLevel, omMinuScore,...
        omMinusLevel);
    
        %Further refinement
        for cI = consistentIdx
            [W, s, r, gradHist, Whistory, mAdam, vAdam] = ...
            SGDvariation(t, x(:,cI), y(cI), W, eta, activation, ...
            iAlgo, s, r, gradHist, Whistory, mAdam, vAdam);

            NFC = NFC+1;
        end

        pe = ve;
        
        if mod(t,ropeTeamSz) == 0  || mod(t,tmpGuided) == 0
            idx (1:tmpGuided) = [];
            et = 0;
        else
            [~,inconsistentIdx] = setdiff(idx(1:tmpGuided),idx(consistentIdx));
            tmpVals = idx(inconsistentIdx);
            idx(inconsistentIdx) = [];
            idx = [idx tmpVals]; %hide inconsistent instances for a while
        end
            
        %% Plot section
        if(mod(t,10)==0 || t == T)
            plotE = [plotE ve]; 
            plotEgens = [plotEgens t];

            % this is validation error 
            [PocketGoodWeights, doTerminate, SR, E] = validate(PocketGoodWeights,inputVal,W,givenOut,NFC, activation);

            if(best_E<E)
                best_E;
            end
            if(doTerminate)
                break;
            end
            plotEout = [plotEout E];
            PlotEoutSR = [PlotEoutSR SR];
        end
    end
   [SR, NFC] = PrintFinalResults([],PocketGoodWeights,inputVal,givenOut, false, activation)
   
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
