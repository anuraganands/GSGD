% Guided Stochastic Gradient Descent (GSGD)
% 
% Copyright (c) 2018, Anuraganand Sharma - All rights reserved.
% 
% This Matlab code is the implementation of GSGD proposed by Anuraganand Sharma in
% the paper:
% A. Sharma, “Guided Stochastic Gradient Descent Algorithm for inconsistent datasets,” 
% Applied Soft Computing, vol. 73, pp. 1068–1080, Dec. 2018 


function GSGD
clear all;
clc;
close all;
% opengl('save', 'software');

% % set(handleToYourMainGUI, 'HandleVisibility', 'off');
% % close all;
% % set(handleToYourMainGUI, 'HandleVisibility', 'on');

import Structure.*;

algos = {'Canonical','Momentum','Nesterov','Adagrad','Adadelta','RMSprop','Adam'}; 
bMNIST = false; %use or not to use MNIST datasets
bPlot = false;

if(bMNIST)
    filesExp = [];
    totalExpFiles = 1;
else
    dirExp = uigetdir(pwd, 'Select a folder');
    filesExp = dir(fullfile(dirExp, '*.data'));
    totalExpFiles = size(filesExp,1);
end

for ropeTeamSz = 10:10 %[1:10] %neighborhood size % May use Bayesian for parameter tuning

for iAlgo = algos
    keepMainVars = {'dirExp', 'filesExp','totalExpFiles', 'algos','iAlgo', ...
        'bMNIST','bPlot','ropeTeamSz'};
    clearvars('-except', keepMainVars{:});
    
for ff = 1:totalExpFiles
  
    %Comment out this code%%%%%%%%%%%%%%%%%%%%%%%%%
%     if (ff ~= 2) % only for given data set(s) 1=> breast cancer, 7 => new-thyroid, 2=> cancer
%         continue;
%     end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    
    keepMainVars = {'dirExp', 'filesExp','totalExpFiles','ff', 'algos','iAlgo', ...
        'bMNIST','bPlot','ropeTeamSz'};
    clearvars('-except', keepMainVars{:});
     
    bestFig = figure('Name','Best Figure');
    avgFig = figure('Name','Average Figure');
    digestFig = figure('Name','Digest Figure');

    bPlotMain = false;
    AggregateAnalysisL = -1;
    AggregateAnalysisLsgd = -1;
    file_path = [];
    data = [];
    totalRun = 5; 
    
    bestFig = clf(bestFig);
    bestFig = figure(bestFig);
    avgFig = clf(avgFig);
    avgFig = figure(avgFig);

for runs = 1:totalRun
    fprintf('Run No. %d\n',runs);
    keepvars = {'data', 'file_path','totalRun','runs','bPlotMain', 'bestFig',...
        'avgFig','digestFig','AggregateAnalysisL','AggregateAnalysisLsgd',...
        'dirExp', 'filesExp','totalExpFiles','ff', 'algos','iAlgo', ...
        'bMNIST','bPlot','ropeTeamSz'};
    clearvars('-except', keepvars{:});
    if(runs == totalRun)
        bPlotMain = true;
    end

    NC = 0; %Number of classes    
    T = 0;
    range = 0;
    fold = 3; %max(2,5); %minimum is 2
    
    if(~bMNIST)
        [NC, data, file_path, x, y, N, d, inputVal, givenOut] = readData(data, file_path,filesExp(ff).name,filesExp(ff).folder);
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
        T = 1000;% 2*range*fold; %multiply by 2 is because of another "consistent run".
    else
        images = loadMNISTImages('data/train-images.idx3-ubyte');
        labels = loadMNISTLabels('data/train-labels.idx1-ubyte');
        x=images;
        [d,N] = size(x);
        newX0 = ones(1,N);
        x = vertcat(newX0, x);
        d = d+1;
        y = labels;
        y = y';
        images = loadMNISTImages('data/t10k-images.idx3-ubyte');
        labels = loadMNISTLabels('data/t10k-labels.idx1-ubyte');
        inputVal = images;
        [~,tmp] = size(inputVal);
        newX0 = ones(1,tmp);
        inputVal = vertcat(newX0, inputVal);
        givenOut = labels;
        givenOut = givenOut';
        file_path = 'data/train-images.idx3-ubyte';
        [~,file_used,~] = fileparts(file_path); %get only file name. Remove extension.
        NC = 10; %Number of classes
        activation='softmax';
        T = 10000;
    end

    eta = 0.2;
    
    NFC = 0;
    NFCsgd = 0;
    applyNoiseFilter = true; %make it false and this algorithm will become simple SGD.
    deleteWorstNoisePermanently = false; 
    bNoiseSorting = true;

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

    noiseIdx = zeros(1,N); 
    noiseScore = zeros(1,N); % can range from 0-N in each sequence.
    noiseLevel = zeros(1,N); % still not clear whether to take average or Max. currently it is average.

    %increasing maxNoisePerGen makes significant difference more significant.
    maxNoisePerGen = max(1,ceil(N*0.25));%***** % percentage of data filtered in one full Nfolds.
    maxNoiseIdx = 3; %maxNoisePerGen; %ceil(maxNoisePerGen*0.5);  % Times noisy data remain filtered in different full Nfolds. Like Jailed for some time....

    PocketGoodWeights = cell(5,2); %for 51-60%, 61-70%, 71-80%, 81-90%, 91-100% SRs
    PocketGoodWeightsSGD = cell(5,2);
    plotE = [];
    plotEgens = [];
    plotV = [];
    plotVgens = [];
    plotEsgd = [];
    plotEsgdGens = [];
    plotVsgd = [];
    plotEout = [];
    PlotEoutSR = [];
    PlotEoutSRsgd = [];
    plotVfullfold = [];
    plotVfullfoldGens = [];
    hold on;
    clr = 'k';
    %get current legend if exists:
    [bAddLegend, hPrev, prevLegendNames, curLegendName] = myUpdateLegend(iAlgo);

    figLabeled = false;
    pe = inf;  

    v = 1;    
    cvSets = makeCrossValidationSets(N,fold,true); %randomized.
    idx = []; 
    for cv = v+1:fold
        idx = [idx  cvSets{cv}]; % only training data
    end
    for cv = 1:v-1
        idx = [idx  cvSets{cv}]; % only training data
    end
    processedIdx = [];
    consistentIdx = []; %idx;
%     curFoldConsistentIdx = consistentIdx;

    bestVEsoFar=inf;
    EvFold = zeros(fold,1);
    bestWsoFarVE = W;

    EinSGDtail = GradientDescentTail;
    EinSGDIIItail = GradientDescentTail;
    EvSGDtail = [];
    EvSGDIIItail = [];

    bFilter = false;
    isGuided = false; %normal if false, guided if true;
%     ropeTeamSz = min(10, floor(0.75*N/fold));%rope-team: neighborhood size in paper
    ropeTeam = [];%will have: |ropeTeam|<=ropeTeamSz; 
    % These are only rope-team contenders. Actual rope-team is consistentId
    
    gs = 0;
    t = 0;
% %     fprintf('Iteration t = ');
    while t < T 
        t = t+1;
        %<< cross validation
        if(isempty(idx))% || mod(t,100)==0) %% && isempty(consistentIdx) && t>fold*fold) %it implicitly means one complete sequence [1-N] has been completed.        
            if(applyNoiseFilter && v==1) %v==fold && applyNoiseFilter) %the last fold is emptied.
                cvSets = makeCrossValidationSets(N,fold,true);           
            end
  
            idx = [];
            processedIdx = [];
            v = mod(v,fold)+1;
            for cv = v+1:fold
                idx = [idx  cvSets{cv}]; % only training data
            end
            for cv = 1:v-1
                idx = [idx  cvSets{cv}]; % only training data
            end
%             curFoldConsistentIdx = consistentIdx; %just save a copy.
        end   
        %>>

        if(isGuided)
            %consistentIdx;
            ropeTeam = [];
        else
            curIdx = idx(1); %it can be noise...........................
            idx(1) = [];
            processedIdx = [processedIdx curIdx];

            [Wsgd, sSGD, rSGD, gradHistSGD, WsgdHistory, mAdamSGD, vAdamSGD] = ...
                SGDvariation(t, x(:,curIdx), y(curIdx), Wsgd, eta, activation, ...
                iAlgo, sSGD, rSGD, gradHistSGD, WsgdHistory, mAdamSGD, vAdamSGD);
            
            NFCsgd = NFCsgd+1;
       
            consistentIdx = curIdx; %[curIdx consistentIdx]; %no fix it.
            ropeTeam = [ropeTeam curIdx];
        end
        
        for cI = consistentIdx 
%         if(~isempty(consistentIdx)) % for cI = consistentIdx %Note curIdx variable is used for two separate purpose
            %one for normal phase and one for guided phase
            if(isGuided)
                curIdx = cI;
                ropeTeam = [];
                %ignore first one
                if(cI ~= consistentIdx(1))
                    t = t+1;
                    if(t>T)
                        break;
                    end
                end
            else
                curIdx = consistentIdx(1); % same as in SGD
            end
            
            [W, s, r, gradHist, Whistory, mAdam, vAdam] = ...
                SGDvariation(t, x(:,curIdx), y(curIdx), W, eta, activation, ...
                iAlgo, s, r, gradHist, Whistory, mAdam, vAdam);
                  
            NFC = NFC+1;
     
        
        %<<< Ein error will be removed in future version. Kept for display perpose     
            otherAffectedPointsScore = zeros(1,N);
            otherAffectedPointsLevel = zeros(1,N);
            
            otherAffectedPointsScore2 = zeros(1,N);
            otherAffectedPointsLevel2 = zeros(1,N);
                      
            e = 0;
            eSGD = 0;
            tmpGuided = 0;
            for in =  [idx processedIdx] % or [consistentIdx inconsistentIdx] same thing  %cvSets{v} %
                tmpGuided = tmpGuided + 1;
                eSGD = eSGD + getError(in,x,y,Wsgd,activation);
                nErr = getError(in,x,y,W,activation);
                
                
                if(min(size(eSGD)) ~= min(size(nErr)) ||  min(size(eSGD)) ~= 1 || min(size(nErr)) ~= 1)
                    fprintf('Error! Not possible...\n');
                end

                %vector direction may be different but atleast getting lower
                %error
                %check with avgError of previous iteration???
                %object is to separate consistent with inconsistent
                try
                    if(~isGuided && (tmpGuided >= size([idx processedIdx],2)-ropeTeamSz ) )
                        if(nErr>pe)%bad - not contributing
                            %only hold the data temporarily.... this is om+
                            otherAffectedPointsScore(in) = 1;
                            otherAffectedPointsLevel(in) = nErr-pe;% how much not contributing .. om+
                        else
                            %the curIdx will fall here because it is always better
                            %than pe. ... this is om-
                            otherAffectedPointsScore2(in) = 1;
                            otherAffectedPointsLevel2(in) = pe-nErr;% how much not contributing ..om-
                        end 
                    end
                catch
                     fprintf('Error!!\n');
                end
                
                e = e+nErr;
            end  %end for loop                
        
            e = e/size([idx processedIdx],2); %average error
            eSGD = eSGD/size([idx processedIdx],2);  %cvSets{v},2); %
            pe = e; %Now previous error is used losely, however it ensures "ones upon a time" we had that error.
            
% % %       fprintf('%.0f - Cross Entropy Error%%: %.2f, idx=%.0f, [size: %.0f/%.0f]\n', t, e,curIdx,curN,N-size(cvSets{v},2)); %N);
        %>>> Ein error

            if(~isGuided)
                % max score: 1 point / fold.
                if(e<pe)%if good  
                    inw = 1;
                    noiseScore = noiseScore + otherAffectedPointsScore*inw; %Few other points are bad
                    noiseLevel = noiseLevel + otherAffectedPointsLevel*inw;
                else
                    %noise score update gives good results.
                    inw = 1;
                    noiseScore = noiseScore + otherAffectedPointsScore2*inw; %Few other points are bad
                    noiseLevel = noiseLevel + otherAffectedPointsLevel2*inw;
                    %curIdx is already catered above
                end
            end
   
        %<<< Guided approach
            if(ready4revisit(processedIdx, idx, ropeTeamSz) && ~isGuided)
                if(bNoiseSorting)  
                    bRefresh = false;
                    if (v==fold)
                        bRefresh = true;
                    end               

                    threshold = getNoiseScoreThreshold(v, fold);
                    [noiseIdx, noiseScore, noiseLevel] = noiseSorting(noiseIdx, noiseScore, noiseLevel, ...
                        maxNoiseIdx, maxNoisePerGen, deleteWorstNoisePermanently,'S->L', bRefresh, threshold);
                end

                nse = zeros(1,maxNoiseIdx+1);  %noiseIdx is from 0 - 5.
                tmpIdx = [];
                for i = 1:N      
                    nse(noiseIdx(i)+1) = nse(noiseIdx(i)+1)+1; %noiseIdx is from 0 - 5.
                    if(noiseIdx(i)>= maxNoiseIdx)
                        tmpIdx = [tmpIdx i];
                    end
                end
       
                consistentIdx = [];
                inconsistentIdx = [];
                for i = processedIdx(end-ropeTeamSz+1:end)
                    if(~isNoisy(noiseIdx(i), 0))
                        consistentIdx = [consistentIdx i]; %staying for too long....
                    else
                        inconsistentIdx = [inconsistentIdx i];
                    end
                end
                consistentIdx = consistentIdx(1:min(size(consistentIdx,2),ropeTeamSz));
                
                noiseIdx = zeros(1,N); 
                noiseScore = zeros(1,N); % can range from 0-N in each sequence.
                noiseLevel = zeros(1,N); % still not clear whether to take average or Max. currently it is average.
    
                isGuided = true; %ready to start guided phase
            end
        %>>>

        % if overallBest is not being used...
        ve = getError(cvSets{v},x,y,W, activation);
        veSGD = getError(cvSets{v},x,y,Wsgd, activation);
        EvFold(v) = ve;
        plotV = [plotV ve];
        plotVgens = [plotVgens t];
        plotVsgd = [plotVsgd veSGD];

            plotE = [plotE e]; %EsgdIII];
            plotEgens = [plotEgens t];

%             if(~isGuided) - now it will have duplicate eSGDs
                plotEsgd = [plotEsgd eSGD]; % Esgd]; %will give same values incase of isGuided.
                plotEsgdGens = [plotEsgdGens t];
%             end
              
            if(v == fold && isempty(idx)) %the last fold
                mn = mean(EvFold);
                EvFold = zeros(fold,1);
                if (mn<bestVEsoFar)
                    bestVEsoFar = mn;
                    bestWsoFarVE = W;
                end
    % %             fprintf('mean value %.02f\n',mn);
                if(mn>0)
                    plotVfullfold = [mn]; %validation error
                    plotVfullfoldGens = [t];
                end
            end
            
            if(isGuided && cI == consistentIdx(end) && isempty(ropeTeam))%last one
                isGuided = false;
            end 
% %             fprintf('%d, ',t);
  
        % Eout .......................................
        % <<
% % %             [PocketGoodWeights, doTerminate, SR, E] = validate(PocketGoodWeights,inputVal,W,givenOut,NFC, activation);
% % %             [PocketGoodWeightsSGD, ~, ~, ~] = validate(PocketGoodWeightsSGD,inputVal,Wsgd,givenOut,NFCsgd, activation);
% % %             
% % %             if(doTerminate)
% % %                 break;
% % %             end
% % %             plotEout = [plotEout E];
% % %             PlotEoutSR = [PlotEoutSR SR];

            % this is validation error 
            [PocketGoodWeights, doTerminate, SR, E] = validate(PocketGoodWeights,x(:,cvSets{v}),W,y(cvSets{v}),NFC, activation);
            [PocketGoodWeightsSGD, ~, SRsgd, ~] = validate(PocketGoodWeightsSGD,x(:,cvSets{v}),Wsgd,y(cvSets{v}),NFCsgd, activation);
            
            if(doTerminate)
                break;
            end
            plotEout = [plotEout E];
            PlotEoutSR = [PlotEoutSR SR];
            PlotEoutSRsgd = [PlotEoutSRsgd SRsgd];
        %%>>

% % %             if(bPlotMain)
% % %                 mainFig = figure(mainFig);
% % %                 hCur=plot(plotEgens,plotE, 'color', clr);
% % %                 set(hCur,'XData',plotEgens,'YData',plotE);
% % %                 hCurSGD = plot(plotEsgdGens,plotEsgd,'--m');
% % %                 set(hCurSGD,'XData',plotEsgdGens,'YData',plotEsgd);
% % %                 hVal=plot(plotVgens,plotV, ':r');
% % %                 set(hVal,'XData',plotVgens,'YData',plotV);
% % % 
% % % % %                 if(v == fold && ~isempty(plotVfullfold)) % && isempty(idx))
% % % % %                     plot(plotVfullfoldGens,plotVfullfold, 'bs',...
% % % % %                     'LineWidth',2,...
% % % % %                     'MarkerSize',10,...
% % % % %                     'MarkerEdgeColor','b',...
% % % % %                     'MarkerFaceColor',[0.5,0.5,0.5]);
% % % % %                 end
% % % %             hEout = plot(plotEgens,plotEout, '+b');
% % % %             set(hEout,'XData',plotEgens,'YData',plotEout);
% % % %             plot(1:size(PlotEoutSR,2),PlotEoutSR, 'color', 'c');
% % %                 drawnow
% % % 
% % %                 if(~figLabeled)
% % %                     figLabeled = true;
% % %                     xlabel('Epochs','fontsize',10,'color','b')
% % %                     ylabel('Error','fontsize',10,'color','b')
% % % 
% % %                     if bAddLegend
% % %                         curLegendNameSGD = {'SGD E_i_n'};
% % %                         legend(vertcat(hPrev, hCur, hCurSGD, hVal),vertcat(prevLegendNames, curLegendName, curLegendNameSGD, 'validation'));%, 'E_o_u_t')); 
% % %                     end
% % %                     title(['Gradient Descent Algorithms: ' iAlgo]);
% % %                 end
% % %             end
        end %guided for loop
    % SGD >>         
    end % iteration loop

fprintf('Final Eout for SGDIII\n');
[SR_SGDIII, NFC] = PrintFinalResults([],PocketGoodWeights,inputVal,givenOut, false, activation);

fprintf('Final Eout for SGD ONLY\n');
[SR_SGD, NFCsgd] = PrintFinalResults([],PocketGoodWeightsSGD,inputVal,givenOut, false, activation);

fprintf('Gradient Descent Algorithms: %s\n', cell2mat(iAlgo));

fprintf('Testing phase: Best W with validation error ONLY\n');
PrintFinalResults([],bestWsoFarVE,inputVal,givenOut, false, activation);

% s = floor(T/(N*(fold-1)/fold)); %if n fold cross validation is used
% range = max(3,floor(s/10));

% % range = ceil(N*(fold-1)/fold);
sampleSz = 5;
range = floor(range/sampleSz);

%SGD-canonical
%<<
    SGDplot = [];
    tSGDplot = [];
%     Tsgd = size(plotEsgd,2); %it is less than T
%     Tsgd = min(T,Tsgd);
    for k = 0:floor(T/range)-1
        SGDplot  = [SGDplot min(plotEsgd(k*range+1:(k+1)*range))];
        tSGDplot = [tSGDplot plotEsgdGens(k*range+1)];
    end
    SGDplot  = [SGDplot min(plotEsgd(k*range+1:T))];
    tSGDplot = [tSGDplot plotEsgdGens(min((k+1)*range,T))]; %last one
%>>

%GSGD
%<<
    GSGDplot = [];
    tGSGDplot = [];
%     plotE = plotE(1:min(size(plotE,2),T));
%     plotEgens = plotEgens(1:min(size(plotEgens,2),T));
%     TE = size(plotE,2);
%     TE = min(T,TE); % to have fair comparision

    for k = 0:floor(T/range)-1
        GSGDplot  = [GSGDplot min(plotE(k*range+1:(k+1)*range))];
        %just take the first iteration, otherwise you will have to select which 
        %one is giving min and then get that one.
        tGSGDplot = [tGSGDplot plotEgens(k*range+1)];  
    end
    GSGDplot  = [GSGDplot min(plotE(k*range+1:T))];
    tGSGDplot = [tGSGDplot plotEgens(min((k+1)*range,T))]; %last one

%>>


%validation with GSGD
%<<
    Vplot = []; %all happens after range*sampleSz;
    tVplot = [];
%     TV = TE; %same T
    for k = 0:floor(T/range)-1
        Vplot  = [Vplot min(plotV(k*range+1:(k+1)*range))];
        %just take the first iteration, otherwise you will have to select which 
        %one is giving min and then get that one.
        tVplot = [tVplot plotVgens(k*range+1)];  
    end
    Vplot  = [Vplot min(plotV(k*range+1:T))];
    tVplot = [tVplot plotVgens(min((k+1)*range,T))]; %last one

    VsgdPlot = []; %all happens after range*sampleSz;
    for k = 0:floor(T/range)-1
        VsgdPlot  = [VsgdPlot min(plotVsgd(k*range+1:(k+1)*range))];
    end
    VsgdPlot  = [VsgdPlot min(plotVsgd(k*range+1:T))];
%>>


    %Save micro Ein
    outFileMicroEin = ['Result/' cell2mat(iAlgo) '/' file_used '_microEin' '.csv' ];
    if(runs == 1)
        dlmwrite(outFileMicroEin,plotE);  
    else
        dlmwrite(outFileMicroEin,plotE,'-append');
    end
    dlmwrite(outFileMicroEin,plotV,'-append');
    dlmwrite(outFileMicroEin,plotEsgd,'-append');
    dlmwrite(outFileMicroEin,plotEsgdGens,'-append');
    dlmwrite(outFileMicroEin,plotVsgd,'-append');
    dlmwrite(outFileMicroEin,PlotEoutSR,'-append');
%     dlmwrite(outFileMicroEin,PlotEoutSRsgd,'-append');
    
    plotEsgd = [];
    plotEsgdGens = [];
    plotE = [];
    plotEgens = [];
    plotV = [];
    plotVgens = [];
    plotVsgd = [];
    
    L = size(GSGDplot,2);
    Lsgd = size(SGDplot,2);
    
    if(bPlot)
        digestFig = figure(digestFig);
        
        plot(tGSGDplot,GSGDplot,'-', 'color', clr); %tGSGDplot
        hold on; 
        plot(tSGDplot,SGDplot,'--','color', 'm');
    %     plot(tVplot,Vplot,':','color', 'r'); %tVplot
    %     plot(tVplot,VsgdPlot,'-.','color', 'b'); %tVplot
        % % plot(tVplot,Vplot, 'bs',...
        % %                    'LineWidth',2,...
        % %                    'MarkerSize',10,...
        % %                    'MarkerEdgeColor','b',...
        % %                    'MarkerFaceColor',[0.5,0.5,0.5]);

        drawnow

        % range = sprintf('range*%d',range);
        xlabel('Selected Iterations','fontsize',10,'color','b')
        ylabel('Error (E_i_n/E_v)','fontsize',10,'color','b')
        title({['Overall Performance Digest: GSGD/SGD-' cell2mat(iAlgo)], file_used});
        legend('GSGD','SGD');%,'Validation');
    end

    %Save macro Ein
    outFileSGDIII = ['Result/' cell2mat(iAlgo) '/' file_used 'GSGD' '.csv' ];
    GSGDplot = [runs GSGDplot SR_SGDIII NFC]; %1+L+2 %column size
    if(runs == 1)
        dlmwrite(outFileSGDIII,GSGDplot);  
    else
        dlmwrite(outFileSGDIII,GSGDplot,'-append');
    end
    if(AggregateAnalysisL<1+L+2) %column size
        AggregateAnalysisL = 1+L+2;
    end

    %Save Ein
    outFileSGD = ['Result/' cell2mat(iAlgo) '/' file_used 'SGD' '.csv' ];
    SGDplot = [runs SGDplot SR_SGD NFCsgd];
    if(runs == 1)
        dlmwrite(outFileSGD,SGDplot);
    else
        dlmwrite(outFileSGD,SGDplot,'-append');
    end
    if(AggregateAnalysisLsgd<1+Lsgd+2)  %column size
        AggregateAnalysisLsgd = 1+Lsgd+2;
    end

end % go to next run


%read the files and analyse
[SRsSGDIII, fileData1, microEin] = aggregateAnalysis(outFileSGDIII,AggregateAnalysisL, outFileMicroEin, T, totalRun*4);
[SRsSGD, fileData2, ~] = aggregateAnalysis(outFileSGD,AggregateAnalysisLsgd, outFileMicroEin, T, totalRun*4);
%microEinSGDIII :cell(4,2) %     
%   1-plotE - 1:best; 2:average
%   2-plotV - 1:best; 2:average
%   3-plotEsgd - 1:best; 2:average
%   4-plotEsgdGens - 1:best; 2:average
%   5-plotVsgd - 1:best; 2:average
%   x6-PlotEoutSR - 1:best; 2:average
%   x7-PlotEoutSRsgd - 1:best; 2:average

outFileAggEin = ['Result/' cell2mat(iAlgo) '/' file_used '_aggEin' '.csv' ];
dlmwrite(outFileAggEin,[ropeTeamSz microEin{1,1}-microEin{3,1}],'-append');% best: E-Esgd 
dlmwrite(outFileAggEin,[ropeTeamSz microEin{1,2}-microEin{3,2}],'-append');% average: E-Esgd 
dlmwrite(outFileAggEin,[ropeTeamSz microEin{6,2}],'-append');% average SR

if(bPlot)
%Plot overall best
%<<
    bestFig = figure(bestFig);
    hold on;
    hCur=plot(1:T,microEin{1,1}, 'color', clr);
    set(hCur,'XData',1:T,'YData',microEin{1,1});
    hCurSGD = plot(microEin{4,1},microEin{3,1},'--m');
    set(hCurSGD,'XData',microEin{4,1},'YData',microEin{3,1});
    hVal=plot(1:T,microEin{2,1}, ':r');
    set(hVal,'XData',1:T,'YData',microEin{2,1});
    hValSGD=plot(1:T,microEin{5,1}, '-.b');
    set(hValSGD,'XData',1:T,'YData',microEin{5,1});

    drawnow

    xlabel('Epochs','fontsize',10,'color','b')
    ylabel('Error (E_i_n/E_v)','fontsize',10,'color','b')

    legend(vertcat(hCur, hCurSGD, hVal, hValSGD),vertcat({'GSGD'},{'SGD'},{'Gvalidation'},{'Validation'}));
    
%     if bAddLegend
%         curLegendNameSGD = {'SGD E_i_n'};
%         legend(vertcat(hPrev, hCur, hCurSGD, hVal),vertcat(prevLegendNames, curLegendName, curLegendNameSGD, {'Validation'}));%, 'E_o_u_t')); 
%     end
    title({['GSGD/SGD-' cell2mat(iAlgo) ' (best results)'], file_used});
%>>

%Plot average of complete run for a given data set.
%<<
    avgFig = figure(avgFig);
    hCur=plot(1:T,microEin{1,2}, 'color', clr);
    set(hCur,'XData',1:T,'YData',microEin{1,2});
    hCurSGD = plot(microEin{4,2},microEin{3,2},'--m');
    set(hCurSGD,'XData',microEin{4,2},'YData',microEin{3,2});
    hVal=plot(1:T,microEin{2,2}, ':r');
    set(hVal,'XData',1:T,'YData',microEin{2,2});
    hValSGD=plot(1:T,microEin{5,2}, '-.b');
    set(hValSGD,'XData',1:T,'YData',microEin{5,2});

    drawnow

    xlabel('Epochs','fontsize',10,'color','b')
    ylabel('Error (E_i_n/E_v)','fontsize',10,'color','b')
    
    legend(vertcat(hCur, hCurSGD, hVal, hValSGD),vertcat({'GSGD'},{'SGD'},{'Gvalidation'},{'Validation'}));

% %     if bAddLegend
% %         curLegendNameSGD = {'SGD E_i_n'};
% %         legend(vertcat(hPrev, hCurav, hCurSGDav, hValav),vertcat(prevLegendNames, curLegendName, curLegendNameSGD, 'validation'));%, 'E_o_u_t')); 
% %     end
    title({['GSGD/SGD-' cell2mat(iAlgo) ' (average results)'], file_used});
%>>
end

%close open Figures
%<<

outFile = ['Result/' cell2mat(iAlgo) '/' file_used '_Best.fig' ];
savefig(bestFig, outFile);
close(bestFig);  

outFile = ['Result/' cell2mat(iAlgo) '/' file_used '_Avg.fig' ];
savefig(avgFig, outFile);
close(avgFig);

outFile = ['Result/' cell2mat(iAlgo) '/' file_used '_Digest.fig' ];
savefig(digestFig, outFile);
close(digestFig);   
%     set(groot,'ShowHiddenHandles','on')
%     c = get(groot,'Children');
%     delete(c)
%>>


fprintf('Wilcocon (two tail) test\n');
SD = getWilcoxonTest(fileData1, fileData2);
dlmwrite(outFileSGDIII,[-1 SD],'-append');
dlmwrite(outFileSGD,[-1 SD],'-append');


%compare digested/analysed files)
[SGDIIIwinsPC, SGDwinsPC, SGDIIIwin, SGDwins] = compareAlgorithms(SRsSGDIII, SRsSGD);
fprintf('Win %% for SGDIII: %0.1f (%d/%d)\n',SGDIIIwinsPC, SGDIIIwin, totalRun);
fprintf('Win %% for SGD: %0.1f (%d/%d)\n',SGDwinsPC, SGDwins, totalRun);

%save wins in the corresponding files
fill = zeros(1,AggregateAnalysisL-2);
aggregateValues = [-1 fill SGDIIIwinsPC]; 
dlmwrite(outFileSGDIII,aggregateValues,'-append');

fill = zeros(1,AggregateAnalysisLsgd-2);
aggregateValues = [-1 fill SGDwinsPC];
dlmwrite(outFileSGD,aggregateValues,'-append');

% % biSGD = EinSGDtail.getBestIdx;
% % fprintf('Min Ein for SGD: %0.4f\n', EinSGDtail.Ebests(biSGD));
% % fprintf('Min Eval for SGD: %0.4f\n', min(EvSGDtail));
% % 
% % biSGDIII = EinSGDIIItail.getBestIdx;
% % fprintf('Min Ein for SGDIII: %0.4f\n', EinSGDIIItail.Ebests(biSGDIII));
% % fprintf('Min Eval for SGDIII: %0.4f\n', min(EvSGDIIItail));

end % one dataset completed
end % one Algorithm completed
end % rope team size / neighborhood size ---- may delete it
end % end of function

%% analysis of the outputs
% SRs - Success Rates of all the runs
function [SRs, fileData, bestMicros] = aggregateAnalysis(filename,AggregateAnalysisL, fileMicroEin, maxCols, maxRows)
%aggregateAnalysis it creates a CSV file that stores the analysis of algorithm's 
%performance. The details are given below:
% 1st column: run No (one complete execution)
% 2nd - [end-2] columns: intermediate Ein at the given range. Like 1-20, 21-40... etc
% [end-1] column: final test data SR.
% [end] column: NFC (number of function calls)
% The bottom data are aggregate analysis where the first column starts with
% -1 and 0s are just place holders which doesn't indicate any information.
% The [end -1] from top to bottom shows: mean, meadian, best, worst and win%. 
% win% is total wins compared to other given algorithm.
% [end] column indicates NFC.

    [fid,msg] = fopen(filename,'r');

    if(fid<0)
        error(['Cannot open ' filename]); 
        exit
    end

    fprintf('file (%s) opened successfully\n', filename);
    fprintf('Please wait...\n');
    format = '';
    separator =',';
    data_type = '%g';
    for(f = 1:AggregateAnalysisL-1)
        format = [format data_type separator];
    end
    format = [format data_type];

    aggregateData = fscanf(fid, format, [AggregateAnalysisL inf]);
    fclose(fid);
    aggregateData = aggregateData';

    fill = zeros(4,AggregateAnalysisL-2);
    NFCs = aggregateData(:,end);
    SRs = aggregateData(:,end-1);
    fileData = aggregateData(:,2:end-2);
    aggregateData = [];
    
    bestEins = min(fileData');
    [maxSRs, bestNFC] = max(SRs);
    [minSRs, worstNFC] = min(SRs);
    
    aggSR = [mean(SRs); median(SRs); maxSRs; minSRs];
    aggNFC = [floor(mean(NFCs)); floor(median(NFCs)); NFCs(bestNFC); NFCs(worstNFC)];
    
    [minEin, bestNFC] = min(bestEins);
    [maxEin, worstNFC] = max(bestEins);
    aggEins = [mean(bestEins); median(bestEins); minEin; maxEin];
    aggNFC4Eins = [floor(mean(NFCs)); floor(median(NFCs)); NFCs(bestNFC); NFCs(worstNFC)];
    
    desc = {'mean';'meadian';'best';'worst'}; %cannot be added to file :(
    fprintf(['Statistical analysis:\n'...
    '%s:\t %0.4f@%d %0.4f@%d\n'...
    '%s: %0.4f@%d %0.4f@%d\n'...
    '%s:\t %0.4f@%d %0.4f@%d\n'...
    '%s:\t %0.4f@%d %0.4f@%d\n'], ...
        desc{1}, aggSR(1), aggNFC(1), aggEins(1), aggNFC4Eins(1),...
        desc{2}, aggSR(2), aggNFC(2), aggEins(2), aggNFC4Eins(2),...
        desc{3}, aggSR(3), aggNFC(3), aggEins(3), aggNFC4Eins(3),...
        desc{4}, aggSR(4), aggNFC(4), aggEins(4), aggNFC4Eins(4));
    
    aggregateValues = [-1*ones(4,1) fill aggSR aggNFC aggEins aggNFC4Eins]; 
    dlmwrite(filename,aggregateValues,'-append');
    
    %%% Micro Ein Process
    %<<
    data = cell(maxRows,1);
    fid = fopen(fileMicroEin);
    thisLine = fgetl(fid);
    r = 1;
    while ischar(thisLine)
        data{r,1} = sscanf(sprintf('%s,', thisLine), '%g,', [1, inf]);

        r = r+1;
        thisLine= fgetl(fid);
    end
    fclose(fid);
   
%     plotE
%     plotV
%     plotEsgd
%     plotEsgdGens
%     plotVsgd
%     plotEoutSR
%     plotEoutSRsgd
types = 6;    

    bestMicros = cell(types,2); %1 - best, 2 - average
    
    for i = 1:types
         bestMicros{i,1} = data{(bestNFC-1)*types+i};
         bestMicros{i,2} = zeros(1,maxCols);
    end   
    
    %find average 
    %<<
    R = floor(maxRows/types);
    for i = 1:R
        for j = 1:types
            sz = min( size(bestMicros{j,2},2) , size(data{(i-1)*types+j},2) ); 
            bestMicros{j,2} = bestMicros{j,2}(1:sz) + data{(i-1)*types+j}(1:sz);
        end
    end
    
    for i = 1:types
        bestMicros{i,2} = bestMicros{i,2}/R;
    end
    %>>
    
    fprintf('file (%s) analysed and saved successfully\n\n', filename);
end

%% returns significance difference value
function SD = getWilcoxonTest(fileData1, fileData2)
    
    [r1, c1] = size(fileData1);
    [r2, c2] = size(fileData2);
    r = min(r1,r2);
    c = min(c1,c2);
    
    SD = [];
    for i = 1:c
        SD = [SD signrank(fileData1(1:r,i),fileData2(1:r,i))]; 
    end

end

%%
function [SGDIIIwinsPC, SGDwinsPC, SGDIIIwins, SGDwins] = compareAlgorithms(SRsSGDIII, SRsSGD)
    SGDIIIwins = 0;
    SGDwins = 0;
    N = min(size(SRsSGDIII,1),size(SRsSGD,1));
    for i = 1:N
        if (SRsSGDIII(i)>SRsSGD(i))
            SGDIIIwins = SGDIIIwins+1;
        end
        
        if (SRsSGD(i)>SRsSGDIII(i))
            SGDwins = SGDwins+1;
        end
    end
    SGDIIIwinsPC = SGDIIIwins*100/N;
    SGDwinsPC = SGDwins*100/N;
end

%% Use before the algorithm starts to run in loops
function [bAddLegend, hPrev, prevLegend, curLegendName] = myUpdateLegend(algo)        
% % %     algo = mfilename;

    curLegendName = strcat(algo, {' Ein'});
    [bAddLegend, hPrev, prevLegend, curLegendName] = updateLegend(curLegendName);
end

%%
% Threshold for noise score.
function [threshold] = getNoiseScoreThreshold(v, fold)
    threshold = ceil(0.45*fold*(1/(1+exp(0.5*fold-v)))); 
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
% only maxNoisePerGen data are considered, rest aren't treated as noise. 
% previous worst ones are removed if deleteWorstNoisePermanently is true;
function [noiseIdx, noiseScore, noiseLevel] = noiseSorting(noiseIdx, noiseScore, ...
    noiseLevel, maxNoiseIdx, maxNoisePerGen, deleteWorstNoisePermanently, ranking, bRefresh, threshold)

    worstIdx = find(noiseIdx>=maxNoiseIdx);
    noiseScore(worstIdx) = -3; % or val-1 so that at least not picked here... floor(noiseScore(worstIdx)./2);
    noiseLevel(worstIdx) = 0; %floor(noiseLevel(worstIdx)./2);
    
    
    if (~deleteWorstNoisePermanently) %these idx >= maxNoiseId will remain same to be ignored in algorithm
        noiseIdx(worstIdx) = 0;
        worstIdx = [];
    end

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

    nval = nval(1:maxNoisePerGen);
    nidx = nidx(1:maxNoisePerGen);
    
    %PRECAUTION: if maxNoisePerGen is very large then it may include
    %non-inconstent data as well that have 0 value.
    for i = size(nval,2):-1:1 %fliplr(nidx)
        if(nval(i)<=threshold) % you may change 0 to a small threathold as well :)
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
    
    noiseIdx(finalIdx) = noiseIdx(finalIdx)+1;
    
    finalIdx = [finalIdx worstIdx]; %these indices would remain noisy for next generation.
    % finalIdx may have DUPLICATE idx because of worstIdx
    
    N = size(noiseIdx,2);
    [~,cleanIdx] = setdiff(1:N,finalIdx);

    if(ranking == 'S->L')
        noiseScore = rank1; %remove
        noiseLevel = rank2; %remove
    elseif(ranking == 'L->S')
        noiseLevel = rank1; %remove
        noiseScore = rank2; %remove
    end

    % This one will cater for dynamic change in environment of landscape of
    % the search space. For example one point is inconsitent at iteration p
    % but may not be inconsistent after iteration q.
    for i = cleanIdx
        if(noiseScore(i)>0)
            noiseLevel(i) = noiseLevel(i) - noiseLevel(i)/noiseScore(i);
            noiseScore(i) = noiseScore(i)-floor(noiseScore(i)/2); %%%%%%%%%%%%%%%%%%%%%%%%?thresh-1??% -1 is ok if not weighted. 
        else
            noiseLevel(i) = 0;
        end
        
        if(noiseIdx(i)>0)
            noiseIdx(i) = noiseIdx(i)-1;
            if(noiseIdx(i) == 0)
                noiseLevel(i) = 0;
                noiseScore(i) = 0;
            end
        end
    end     
    
    %After the assignment of noiseIdx score & level should be cleaned.
    if(bRefresh)
        noiseScore(1:N) = 0;
        noiseLevel(1:N) = 0;
        noiseIdx(1:N) = 0; %its conditional. So keeep it separate.
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

%% overall best and accumulation of Weights
% Since SGD slowly converges with fluctuations so we cannot rely on weight
% of any given generation. Hence we take consideration of TAIL_SIZE number
% of iterations (that consist of fluctuating - ups and downs error vals)
% and get the best Weight W. 
function [E, W, EinTail, EvTail] = overallBest(idx,N,x,y,W,activation,tmpGr, EinTail, EvTail, cvSets_v)
%     if(size(idx,2)<=EinTail.TAIL_SIZE)
        curtmpErr = getError(1:N,x,y,W, activation); %Ein
        EinTail = EinTail.accumulateTail(W,tmpGr,curtmpErr);
%     end

    E = [];
    if(isempty(idx)) %accumulation completed for now
        if(~isempty(EinTail.Wbests))
            W = EinTail.Wbests{end}; %its not overall best
            E = EinTail.Ebests(end);
        end   
        curtmpErr = getError(cvSets_v,x,y,W, activation); %Eval
        EvTail = [EvTail curtmpErr];
    end
end

%% 
function ready = ready4revisit(processedIdx, idx, GuidedSize)
    if(mod(size(processedIdx,2), GuidedSize)==0 || isempty(idx))
        ready = true;
    else
        ready = false;
    end
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
