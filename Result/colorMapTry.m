
%% How to Run:
% 1 - simply choose "TestResults" folder. 
function colorMapTry()
    clear all
    clc

    d = uigetdir(pwd, 'Select a folder');
    files = dir(fullfile(d, '*aggEin.csv'));
   
    COL = 1001;
    % 1 - ro value,
    % 2-1001 - error diff, 
    % odd rows are best error diff
    % even rows are average error diff
    

    total = size(files,1);
    maxRun = 50;
    types = 2;
    totalRuns = (maxRun-2+1)*types;
    fprintf(['file name']);
    
    for(i = 1:total)
%         fprintf('proccessing file %s\n',files(i).name);
        [fid,msg] = fopen(files(i).name,'r');

        HL = 0;  %ignore header lines (first few lines)
        HC = 0;  %ignore columns (first few columns)
        result = textscan(fid, '', 'HeaderLines', HL, 'HeaderColumns', HC, 'Delimiter', ',');
        fclose(fid);
     
        newResult = cell(totalRuns,1);
        for r = 1:totalRuns
            for c = 1:COL
                newResult{r} = [newResult{r} result{c}(r)];
            end
        end
        
        eAvgDelta=[];
        eBestDelta=[];
        SR = [];
        totalIterationPlot = 1000;
        av = 1;
        bs = 1;
        sr = 1;
        for r = 1: totalRuns
            if(mod(r,types) == inf) 
                tmp = newResult{r}(2:end);
                SR(sr,:) = tmp(1:totalIterationPlot);
%                 eAvgDelta(k,:) = norm_scale01(eAvgDelta(k,:));
                sr = sr+1;
            elseif(mod(r,types) == 1)% odd - best
                tmp = newResult{r}(2:end);
                eBestDelta(bs,:) = tmp(1:totalIterationPlot);
%                 eAvgDelta(k,:) = norm_scale01(eAvgDelta(k,:));
                bs = bs+1;
            elseif(mod(r,types) == 0) % even - average
                tmp = newResult{r}(2:end);
                eAvgDelta(av,:) = tmp(1:totalIterationPlot);
%                 eAvgDelta(k,:) = norm_scale01(eAvgDelta(k,:));
                av = av+1; 
            end
        end
              
        x = 1:totalIterationPlot;
        y = [2:maxRun];
        [X,Y] = meshgrid(x,y);
        Z = eAvgDelta;
%         Z = SR;
%         Z = eBestDelta;
%         surf(X,Y,Z)
        cMap=jet(256);
        figure('Name','Average Figure');
        [c,h] = contourf(X,Y,Z);
        set(h, 'edgecolor','none');       
        colormap(cMap);
%         colormap(gray)
        colorbar
%         colorbar('Ticks',[-0.06,0,0.03],...
%             'TickLabels',{'GSGD','<->','SGD'})
        xlabel('Iterations','fontsize',10,'color','b')
        ylabel('{\rho}','fontsize',10,'color','b')
        title(['Effect of {\rho} on Ein_G_S_G_D - Ein_S_G_D (' strtok(files(i).name, '_')  ')']);
% %         figure('Name','Best Figure');
% %         Z = eBestDelta;
% %         [c,h] = contourf(X,Y,Z);
% %         set(h, 'edgecolor','none');       
% %         colormap(cMap);
% %         colorbar

%<<option - 2      
% %         newpoints = 100;
% %         x = 1:totalIterationPlot;
% %         y = [2:maxRun];
% %         z = eAvgDelta;
% %         [xq,yq] = meshgrid(...
% %             linspace(min(min(x,[],2)),max(max(x,[],2)),newpoints ),...
% %             linspace(min(min(y,[],1)),max(max(y,[],1)),newpoints )...
% %           );
% %         BDmatrixq = interp2(x,y,z,xq,yq); %,'cubic');
% %         [c,h]=contourf(xq,yq,BDmatrixq,5); %,50);  
% %           colorbar;
%>>

%<<option - 3
%      f = figure;
%      ax = axes('Parent',f);
%      newpoints = 100;
%         x = 1:totalIterationPlot;
%         y = [2:maxRun];
%         z = eAvgDelta;
%         [xq,yq] = meshgrid(...
%             linspace(min(min(x,[],2)),max(max(x,[],2)),newpoints ),...
%             linspace(min(min(y,[],1)),max(max(y,[],1)),newpoints )...
%           );
%       BDmatrixq = interp2(x,y,z,xq,yq,'cubic');
%      h = surf(xq,yq,BDmatrixq,'Parent',ax);
%      set(h, 'edgecolor','none');
%      view(ax,[0,90]);
%      colormap(jet(256));
%      colorbar;
%>>
        
        
%<<other useful code        
% % %         contourf(peaks)
% % % colorbar('Ticks',[-5,-2,1,4,7],...
% % %          'TickLabels',{'Cold','Cool','Neutral','Warm','Hot'})       
        
% % % % %         for r = 1: totalRuns
% % % % %             if(mod(r,2) == 0) % even - average
% % % % %                 eAvgDelta = newResult{r};
% % % % %                 gens = 1:size(eAvgDelta,1)-1;
% % % % %                 mymap = [ro' gens' eAvgDelta']; 
% % % % %                 rgbplot(mymap)
% % % % %                 hold on
% % % % %                 colormap(mymap)
% % % % %             end
% % % % %         end
        

    end
end


