%% Use before the algorithm starts to run in loops
function [bAddLegend, hPrev, prevLegendNames, curLegendName] = updateLegend(curLegendName)
    allLegendNames = getAllalgosNames(); 
    arg = cell(0);
    sz = size(allLegendNames,2);
    i = 1;
    for nm = allLegendNames
        arg(end+1) = {'DisplayName'};
        arg(end+1) = nm;
        if(i<sz)
            arg(end+1) = {'-or'};
        end
        i = i+1;
    end
        
    hPrev = findobj(gca,'Type','line',arg); %This one will be vertically concatenated in caller with new hCur.
% % %     hPrev = findobj(gca,'Type','line', 'DisplayName', allLegendNames{1}, '-or', ...
% % %     'DisplayName', allLegendNames{2},  '-or', 'DisplayName', allLegendNames{3}, '-or',...
% % %     'DisplayName', allLegendNames{4}); 

    prevLegendNames = cell(0);
    if ~isempty(hPrev)
        [~,u] = unique({hPrev.DisplayName});% to get only unique legends, change it to [~,u]
        hPrev = hPrev(u);
        prevLegendNames = {hPrev(:).DisplayName};
    end
    
    %check if the new one already exists:
    ind = strfind(prevLegendNames,cell2mat(curLegendName));
    ind = find(not(cellfun('isempty',ind)));
    
    bAddLegend = false;
    if(isempty(ind))
        bAddLegend = true;
    end
    
    prevLegendNames = prevLegendNames';
end