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