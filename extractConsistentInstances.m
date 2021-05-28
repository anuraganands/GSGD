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