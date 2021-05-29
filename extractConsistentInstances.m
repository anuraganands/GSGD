%%
function [consistentIdx] = extractConsistentInstances ...
    (avgE, pe, omPlusScore, omPlusLevel, omMinuScore,...
    omMinusLevel)
     
    omMidx = find(omMinusLevel>0);
    omPidx = find(omPlusLevel>0);
    avgMscore = sum(omMinusLevel(omMidx))/size(omMidx,2);
    avgPscore = sum(omPlusLevel(omPidx))/size(omPidx,2);
    
    omMidx = find(omMinusLevel>0.5*avgMscore);
    omPidx = find(omPlusLevel>0.5*avgPscore);
        
    if(avgE<pe)%if good  
        consistentIdx = omMidx;
    else
        consistentIdx = omPidx;% others may be good but because of the ith instances we are getting poor result. 
    end

end