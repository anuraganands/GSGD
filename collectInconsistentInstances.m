%% Collect Inconsistent Instances
%  omPlusScore = otherAffectedPointsScore;
%  omPlusLevel = otherAffectedPointsLevel
%  omMinuScore = otherAffectedPointsScore;
%  omMinusLevel = otherAffectedPointsLevel 
function [omPlusScore, omPlusLevel, omMinuScore, omMinusLevel] = ...
    collectInconsistentInstances (omPlusScore, omPlusLevel, omMinuScore,...
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