%% Collect Inconsistent Instances
% In this function only collect relevant information are being collected as 
% it is yet not known which ones are consistent or inconsistant. 
%  omPlusScore: score when individual error ascends
%  omPlusLevel: level when individual error ascends
%  omMinuScore: score when individual error descends
%  omMinusLevel: level when individual error descends 
function [omPlusScore, omPlusLevel, omMinuScore, omMinusLevel, tmpGuided] = ...
    collectInconsistentInstances (idx,x,y,W,activation, ropeTeamSz, pe)

    omPlusScore = zeros(1,ropeTeamSz);
    omPlusLevel = zeros(1,ropeTeamSz); 
    omMinuScore = zeros(1,ropeTeamSz);
    omMinusLevel = zeros(1,ropeTeamSz);  

    tmpGuided = 0;
%     for in =  idx % its never empty so tmpGuided > 0         
%         if tmpGuided >= ropeTeamSz
%             break;
%         end
%         tmpGuided = tmpGuided + 1;
% %             nErr = getError(in,x,y,W,activation);
% %             e = e+nErr;
        for j = 1:min(ropeTeamSz,size(idx,2))
            tmpGuided = tmpGuided + 1;
            nErr = getError(idx(j),x,y,W,activation);
            %vector direction may be different but atleast getting lower error
            %check with avgError of previous iteration.
            %objective is to separate consistent with inconsistent
            if(nErr>pe)% of course, not applicable for the first index. 
                omPlusScore(j) = omPlusScore(j)+1;
                omPlusLevel(j) = omPlusLevel(j) + (nErr-pe);% presumably not contributing .. om+
            else
                omMinuScore(j) = omMinuScore(j)+1;
                omMinusLevel(j) = omMinusLevel(j) + (pe-nErr);% presumably contributing ..om-
            end 
        end 
%     end  %end for loop     
  
end