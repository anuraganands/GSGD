%%
% Threshold for noise score.
function [threshold] = getNoiseScoreThreshold(t, T, ropeTeamSz)
%     threshold = ceil(0.45*fold*(1/(1+exp(0.5*fold-v)))); 
 threshold = 1; %max(ceil(0.5*ropeTeamSz*t/T*(1/(1+exp(0.5*T-t)))),1);
end