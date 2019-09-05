%% makeCrossValidationSets
function [ cvSets ] = makeCrossValidationSets( rawDataSizeN, fold, bRandomize )
% prepares n-fold cross validation sets. Output is set of indices of data samples.  
N =rawDataSizeN;
from = 1;
cvSets = cell(1,fold);

for v = 1:fold
    to = v*round(N/fold);
    range = [from: to];
    cvSets{1,v} = range;
    from = to+1;
end
%last one
%collect
lastRange = cvSets{1,fold};
j = size(lastRange,2);
diff = N-lastRange(j);
unAssigned = [];
for i=1:diff %not assigned
    unAssigned = [unAssigned lastRange(j)+i];
end
% unAssigned
j = 1;
for i = unAssigned
    m = cvSets{1,j};
    m = [m i];
    cvSets{1,j} = m;
end

%remove
j = 1;
lastRange = cvSets{1,fold};
for i = lastRange
   if(i>N);
       lastRange(j) = [];
       j= j-1;
   end
   j = j+1;
end
cvSets{1,fold} = lastRange;
% lastRange

if(bRandomize)
    rnd = randperm(N);
    fr = 1;
    to = 0;
    for f = 1:fold
        to = to+size(cvSets{1,f},2);
        cvSets{1,f} = rnd(fr:to);
        fr = to+1;
    end
end
%Printing only: COMMENT IT OUT
%     for i = 1: fold
%         fid = 1;
%         fprintf('(%d) ',i);
%         fprintf(fid, [repmat(' %d ', 1, size(cvSets{1,i},2)) '\n'], cvSets{1,i}');    
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%


end
