%%
% SR - Success Rate
% iteration - iteration when the best results was obtained

function [SR, nfc] = PrintFinalResults(gr, PocketGoodWeights, inputVal,givenOut, printData, type)
%     PocketGoodWeights
    W = nan; %for unacceptable results if Pocket is not filled.
    %Get the best found
    nfc = 0;
    if (iscell(PocketGoodWeights))
        for i = size(PocketGoodWeights,1):-1:1
            if(~isempty(PocketGoodWeights{i,1}))        
                W = PocketGoodWeights{i,1};
                nfc = PocketGoodWeights{i,3};
                break;
            end   
        end
    else
        W = PocketGoodWeights;
        nfc = 0;
    end
    
    
    % clc
    if(printData)
        gr
        W
    end

    xval = inputVal ; 
    

    if(strcmp(type,'cross-entropy'))
        predictedVal = W'*xval;
        predictedVal= horzcat(predictedVal', givenOut');
        
        totCorrect = 0;
        for p = predictedVal'
        %     s = W'*i; %(:,1)
            s = p(1);
            a = exp(s)/(1+exp(s));

            if(a<=0.5 && p(2) == -1 || a>0.5 && p(2) == 1)
                if(printData)
                    fprintf('[%.0f %.2f] - correct\n', p(2),a);
                end
                totCorrect = totCorrect + 1;
            else
                if(printData)
                    fprintf('[%.0f %.2f] - WRONG\n', p(2),a);
                end
            end
        end
    elseif(strcmp(type,'softmax'))
        predictedVal = W'*xval; %(dxT)' x (dxN) = (TxN)
% %         [~,idx] = max(a,[],1);                  
% %         predictedVal = idx - ones(1,max(size(idx)));           
        predictedVal= vertcat(predictedVal, givenOut); %(T+1)xN)
        
        totCorrect = 0;
        for p = predictedVal
            s = p(1:end-1);
            D = -max(s);
            a = exp(s+D)/sum(exp(s+D));
            [~,idx] = max(a);                  
            class = idx - 1;
            
            if(class == p(end))
                totCorrect = totCorrect + 1;
            else
                if(printData)
                    fprintf('[%.0f %.2f] - WRONG\n', p(end),a);
                end
            end
        end
    else
        error('Error! Activation function not supported.\n');
        exit;
    end
        
    SR = totCorrect/size(xval,2)*100;
    fprintf('Success Rate: %.2f found on iteration %.0f\n\n',SR,nfc);
end