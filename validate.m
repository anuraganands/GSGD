%% Validation with Pocket best weights so far
function [PocketGoodWeights, doTerminate, SR, E] = validate(PocketGoodWeights,inputVal,W,givenOut, nfc, type)
    xval = inputVal ; 
    doTerminate = false;
    
    if(strcmp(type,'cross-entropy'))
        predictedVal = W'*xval;
        predictedVal= horzcat(predictedVal', givenOut');
    
        totCorrect = 0;
        for p = predictedVal'
            s = p(1);
            a = exp(s)/(1+exp(s));
            if(a<=0.5 && p(2) == -1 || a>0.5 && p(2) == 1)
                totCorrect = totCorrect + 1;
            end
        end
    elseif(strcmp(type,'softmax')) % W has d dimenstion (same as input) and NC possible outputs
        predictedVal = W'*xval; %(dxT)' x (dxN) = (TxN)
%         [~,idx] = max(a,[],1);                  
%         predictedVal = idx - ones(1,max(size(idx)));           
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
            end
        end
    else
        error('Error! Activation function not supported.\n');
        exit;
    end

    SR = totCorrect/size(xval,2);
    if(SR > 0.5) % change to 0.5 - 0.95 and stop at appropriate             
        if(isempty(PocketGoodWeights{ceil(SR*10)-size(PocketGoodWeights,1),1}))
            PocketGoodWeights{ceil(SR*10)-size(PocketGoodWeights,1),1} = W;
            PocketGoodWeights{ceil(SR*10)-size(PocketGoodWeights,1),2} = SR;
            PocketGoodWeights{ceil(SR*10)-size(PocketGoodWeights,1),3} = nfc;
        else 
            if(PocketGoodWeights{ceil(SR*10)-size(PocketGoodWeights,1),2}<SR)
                PocketGoodWeights{ceil(SR*10)-size(PocketGoodWeights,1),1} = W;
                PocketGoodWeights{ceil(SR*10)-size(PocketGoodWeights,1),2} = SR;
                PocketGoodWeights{ceil(SR*10)-size(PocketGoodWeights,1),3} = nfc;
            end
        end
        if(SR>110.85)  % percentage defined after decimal. <85% would be 0.85>              
            doTerminate = true;
        else
            false;
        end
    end
    
    
    N = size(inputVal,2);
    E = 0;
    for in = 1:N        
        nErr = getError(in,inputVal,givenOut,W,type);  %check this???       
        E = E+nErr; 
    end                  

    E = E/N;
%     fprintf('Error value: %0.2f\n',E);
    
    
end