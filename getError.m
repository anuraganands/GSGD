%% get error
% for SGD idx is 1:N but for GSGD its random
function e = getError(idx,x,y,W,type)
    e = 0;
    if(strcmp(type,'cross-entropy'))
        for in = [idx] %take the full training sequence/data of this fold  
            logits = W'*x(:,in); % W'(1xd) * x(dx1) = logits(1x1); %intermediate output is known as logits in machine learning
%             D = -sign(logits)*D; %to cater large numbers 
            D =max(ceil(log10(abs(logits))),1)-2;
            if(D>=1)
                logits = logits/10^D;
            end
            
            e = e + log(1+exp(-y(in)*(logits)));   
        end                  
    elseif(strcmp(type,'softmax')) % W has d dimenstion (same as input) and NC possible outputs
        datasize = size(idx,2);        
        [d, NC] = size(W);
        labels = full(sparse([ 1:datasize+1],[y(idx)'+1;NC],1)); %groundTruth
        labels(end,:) = []; % labels (datasize x NC)
        
        lb = 1;
        for in = [idx] 
            logits = W'*x(:,in); % W'(NCxd) * x(dx1) = logits(NCx1); %intermediate output is known as logits in machine learning
            D = -max(logits); %to cater large numbers                       
        
            tmp = exp(logits+D)/sum(exp(logits+D));
            tmp = labels(lb,:)*tmp;
            lb = lb+1;
            tmp = 1/tmp;
            e = e + log(tmp);
        end
    else
        error('Error! Activation function not supported.\n');
        exit;
    end
    
    sz = max(size(idx));
    e = e/sz; %average error         
end