%% SGD gradient calculation
function gr = getSGD(xi, yi, W, type)
    if(strcmp(type,'cross-entropy'))
        s = W'*xi;
        gr = - yi*xi/(1+exp(yi*s)); %W'*xi)); % W at t i.e. W(t)
    elseif(strcmp(type,'softmax'))
        s = W'*xi; % W'(NCxd) * x(dx1) = s(NCx1); 
        [d, NC] = size(W);
        D = -max(s);
        s = exp(s+D)/sum(exp(s+D)); % s(NCx1)
        
        datasize = size(xi,2);      
        labels = full(sparse([ 1:datasize+1],[yi'+1;NC],1)); %groundTruth
        labels(end,:) = []; % labels (datasize x NC)
        labels = labels';
        try
            gr = (s-labels)*xi';
            gr = gr';
        catch ME
            d
            NC
            fprintf('s: [%d %d]\n',size(s));
            fprintf('xi: [%d %d]\n',size(xi));  
        end
    else
        error('Error! Activation function not supported.\n');
        exit
    end
end