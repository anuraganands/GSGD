%% all variations of the SGD algorithm
function [W, s, r, gradHist, Whistory, m, v] = SGDvariation(t, x, y, W, eta, activation, ...
    algo, s, r, gradHist, Whistory, m, v)
    beta = 0.9;
    epsilon = 1e-8;

    gr = getSGD(x, y, W, activation);

    if(strcmp(algo,'Canonical'))
        W = W-eta*gr;
    elseif(strcmp(algo,'Momentum'))
        bapplyMomentum = false;
        if(isempty(Whistory{1}) && isempty(Whistory{2})) 
            Whistory{1} = W;
        elseif(isempty(Whistory{2}))
            Whistory{2} = W;
        else % 2
            Whistory{1} = Whistory{2}; %Wt-2
            Whistory{2} = W; %Wt-1
            bapplyMomentum = true;
        end
        
        if(bapplyMomentum)
            rho = 0.9;
            W = Whistory{2}-eta*gr - rho*(Whistory{2}-Whistory{1});%momentum SGD
        else
            W = W-eta*gr;%simple SGD
        end
    elseif(strcmp(algo,'Nesterov'))
        bapplyNesterov = false;
        if(isempty(Whistory{1}) && isempty(Whistory{2})) 
            Whistory{1} = W;
        elseif(isempty(Whistory{2}))
            Whistory{2} = W;
        else % 2
            Whistory{1} = Whistory{2}; %Wt-2
            Whistory{2} = W; %Wt-1
            bapplyNesterov = true;
        end
        
        if(bapplyNesterov)
            rho = 0.9;
            gr = getSGD(x, y, W-rho*Whistory{2}, activation);
            W = Whistory{2}-eta*gr - rho*(Whistory{2}-Whistory{1});%nesterov SGD
        else
            gr = getSGD(x, y, W, activation);
            W = W-eta*gr;%simple SGD
        end
    elseif(strcmp(algo,'Adagrad'))
        gradHist = gradHist + gr.^2;
        W = W - eta * gr ./ (sqrt(gradHist) + epsilon); 
        
    elseif(strcmp(algo,'Adadelta'))
        if(t==1)
            r = gr.^2;
        else
            r = beta * r + (1-beta)* gr.^2;  
        end  
        
        v1 = - (sqrt(s + epsilon)./sqrt(r + epsilon)) .* gr;
        % update accumulated updates (deltas)
        %out
        s = beta * s + (1-beta)* v1.^2;
        %out
        W = W + v1; 

    elseif(strcmp(algo,'RMSprop'))
        if(t==1)
            r = gr.^2;
        else
            r = beta * r + (1-beta)* gr.^2;  
        end
        W = W - eta * gr ./ sqrt(r + epsilon); 
    elseif(strcmp(algo,'Adam'))
        beta1 = 0.9;
        beta2 = 0.999;
        
        m = beta1.*m + (1 - beta1).*gr;
        % Update biased 2nd raw moment estimate
        v = beta2.*v + (1 - beta2).*(gr.^2);

        % Compute bias-corrected 1st moment estimate
        mHat = m./(1 - beta1^t);
        % Compute bias-corrected 2nd raw moment estimate
        vHat = v./(1 - beta2^t);

        % Update decision variables
        W = W - eta.*mHat./(sqrt(vHat) + epsilon);
    else %error
        W = [];
        s = [];
        gradHist = [];
        warning('Incorrect choice of algorithm in SGDvariation!\n');
        exit
    end
            
end