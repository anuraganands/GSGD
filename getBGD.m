%% BGD gradient calculation
function gr = getBGD(N, d, x, y, W)
    gr = zeros(d,1);
    for i = 1:N
        gri = -y(i)*x(:,i)/(1+exp(y(i)*W'*x(:,i))); % W at t i.e. W(t)
        gr = gr + gri;
    end 
    gr = (1/N)*gr;
end