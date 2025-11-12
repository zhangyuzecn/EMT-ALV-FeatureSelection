function [su] = SU(X,Y,divideNum)
    HF = entropy(X,divideNum);
    tdl = tabulate(Y);
    pc = tdl(:,2) / sum(tdl(:,2));
    HC = -sum(log2(pc)./pc);
    IG = HF - conditionEntropy(X,Y,divideNum);
    su = IG/(HF+HC);
end

