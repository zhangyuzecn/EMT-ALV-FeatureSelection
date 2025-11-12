function [H] = conditionEntropy(X,Y,divideNum)
    dataMin = min(X);
    dataMax = max(X);
    Yindex = unique(Y);
    pc = [];
    for i = 1:divideNum
        Dmin = dataMin + (dataMax-dataMin)*(i-1)/divideNum;
        Dmax = dataMin + (dataMax-dataMin)*(i)/divideNum;
        index = (X>=Dmin) & (X<=Dmax);
        ptemp = [];
        for j = Yindex'
            ptemp = [ptemp sum(Y(index)==j)];
        end
        pc = [pc;ptemp];
    end
    pc = pc/size(X,1);
    X_Y = pc./sum(pc);
    X_Y(X_Y==0) = 1;
    H = sum(sum(-log2(X_Y).* pc));  
end

