function [H] = entropy(X,divideNum)
    dataMin = min(X);
    dataMax = max(X);
    pc = zeros(1,divideNum);
    for i = 1:divideNum
        Dmin = dataMin + (dataMax-dataMin)*(i-1)/divideNum;
        Dmax = dataMin + (dataMax-dataMin)*(i)/divideNum;
        index = (X>=Dmin) & (X<=Dmax);
        pc(i) = sum(index)/size(X,1);
    end
    pc(pc==0) = 1;
    H = -sum(log2(pc).*pc);
end

