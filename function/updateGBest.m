function [gBest,flag] = updateGBest(gBest,pos,fitness,subset,skillFactor,particleLenMask)
    flag = false;
    i = skillFactor==1;
    fit1 = fitness;
    fit2 = fitness;
    fit1(~i) = 1;
    fit2(i) = 1;
    [~,index1] = min(fit1);
    [~,index2] = min(fit2);
    if gBest.task1.fit > fit1(index1)
        gBest.task1.pos = pos(index1,:); 
        gBest.task1.fit = fit1(index1);     
        gBest.task1.mask = subset.*particleLenMask(index1,:);
        gBest.task1.index=index1;
    end
    if gBest.task2.fit > fit2(index2)
       gBest.task2.pos = pos(index2,:); 
       gBest.task2.fit = fit2(index2);
       gBest.task2.mask = particleLenMask(index2,:);
       gBest.task2.index=index2;
    end
    if fit1(index1)<gBest.fit || fit2(index2)<gBest.fit
        if fit1(index1)<=fit2(index2)
            gBest.pos=gBest.task1.pos;
            gBest.fit=gBest.task1.fit; 
            gBest.index=index1;
            gBest.mask=gBest.task1.mask;
        else
            gBest.pos=gBest.task2.pos;
            gBest.fit=gBest.task2.fit; 
            gBest.index=index2;
            gBest.mask=gBest.task2.mask;
        end
        flag = true;
    end
end

