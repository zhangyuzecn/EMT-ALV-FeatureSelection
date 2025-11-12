function [pBest] = updatePBest(pBest,pos,fitness,subset,skillFactor,particleLenMask)
 
    for i = 1:numel(fitness)
        if skillFactor(i) == 1
            if fitness(i) < pBest.task1.fit
                pBest.task1.pos = pos(i,:);
                pBest.task1.fit = fitness(i);
                pBest.task1.mask = subset.*particleLenMask(i,:);
            end
        else
             if fitness(i) < pBest.task2.fit
                pBest.task2.pos = pos(i,:);
                pBest.task2.fit = fitness(i);
                pBest.task2.mask = particleLenMask(i,:);
            end
        end
    end

end

