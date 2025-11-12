function [fitness,newSkillFactor] = getFitness(dataX,dataY,position,subset,skillFactor,particleLenMask)

    popSize = size(position,1);

    alpha = [0.999999 0.9];
    featureNum = zeros(popSize,1);
    error = zeros(popSize,1);
    fitness = zeros(popSize,1);
    newSkillFactor=skillFactor;
    task1 = newSkillFactor==1;
    task2 = newSkillFactor==0;

    [featureNum(task1),error(task1)] = knn5foldFast(dataX,dataY,(position(task1,:).*subset.*particleLenMask(task1,:))>0.6);
    [featureNum(task2),error(task2)] = knn5foldFast(dataX,dataY,(position(task2,:).*particleLenMask(task2,:))>0.6);
    fitness(task1) = alpha(1) * error(task1) + (1-alpha(1))*(featureNum(task1)./size(dataX,2));
    fitness(task2) = alpha(1) * error(task2) + (1-alpha(1))*(featureNum(task2)./size(dataX,2));

end

