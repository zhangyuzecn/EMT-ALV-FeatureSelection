function [cur_position] = TournamentSelection(j,selectedSwarmLength,bestFitIndex,bestFitLength,swarm)

    indices = find(j <= selectedSwarmLength);
    numIndices = numel(indices);
    
    if numIndices > 1
        selectedIndices = randsample(indices, 2);
    elseif numIndices == 1
        selectedIndices = [indices,indices];
    end
    fitnessValues = swarm.fitness(selectedIndices);
    [~, minIndex] = min(fitnessValues);
    selectedIndex = selectedIndices(minIndex);
    if bestFitLength>=j
        selectedIndex=randsample([selectedIndex,bestFitIndex],1);
    end
    cur_position=swarm.position(selectedIndex,j);
end

