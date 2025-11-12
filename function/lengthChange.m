function [swarm] = lengthChange(swarm, dataX, dataY)
    %% Select 30% of particles to check whether their lengths should change
    popSize = size(swarm.position, 1); % population size
    Dimension = size(dataX, 2); % number of dimensions
    selectedChanegeSwarmIdx = 1:popSize; % not randomly selected

    %% Lengths of the selected particles
    selectedSwarmLength = swarm.particleLen(selectedChanegeSwarmIdx);

    %% Length changes for the selected particles that need to be checked
    [bestFitValue, bestFitIndex] = min(swarm.fitness); % find particle with minimum fitness
    bestFitLength = swarm.particleLen(bestFitIndex); % length of particle with best fitness

    % Two cases:
    % When shortening: if particle length is longer than best particle, use selectedSwarmLength - bestFitLength;
    % otherwise, use the second formula.
    % When lengthening: if particle length is shorter than best particle, use bestFitLength - selectedSwarmLength (abs is necessary);
    % otherwise, use the second formula.
    deltaLength_Long = ceil((0.1 + rand(size(selectedSwarmLength)) * 0.5) .* abs(selectedSwarmLength - bestFitLength));
    deltaLength_Short = ceil((0 + rand(size(selectedSwarmLength)) * 0.1) .* selectedSwarmLength);
    deltaLength_Short_a = ceil((rand(size(selectedSwarmLength)) * 0.1) .* selectedSwarmLength);

    count = 0; % record number of times long < short
    count2 = 0;

    for i = 1:length(selectedChanegeSwarmIdx)
        curSelectedIdx = selectedChanegeSwarmIdx(i);

        %% Compute average fitness of particles longer than the current particle
        isLong = swarm.particleLen > selectedSwarmLength(i); % find particles longer than current (boolean)
        countLonger = sum(isLong == 1);
        AvgLongerFit = sum(isLong .* swarm.fitness) / countLonger; % average fitness of longer particles

        %% Compute average fitness of particles shorter than the current particle
        isShort = ~isLong; % particles shorter than current (boolean)
        countShorter = sum(isShort == 1);
        AvgShorterFit = sum(isShort .* swarm.fitness) / countShorter; % average fitness of shorter particles

        %% Decide whether to lengthen or shorten
        % Lower fitness is better (error rate)
        % If the longest particle has AvgLongerFit empty, it should be shortened.
        % If the shortest particle has AvgShorterFit empty, it should be lengthened.
        if isnan(AvgShorterFit) || AvgLongerFit < AvgShorterFit % longer is better
            % Lengthen
            count = count + 1;
            if selectedSwarmLength(i) < bestFitLength % current particle shorter than best fitness particle
                swarm.particleLen(curSelectedIdx) = selectedSwarmLength(i) + deltaLength_Long(i);
            else
                swarm.particleLen(curSelectedIdx) = min(selectedSwarmLength(i) + deltaLength_Short_a(i), max(swarm.particleLen));
            end
            % Update mask
            swarm.particleLenMask(curSelectedIdx, swarm.particleLen(curSelectedIdx):selectedSwarmLength(i)) = 1; 
            % set columns from original length to new length as 1
            for j = selectedSwarmLength(i):swarm.particleLen(curSelectedIdx)
                try
                    if (selectedSwarmLength(i) == 0 || isnan(selectedSwarmLength(i))) % unlikely due to minimum length set to 1
                        selectedSwarmLength(i) = 1;
                        cur_position = TournamentSelection(selectedSwarmLength(i), selectedSwarmLength, bestFitIndex, bestFitLength, swarm);
                    else
                        cur_position = TournamentSelection(j, selectedSwarmLength, bestFitIndex, bestFitLength, swarm);
                    end
                    swarm.position(curSelectedIdx, j) = cur_position;
                catch exception
                    disp('j');
                    disp([num2str(j)]);
                    disp('selectedSwarmLength');
                    disp([size(selectedSwarmLength)]);
                    disp(selectedSwarmLength);
                    disp('i');
                    disp([num2str(i)]);
                    disp(exception.message);
                end
            end

        elseif isnan(AvgLongerFit) || AvgShorterFit < AvgLongerFit % shorter is better
            count2 = count2 + 1;
            % Shorten
            if selectedSwarmLength(i) > bestFitLength % current particle longer than best fitness particle
                swarm.particleLen(curSelectedIdx) = selectedSwarmLength(i) - deltaLength_Long(i);
            else
                swarm.particleLen(curSelectedIdx) = selectedSwarmLength(i) - deltaLength_Short(i);
            end
            swarm.particleLen(curSelectedIdx) = max(swarm.particleLen(curSelectedIdx), 1);
            % Update mask
            swarm.particleLenMask(curSelectedIdx, swarm.particleLen(curSelectedIdx):selectedSwarmLength(i)) = 0; 
            % set columns from new length to original length as 0
            % Update position
            swarm.position(curSelectedIdx, :) = swarm.position(curSelectedIdx, :) .* swarm.particleLenMask(curSelectedIdx, :);
        end
    end
end
