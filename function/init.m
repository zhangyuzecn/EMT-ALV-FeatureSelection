function [swarm, pBest, gBest, skillFactor, fitness] = init(featureNum, popSize, KN_point, subset, weight, data, label)
    %% Initialize the swarm
    maxLength = featureNum; % Maximum individual length
    PopIdx = 1:popSize; % Index of each particle in the population

    %% Variable-length particle initialization
    % Dense sequence (first 90% of the population)
    dense_sequence = normrnd(KN_point, KN_point / 4, [round(popSize * 0.9), 1]); % Normal distribution (mean, std)
    dense_sequence = max(abs(min(dense_sequence, featureNum)), 1); % Ensure values stay within [1, featureNum]

    % Sparse sequence (remaining 10% of the population)
    sparse_sequence = unifrnd(1, featureNum, [popSize - round(popSize * 0.9), 1]); % Uniform distribution

    % Merge the two sequences
    sequence = [dense_sequence; sparse_sequence];

    % Sort and round to obtain integer particle lengths
    sorted_sequence = round(sort(sequence));
    swarm.particleLen = sorted_sequence;

    % Initialize binary mask for particle lengths
    swarm.particleLenMask = zeros(popSize, featureNum);
    for i = 1:popSize
        cur_Length = swarm.particleLen(i); % Current particle length
        swarm.particleLenMask(i, 1:cur_Length) = 1; % Set first cur_Length elements to 1
    end

    %% Initialize positions based on feature relevance
    norm_weights = mapminmax(weight, 0, 1); % Normalize weights to [0, 1]
    % Initialize positions: add random noise in [-0.5, 0.5] around normalized weights
    swarm.position = max(min((norm_weights(1:featureNum) + rand(popSize, featureNum) - 0.5), 1), 0);

    %% Define position boundary (MaxValue)
    LeftMaxValue = ones(popSize, KN_point); % Left part has a maximum of 1
    RightMaxValue = ones(popSize, featureNum - KN_point) - 0.3; % Right part has a maximum of 0.7
    swarm.MaxValue = [LeftMaxValue, RightMaxValue];

    %% Apply mask to positions (account for different particle lengths)
    swarm.position = swarm.position .* swarm.particleLenMask;

    %% Initialize gBest and pBest
    % Obtain initial fitness and skill factors for all particles
    [skillFactor, fitness, fit1, fit2] = getSkillFactorFit(data, label, swarm.position, subset, swarm.particleLenMask);

    % Mask out irrelevant features for task 1
    swarm.position(skillFactor == 1, :) = swarm.position(skillFactor == 1, :) .* subset;

    % Find best-performing individuals for task 1 and task 2
    [minFit1, index1] = min(fit1(skillFactor == 1));
    [minFit2, index2] = min(fit2(skillFactor == 0));

    % Initialize personal best (pBest) for each task
    pBest.task1.pos = swarm.position(index1, :);
    pBest.task2.pos = swarm.position(index2, :);
    pBest.task1.index = index1;
    pBest.task1.fit = fit1(index1);
    pBest.task2.fit = fit2(index2);
    pBest.task1.mask = subset .* swarm.particleLenMask(index1, :);
    pBest.task2.mask = swarm.particleLenMask(index2, :);

    % Initialize global best (gBest)
    gBest.task1.pos = pBest.task1.pos;
    gBest.task2.pos = pBest.task2.pos;
    gBest.task1.index = pBest.task1.index;
    gBest.task1.fit = pBest.task1.fit;
    gBest.task2.fit = pBest.task2.fit;
    gBest.task1.mask = pBest.task1.mask;
    gBest.task2.mask = pBest.task2.mask;

    % Select the overall global best between the two tasks
    if minFit1 <= minFit2
        gBest.pos = gBest.task1.pos;
        gBest.fit = gBest.task1.fit;
        gBest.index = index1;
        gBest.mask = gBest.task1.mask;
    else
        gBest.pos = gBest.task2.pos;
        gBest.fit = gBest.task2.fit;
        gBest.index = index2;
        gBest.mask = gBest.task2.mask;
    end
end
