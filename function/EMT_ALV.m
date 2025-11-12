function [result] = EMT_ALV(data, label, dataName, fold, promisingFeature, featureNum, popSize)
    % Initialize swarm, personal bests, global best, and related parameters
    [swarm, pBest, gBest, skillFactor, fitness] = init( ...
        featureNum, popSize, promisingFeature.KN_point, ...
        promisingFeature.subset, promisingFeature.weights, ...
        data, label);

    %% Hyperparameters
    maxIters = 70; % Maximum number of iterations
    V = zeros(popSize, featureNum); % Initialize velocity matrix
    Vlimit = [0.4, 0.6]; % Velocity bounds for two types of features (absolute value limits)
    RMP = 0.6; % Random mating probability (controls knowledge transfer)
    c1 = 1.49445; 
    c2 = 1.49445; 
    c3 = 1.49445;
    maxValue = swarm.MaxValue; % Position boundary
    subset = promisingFeature.subset; % Feature subset to the left of KN_point
    gBest.notChange = 0; % Count how many iterations gBest has not improved
    beta = 9; % Trigger length-change mechanism if gBest hasn’t improved for 9 iterations

    %% Evolution process
    recordParticleFit = []; % Initialize fitness record array
    for iter = 1:maxIters
        position = swarm.position; % Current particle positions
        W = 0.9 - 0.5 * (iter / maxIters); % Inertia weight decreases linearly

        % Update velocity using the CSO (Competitive Swarm Optimization) strategy
        V = CSO_assortativeMating(V, position, skillFactor, RMP, swarm.particleLenMask, fitness);

        %% Velocity boundary control (based on KN_point)
        k = promisingFeature.KN_point;
        % Limit the first k feature velocities to ±0.4
        V(:, 1:k) = max(min(V(:, 1:k), Vlimit(1)), -Vlimit(1));
        % Limit the remaining feature velocities to ±0.6
        V(:, k+1:end) = max(min(V(:, k+1:end), Vlimit(2)), -Vlimit(2));

        %% Position boundary control
        position(position < 0) = 0;
        position(position > maxValue) = maxValue(position > maxValue);
        position = position .* swarm.particleLenMask; % Adjust for particles with different lengths

        %% Update pBest and gBest
        [fitness, skillFactor] = getFitness(data, label, position, subset, skillFactor, swarm.particleLenMask);
        pBest = updatePBest(pBest, position, fitness, subset, skillFactor, swarm.particleLenMask);
        [gBest, flag] = updateGBest(gBest, position, fitness, subset, skillFactor, swarm.particleLenMask);
        % 'flag' indicates whether gBest was updated (true = improved)

        swarm.position = position;
        swarm.fitness = fitness;

        %% Apply the length-change mechanism if gBest stagnates
        if flag == true
            gBest.notChange = 0;
        else
            gBest.notChange = gBest.notChange + 1;
            if mod(gBest.notChange, beta) == 0
                % Trigger length-change process
                swarm = lengthChange(swarm, data, label);
                position = swarm.position;

                % Recalculate fitness and update bests after length change
                [fitness, skillFactor] = getFitness(data, label, position, subset, skillFactor, swarm.particleLenMask);
                pBest = updatePBest(pBest, position, fitness, subset, skillFactor, swarm.particleLenMask);
                [gBest, flag] = updateGBest(gBest, position, fitness, subset, skillFactor, swarm.particleLenMask);
                if flag == true
                    gBest.notChange = 0;
                end
                swarm.fitness = fitness;
            end
        end

        % Record gBest fitness value for each iteration
        recordParticleFit(iter) = gBest.fit;
    end

    %% Save results
    result.fold = fold;
    result.name = dataName;
    result.pbest = pBest;
    result.gbest = gBest;
    result.recordParticleFit = recordParticleFit;
end
