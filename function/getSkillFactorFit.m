function [skillFactor, fitness, fit1, fit2] = getSkillFactorFit(dataX, dataY, pos, subset, particleLenMask)
    %% Compute task-specific fitness values and assign skill factors

    alpha1 = 0.999999;
    alpha2 = 0.9; % (Note: alpha2 is defined but not used)

    % Task 1: Evaluate using features restricted to the subset region
    % (pos .* subset .* particleLenMask) > 0.6 → Select positions in the subset with activation > 0.6
    [featureNum1, error1] = knn5foldFast(dataX, dataY, (pos .* subset .* particleLenMask) > 0.6);

    % Task 2: Evaluate using all selected features (no subset constraint)
    % (pos .* particleLenMask) > 0.6 → Select features where activation > 0.6
    [featureNum2, error2] = knn5foldFast(dataX, dataY, (pos .* particleLenMask) > 0.6);

    %% Compute fitness for each task
    % Fitness combines classification error (main objective) and feature count penalty (regularization)
    fit1 = alpha1 * error1 + (1 - alpha1) * (featureNum1 ./ size(dataX, 2));
    fit2 = alpha1 * error2 + (1 - alpha1) * (featureNum2 ./ size(dataX, 2));

    %% Determine skill factors (task assignment)
    % If fit1 <= fit2, assign to task 1; otherwise assign to task 2
    % Lower fitness indicates better performance
    skillFactor = fit1 <= fit2;

    %% Handle ties and balance task assignment
    % Compute fitness differences and sort them
    diff_fit1_fit2 = fit1 - fit2;
    [sorted_diff, sorted_indices] = sort(diff_fit1_fit2);

    % Split evenly: first half assigned to task 1, second half to task 2
    num_elements = length(sorted_indices);
    half_num_elements = floor(num_elements / 2);
    skillFactor(sorted_indices(1:half_num_elements)) = 1;
    skillFactor(sorted_indices(half_num_elements + 1:end)) = 0;

    %% Compute overall fitness
    % Each particle’s fitness corresponds to its assigned task
    fitness = fit1 .* skillFactor + fit2 .* ~skillFactor; % logical NOT used for task 2

end
