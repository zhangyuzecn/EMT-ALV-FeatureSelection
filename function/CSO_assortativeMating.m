function [V] = CSO_assortativeMating(V, pos, skillFactor, rmp, particleLenMask, fitness)

    Lambda = 0.6; % Weight factor for inter-task influence

    %% Divide particles into task1 and task2
    task1 = skillFactor == 1;
    task2 = skillFactor == 0;

    % Task1 pairing
    indices_task1 = find(task1);
    Task1_shuffled_indices = indices_task1(randperm(length(indices_task1)));
    if rem(length(Task1_shuffled_indices), 2) == 1
        Task1_shuffled_indices = Task1_shuffled_indices(1:end-1); % Remove one if odd
    end
    half = length(Task1_shuffled_indices) / 2;
    Task1_pairs = [Task1_shuffled_indices(1:half), Task1_shuffled_indices(half+1:end)];

    % Task2 pairing
    indices_task2 = find(task2);
    Task2_shuffled_indices = indices_task2(randperm(length(indices_task2)));
    if rem(length(Task2_shuffled_indices), 2) == 1
        Task2_shuffled_indices = Task2_shuffled_indices(1:end-1);
    end
    half = length(Task2_shuffled_indices) / 2;
    Task2_pairs = [Task2_shuffled_indices(1:half), Task2_shuffled_indices(half+1:end)];

    %% Compare fitness within each task to determine winners and losers
    try
        % Task1
        Task1_mask = fitness(Task1_pairs(:,1)) > fitness(Task1_pairs(:,2));
        Task1_losers = Task1_mask .* Task1_pairs(:,1) + ~Task1_mask .* Task1_pairs(:,2);
        Task1_winners = ~Task1_mask .* Task1_pairs(:,1) + Task1_mask .* Task1_pairs(:,2);

        % Task2
        Task2_mask = fitness(Task2_pairs(:,1)) > fitness(Task2_pairs(:,2));
        Task2_losers = Task2_mask .* Task2_pairs(:,1) + ~Task2_mask .* Task2_pairs(:,2);
        Task2_winners = ~Task2_mask .* Task2_pairs(:,1) + Task2_mask .* Task2_pairs(:,2);

    catch exception
        disp('Task1 length:');
        disp(length(indices_task1));
        disp('Task2 length:');
        disp(length(indices_task2));
        disp(exception.message);
    end

    %% Multi-task evolutionary update
    % Determine which losers undergo inter-task transfer
    Task1_transfer_bool = rand(size(Task1_losers,1),1) < rmp;
    Task2_transfer_bool = rand(size(Task2_losers,1),1) < rmp;

    % --- Task1: losers that transfer ---
    losers = Task1_losers(Task1_transfer_bool);
    winners = Task1_winners(Task1_transfer_bool);
    r1 = rand(length(losers), size(V,2));
    r2 = rand(length(winners), size(V,2));
    r3 = rand(length(losers), size(V,2));
    V(losers,:) = r1.*V(losers,:) ...
                + r2.*(pos(winners,:) - pos(losers,:)) ...
                + r3.*((1-Lambda)*mean(pos(Task1_winners,:)) + Lambda*mean(pos(Task2_winners,:)) - pos(losers,:));
    V(losers,:) = V(losers,:) .* particleLenMask(losers,:);

    % --- Task2: losers that transfer ---
    losers = Task2_losers(Task2_transfer_bool);
    winners = Task2_winners(Task2_transfer_bool);
    r1 = rand(length(losers), size(V,2));
    r2 = rand(length(winners), size(V,2));
    r3 = rand(length(losers), size(V,2));
    V(losers,:) = r1.*V(losers,:) ...
                + r2.*(pos(winners,:) - pos(losers,:)) ...
                + r3.*(Lambda*mean(pos(Task1_winners,:)) + (1-Lambda)*mean(pos(Task2_winners,:)) - pos(losers,:));
    V(losers,:) = V(losers,:) .* particleLenMask(losers,:);

    % --- Task1: losers that do not transfer ---
    losers = Task1_losers(~Task1_transfer_bool);
    winners = Task1_winners(~Task1_transfer_bool);
    r1 = rand(length(losers), size(V,2));
    r2 = rand(length(winners), size(V,2));
    r3 = rand(length(losers), size(V,2));
    V(losers,:) = r1.*V(losers,:) ...
                + r2.*(pos(winners,:) - pos(losers,:)) ...
                + r3.*(mean(pos(Task1_winners,:)) - pos(losers,:));
    V(losers,:) = V(losers,:) .* particleLenMask(losers,:);

    % --- Task2: losers that do not transfer ---
    losers = Task2_losers(~Task2_transfer_bool);
    winners = Task2_winners(~Task2_transfer_bool);
    r1 = rand(length(losers), size(V,2));
    r2 = rand(length(winners), size(V,2));
    r3 = rand(length(losers), size(V,2));
    V(losers,:) = r1.*V(losers,:) ...
                + r2.*(pos(winners,:) - pos(losers,:)) ...
                + r3.*(mean(pos(Task2_winners,:)) - pos(losers,:));
    V(losers,:) = V(losers,:) .* particleLenMask(losers,:);
end
