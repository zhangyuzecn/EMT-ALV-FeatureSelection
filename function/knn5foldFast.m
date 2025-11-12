function [featureNum, error] = knn5foldFast(dataX, dataY, flag)
    fold = 5;
    popSize = size(flag, 1); % Number of individuals
    featureNum = sum(flag, 2); % Number of selected features per individual
    maxFeatureNum = max(featureNum); % Maximum number of selected features among all individuals
    error = ones(popSize, 1); % Initialize error vector

    % Generate 5-fold cross-validation indices
    indices = crossvalind('Kfold', dataY, fold);

    for j = 1:popSize
        if featureNum(j) ~= 0
            RD = 0; % Initialize accumulated error
            % Compute pairwise distances between all samples using selected features
            Dis = pdist2(dataX(:, flag(j, :)), dataX(:, flag(j, :)));
            Dis(logical(eye(size(Dis)))) = inf; % Exclude self-distance (set diagonal to infinity)

            for i = 1:fold
                % Define training and test indices for the current fold
                trainXIndex = indices ~= i;
                trainYIndex = indices ~= i;
                testXIndex  = indices == i;
                testYIndex  = indices == i;

                % Find the nearest neighbor (K = 1) for each test sample
                [~, minIndex] = min(Dis(testXIndex, trainXIndex), [], 2);

                % Get predicted labels using the nearest neighbor
                pre = dataY(trainYIndex);
                pre = pre(minIndex);

                % True labels for test samples
                y = dataY(testYIndex);

                % Compute per-class accuracy
                tbl = tabulate(y); % Returns [class, count, proportion]
                tbl(tbl(:,2) == 0, :) = []; % Remove empty classes
                TPR = 0;
                for k = tbl(:,1)' % Iterate over all classes
                    index = (k == y);
                    TPR = TPR + sum(pre(index) == y(index)) / sum(index);
                end

                % Accumulate misclassification rate
                RD = RD + (1 - TPR / size(tbl, 1));
            end

            % Compute average classification error across folds
            error(j) = RD / fold;
        else
            % If no features are selected, assign the maximum feature number
            featureNum(j) = maxFeatureNum;
        end
    end
end
