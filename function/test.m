function [featureNum, Acc] = test(trainX, trainY, testX, testY, result)
try
    % Select features: those with gbest position * mask > 0.6
    % 0.6 is the threshold; 'feature' is a 1*5469 boolean array
    feature = (result.gbest.pos .* result.gbest.mask) > 0.6;

    % Train 1-NN classifier using selected features
    Mdl = fitcknn(trainX(:, feature), trainY, NumNeighbors=1);

    % Count the number of selected features
    featureNum = sum(feature);

    % Predict labels on test set using selected features
    class = predict(Mdl, testX(:, feature));

    % Initialize performance evaluation
    cp = classperf(testY);

    % Number of unique classes
    num_class = size(unique(testY, 'stable'), 1);

    % Update class performance with predictions
    classperf(cp, class);

    % Number of misclassified samples per class
    error_distribution = cp.ErrorDistributionByClass;

    % Number of samples per class in the test set
    sample_distribution = cp.SampleDistributionByClass;

    % Accuracy per class
    acc_distribution = (sample_distribution - error_distribution) ./ sample_distribution;

    % Average accuracy across all classes
    Acc = sum(acc_distribution) / num_class;

catch exception
    % Display diagnostic information in case of error
    disp('num_class:');
    disp(num_class);
    disp('error_distribution:');
    disp(error_distribution);
    disp('sample_distribution:');
    disp(sample_distribution);
    disp('acc_distribution:');
    disp(acc_distribution);
    disp('Acc:');
    disp(Acc);
    disp('Error occurred during feature selection and model training/prediction:');
    disp(exception.message);
end
