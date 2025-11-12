function [idx, KN_point, weights] = featureSelect(dataX, dataY, dataName)

    % SU: Calculate the correlation between each feature and the label
    SUDivideNum = 10;
    featureNum = size(dataX, 2);
    su = zeros(1, featureNum);
    for i = 1:featureNum
        su(i) = SU(dataX(:, i), dataY, SUDivideNum);
    end

    % Sort features by SU value in descending order
    [SUC, idx] = sort(su, 'descend'); % SUC is the sorted version of su
    weights = su;
    weights(isnan(weights)) = 0; % Replace NaN values with 0

    % Determine the knee point based on coordinates (threshold method)
    SU_Threshold = max(0.2 * SUC(1), SUC(round(featureNum / log2(featureNum) / 2)));
    KN_point = max(find(SUC > SU_Threshold));

    % Determine the knee point based on the distance between the curve and a straight line
    point = [1:featureNum; sort(weights, 'descend')]; % x-axis: feature index, y-axis: SU weight

    dis = zeros(1, featureNum); % Initialize distance array
    for i = 1:featureNum % Compute the distance from each curve point to the straight line
        dis(i) = abs(det([point(:, featureNum) - point(:, 1), point(:, i) - point(:, 1)])) / ...
                 norm(point(:, featureNum) - point(:, 1));
    end
    [~, KN_point] = max(dis); % KN_point is the index of the knee point

    disp([dataName ' : ', num2str(KN_point), '/', num2str(featureNum)]);

    % Plot the knee point figure
    figure;
    xlabel('Feature Index');
    ylabel('Weights (SU values)');
    title('Knee Point Plot');
    hold on;
    plot(point(1, :), point(2, :), 'b'); % Blue curve
    hold on;
    plot([1, featureNum], [max(weights), min(weights)], 'r--'); % Red dashed line
    hold on;
    text(KN_point, point(2, KN_point), ['  (' num2str(KN_point) ',' num2str(point(2, KN_point)) ')']);
    plot(KN_point, point(2, KN_point), 'r*'); % Mark the knee point with a red asterisk
    img = gcf; % Get current figure handle
    print(img, '-dpng', '-r600', ['result/' dataName 'KneePoint.png']); % Save as high-resolution PNG

end
