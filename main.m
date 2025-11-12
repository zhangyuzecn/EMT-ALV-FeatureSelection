clc;
clear;
%% Enable parallel processing
% CoreNum = 8; % Number of CPU cores to use (e.g., 8 cores)
% if isempty(gcp('nocreate')) % If no parallel pool is currently active
%     parpool(CoreNum); % Start a parallel pool with the specified number of workers
% end
rng('shuffle'); % Use the current time as the random seed for more randomness

dataList = {'SRBCT','Lymphoma','Leuk1', 'DLBCL','TOX_171','Brain1','Prostate6033','ALLAML','Nci9', 'Brain2','Prostate', 'Leuk2', '11Tumor', 'Lung2'};

for name = 1:numel(dataList)
    tic;
    annotation="(EMT-ALV,repeat10times,maxiters70)";
    dataName = dataList{name};
    load(['dataset/' dataName])
    %% Hyperparameters
    dataX = mapminmax(X', 0, 1)'; % Normalize X so that all values are scaled to the range [0, 1]
    dataY = Y;
    clear X Y;
    
    foldNum = 10; % Number of folds for cross-validation (e.g., 10)
    featureNum = size(dataX, 2); % Number of features
    task_num = 2; % Number of tasks
    popSize = min(2 * floor(featureNum / 20 / 2), 300); % Population size (limited to 300)
    beta = 9; % Threshold for determining changes in length (adaptation mechanism)
    fitnessFun = @fitnessKfold; % Fitness function handle
    disp([dataName ' data loaded successfully  ' num2str(toc)]) % Display message with elapsed time

    
    %% Initialization Based on SU Sorting     
    [newIDX,promisingFeature.KN_point,promisingFeature.weights] = featureSelect(dataX,dataY,dataName);
    promisingFeature.subset = false(1,size(dataX,2));
    promisingFeature.subset(1:promisingFeature.KN_point) = true;
    dataX = dataX(:,newIDX);
    promisingFeature.weights = promisingFeature.weights(newIDX);

    repeat=10;
    result = cell(1,repeat);
    for i = 1:repeat
        depen=cell(1,foldNum);
        indices = crossvalind('Kfold',dataY,foldNum);
        for fold = 1:foldNum 
            testX = dataX(indices == fold,:);
            testY = dataY(indices == fold,:); 
            trainX = dataX(indices ~= fold,:);
            trainY = dataY(indices ~= fold,:); 
            res = EMT_ALV(trainX, trainY, dataName,fold,promisingFeature,featureNum,popSize);
            [res.featureNum,res.acc] = test(trainX,trainY,testX,testY,res);   
            res2 = struct(); 
            res2.acc=res.acc;
            res2.featureNum=res.featureNum;
            res2.recordparticleFit=res.recordParticleFit;
            depen{fold}=res2;
            disp(strcat("EMT-ALV on, ", dataName,'run == ',num2str(i)," fold ==", num2str(fold),' ,feature num ==',num2str(res.featureNum),' ,test Acc ==',num2str(res.acc), ' time= ',datestr(now,31),' !!!!!!'));
        end
        result{i} = depen;
    end
           %% Plot convergence curve
    Each_Run_recordparticleFit = []; % Initialize as double
    recordparticleFit = cell(1, repeat); % Preallocate memory
    
    for run = 1:repeat % Perform 'repeat' runs of k-fold cross-validation; when repeat = 1, it means a single run
        res = result{run};
        recordparticleFit{run}.record = [];
        for fold = 1:foldNum % Perform k-fold cross-validation (e.g., 10 folds)
            recordparticleFit{run}.record(fold, :) = res{fold}.recordparticleFit;
        end
        Each_Run_recordparticleFit(run, :) = mean(recordparticleFit{run}.record); % Mean fitness per iteration for this run
    end
    
    Mean_Each_iter_Fit = mean(Each_Run_recordparticleFit); % Used for plotting the convergence curve
    
    total_time = toc;
    save(['result/' dataName], 'result', 'foldNum', 'repeat', 'total_time', 'annotation', 'Mean_Each_iter_Fit');
    
    %% Output bestAcc, meanAcc, and meanFeaNum across all repeated runs
    featureNum = zeros(foldNum, repeat); % Initialize matrix for storing the number of selected features
    acc = zeros(foldNum, repeat); % Initialize matrix for storing accuracy values
    
    for run = 1:repeat % Perform 'repeat' runs of k-fold cross-validation; when repeat = 1, it means a single run
        res = result{run}; % Read the results of the current run
        for fold = 1:foldNum % For each fold (each fold has one test set and nine training sets)
            featureNum(fold, run) = res{fold}.featureNum; % Store number of selected features for this fold's test set
            acc(fold, run) = res{fold}.acc; % Store accuracy for this fold's test set
        end
    end
        
    resFeatureNum = mean(featureNum);
    resMeanAcc = mean(acc);
    Best_Acc=max(resMeanAcc);
    disp([dataName, ' ---- bestAcc= ', num2str(Best_Acc),' ---- meanAcc=',num2str(mean(resMeanAcc)),' ---- meanFeaNum= ',num2str(mean(resFeatureNum))]);
end











