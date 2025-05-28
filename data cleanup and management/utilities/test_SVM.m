clc; close all; clear;


% Add path if needed
    baseDir = fileparts(pwd); % go up from /utilities/
    dataDir = fullfile(baseDir, 'final files');
    
    trainFile = fullfile(dataDir, 'train_features_Q2.csv');
    testFile  = fullfile(dataDir, 'test_features_Q2.csv');
    
    % Load training data
    trainData = readtable(trainFile);

    % Identify numeric columns
    numericVars = varfun(@isnumeric, trainData, 'OutputFormat', 'uniform');
    numericData = trainData{:, numericVars};

    % LOCF imputation
    for i = 1:size(numericData,2)
        col = numericData(:,i);
        for j = 2:length(col)
            if isnan(col(j))
                col(j) = col(j-1);
            end
        end
        numericData(:,i) = col;
    end
    trainData{:, numericVars} = numericData;

    % Define features and target
    X = numericData(:, 1:end-3); % all features except last 3 columns
    y = numericData(:, end-1);   % second-to-last column is the target

    % Nested cross-validation
    outerCV = cvpartition(size(X,1), 'KFold', 5);
    mseOuter = zeros(outerCV.NumTestSets,1);

    for i = 1:outerCV.NumTestSets
        trainIdx = training(outerCV, i);
        testIdx  = test(outerCV, i);

        Xtrain = X(trainIdx,:);
        ytrain = y(trainIdx);
        Xtest  = X(testIdx,:);
        ytest  = y(testIdx);

        % Inner cross-validation for hyperparameter tuning
        innerCV = cvpartition(length(ytrain), 'KFold', 3);
        bestMSE = inf;
        bestParams = [];

        boxVals = logspace(-2, 2, 5);
        epsilonVals = [0.1, 0.5, 1];

        for box = boxVals
            for eps = epsilonVals
                mseInner = zeros(innerCV.NumTestSets, 1);

                for j = 1:innerCV.NumTestSets
                    innerTrainIdx = training(innerCV, j);
                    innerValIdx   = test(innerCV, j);

                    mdl = fitrsvm(Xtrain(innerTrainIdx,:), ytrain(innerTrainIdx), ...
                        'BoxConstraint', box, ...
                        'Epsilon', eps, ...
                        'KernelFunction', 'linear'); % or 'gaussian'

                    yPredVal = predict(mdl, Xtrain(innerValIdx,:));
                    mseInner(j) = mean((yPredVal - ytrain(innerValIdx)).^2);
                end

                meanMSE = mean(mseInner);
                if meanMSE < bestMSE
                    bestMSE = meanMSE;
                    bestParams = struct('Box', box, 'Epsilon', eps);
                end
            end
        end

        % Train on full outer training set with best hyperparameters
        finalModel = fitrsvm(Xtrain, ytrain, ...
            'BoxConstraint', bestParams.Box, ...
            'Epsilon', bestParams.Epsilon, ...
            'KernelFunction', 'linear');

        yPredTest = predict(finalModel, Xtest);
        mseOuter(i) = mean((yPredTest - ytest).^2);
    end

    fprintf('Nested CV MSE: %.4f ± %.4f\n', mean(mseOuter), std(mseOuter));