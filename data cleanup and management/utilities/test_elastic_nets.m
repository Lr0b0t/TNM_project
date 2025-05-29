clc; close all; clear;


% Setup paths
    baseDir = fileparts(pwd); % go up from /utilities/
    dataDir = fullfile(baseDir, 'final files');
    trainFile = fullfile(dataDir, 'train_features_Q2.csv');

    % Load training data
    trainData = readtable(trainFile);

    % LOCF Imputation (Last Observation Carried Forward)
    numericVars = varfun(@isnumeric, trainData, 'OutputFormat', 'uniform');
    numericData = trainData{:, numericVars};
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
    X = numericData(:, 1:end-3);     % All but last 3 columns
    y = numericData(:, end-1);       % Second-to-last column is target

    % Outer cross-validation
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

        % Grid search over alpha (L1/L2 mix) and lambda (penalty)
        alphaVals = [0.1, 0.5, 0.9];
        lambdaVals = logspace(-4, 0, 10);

        for alpha = alphaVals
            for lambda = lambdaVals
                mseInner = zeros(innerCV.NumTestSets,1);
                for j = 1:innerCV.NumTestSets
                    iTrain = training(innerCV,j);
                    iVal = test(innerCV,j);

                    mdl = fitrlinear(Xtrain(iTrain,:), ytrain(iTrain), ...
                        'Learner', 'leastsquares', ...
                        'Regularization', 'elasticnet', ...
                        'Alpha', alpha, ...
                        'Lambda', lambda);

                    yValPred = predict(mdl, Xtrain(iVal,:));
                    mseInner(j) = mean((yValPred - ytrain(iVal)).^2);
                end
                meanMSE = mean(mseInner);
                if meanMSE < bestMSE
                    bestMSE = meanMSE;
                    bestParams = struct('Alpha', alpha, 'Lambda', lambda);
                end
            end
        end

        % Final Elastic Net model with best hyperparameters
        finalModel = fitrlinear(Xtrain, ytrain, ...
            'Learner', 'leastsquares', ...
            'Regularization', 'elasticnet', ...
            'Alpha', bestParams.Alpha, ...
            'Lambda', bestParams.Lambda);

        % Predict and evaluate
        yPred = predict(finalModel, Xtest);
        mseOuter(i) = mean((yPred - ytest).^2);
    end

    fprintf('Elastic Net Nested CV MSE: %.4f ± %.4f\n', mean(mseOuter), std(mseOuter));