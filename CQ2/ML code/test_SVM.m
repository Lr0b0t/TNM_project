close all; clear;


%% Step 0: Load Training and Test Data

% Define base path to data files
baseDir = fileparts(pwd);                     % e.g., if in /utilities/
dataDir = fullfile(baseDir, 'final files');

% Define file paths
trainFile = fullfile(dataDir, 'train_features_Q2.csv');
testFile  = fullfile(dataDir, 'test_features_Q2.csv');

% Load training and test data tables
trainTable = readtable(trainFile);
testTable  = readtable(testFile);

% Remove ID column (assumed to be the first column)
trainTable(:,1) = [];
testTable(:,1) = [];

% Identify numeric columns (e.g., skip categorical if present)
numericVarsTrain = varfun(@isnumeric, trainTable, 'OutputFormat', 'uniform');
numericVarsTest  = varfun(@isnumeric, testTable,  'OutputFormat', 'uniform');

% Extract numeric matrices for modeling
numericDataTrain = trainTable{:, numericVarsTrain};
numericDataTest  = testTable{:, numericVarsTest};  % No missing values assumed

%% Step 1: Handle Missing Values in Training Set Only
[trainFilled, meanVec] = impute_data_simple(numericDataTrain, 'mean');

% No imputation needed for test set
testFilled = numericDataTest;

%% Step 2: Define Target Columns and Extract MMSE
numTargets = 3;

% Training set
X_train = trainFilled(:, 1:end - numTargets);        % Features
Y_all_train = trainFilled(:, end - numTargets + 1:end);
Y_train = Y_all_train(:, 2);                         % MMSE (second-to-last)

% Test set
X_test = testFilled(:, 1:end - numTargets);          % Features only
Y_all_test = testFilled(:, end - numTargets + 1:end);
Y_test = Y_all_test(:, 2);                           % MMSE


Y_test_baseline = testFilled(:, 3);
Y_train_baseline = trainFilled(:, 3);

%% Step 3: Normalize Features (Z-score using training stats only)
[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

%% Step 4: Run Nested Cross-Validation and Final Test
results = run_nested_cv_svm(X_train_norm, Y_train, X_test_norm, Y_test, Y_test_baseline);

%%


function results = run_nested_cv_svm(X_train, Y_train, X_test, Y_test, Y_test_baseline)
% RUN_NESTED_CV_SVM SVM regression with nested CV and full reporting (R^2 optimization)

    outerK = 5;
    innerK = 3;
    outerCV = cvpartition(size(X_train, 1), 'KFold', outerK);

    % Parameter grid
    C_values = [0.001 0.01 0.1 1 10 20 50 80 100 1000];
    sigma_values = [0.001 0.01 0.1 1 10 20 50 80 100 1000];
    epsilon_values = [0.1, 0.2, 0.5, 1];

    outerR2 = zeros(outerK, 1); % for info
    bestParamsList = cell(outerK, 1);
    allResults = cell(outerK, 1);

    for i = 1:outerK
        fprintf('\n====== Outer Fold %d/%d ======\n', i, outerK);

        trainIdx = training(outerCV, i);
        testIdx = test(outerCV, i);
        X_outerTrain = X_train(trainIdx, :);
        Y_outerTrain = Y_train(trainIdx);
        X_outerTest = X_train(testIdx, :);
        Y_outerTest = Y_train(testIdx);

        innerCV = cvpartition(size(X_outerTrain, 1), 'KFold', innerK);

        bestR2 = -Inf;  % We're maximizing R2 now!
        hyperparamTable = [];

        for c = C_values
            for sigma = sigma_values
                for eps = epsilon_values
                    r2_inner = zeros(innerK, 1);
                    rmse_inner = zeros(innerK, 1);
                    for j = 1:innerK
                        trIdx = training(innerCV, j);
                        valIdx = test(innerCV, j);

                        X_in = X_outerTrain(trIdx, :);
                        Y_in = Y_outerTrain(trIdx);
                        X_val = X_outerTrain(valIdx, :);
                        Y_val = Y_outerTrain(valIdx);

                        model = fitrsvm(X_in, Y_in, ...
                            'KernelFunction', 'rbf', ...
                            'BoxConstraint', c, ...
                            'KernelScale', sigma, ...
                            'Epsilon', eps, ...
                            'Standardize', false);

                        Y_pred = predict(model, X_val);

                        errors = Y_val - Y_pred;
                        rmse_inner(j) = sqrt(mean(errors.^2));
                        r2_inner(j) = 1 - sum(errors.^2) / sum((Y_val - mean(Y_val)).^2);
                    end
                    mean_rmse = mean(rmse_inner);
                    mean_r2 = mean(r2_inner);

                    newrow = table(c, sigma, eps, mean_rmse, mean_r2, ...
                        'VariableNames', {'C', 'Sigma', 'Epsilon', 'RMSE', 'R2'});
                    if isempty(hyperparamTable)
                        hyperparamTable = newrow;
                    else
                        hyperparamTable = [hyperparamTable; newrow];
                    end

                    % -- Maximize mean_r2 --
                    if mean_r2 > bestR2
                        bestR2 = mean_r2;
                        bestParams = struct('C', c, 'sigma', sigma, 'epsilon', eps);
                    end
                end
            end
        end

        allResults{i} = hyperparamTable;

        % Train best model on all outer train
        finalModel = fitrsvm(X_outerTrain, Y_outerTrain, ...
            'KernelFunction', 'rbf', ...
            'BoxConstraint', bestParams.C, ...
            'KernelScale', bestParams.sigma, ...
            'Epsilon', bestParams.epsilon, ...
            'Standardize', false);

        Y_outerPred = predict(finalModel, X_outerTest);

        % Info: compute R^2 on this fold
        errors = Y_outerTest - Y_outerPred;
        outer_r2 = 1 - sum(errors.^2) / sum((Y_outerTest - mean(Y_outerTest)).^2);
        outerR2(i) = outer_r2;
        bestParamsList{i} = bestParams;

        % Print summary for this fold
        fprintf('Best params for fold %d: C=%.4g, sigma=%.4g, epsilon=%.3f, R2=%.4f\n', ...
            bestParams.C, bestParams.sigma, bestParams.epsilon, bestR2);
        disp(sortrows(hyperparamTable, 'R2', 'descend')); % Show all sorted by R2 (descending)
    end

    % Find most common/frequent best hyperparameters
    Cs = cellfun(@(s) s.C, bestParamsList);
    sigmas = cellfun(@(s) s.sigma, bestParamsList);
    epsilons = cellfun(@(s) s.epsilon, bestParamsList);

    bestC = mode(Cs);
    bestSigma = mode(sigmas);
    bestEpsilon = mode(epsilons);

    fprintf('\n*** Most common selected params across folds: C=%.4g, sigma=%.4g, epsilon=%.3f ***\n', ...
        bestC, bestSigma, bestEpsilon);

    % Retrain final model on all training data
    finalModel = fitrsvm(X_train, Y_train, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', bestC, ...
        'KernelScale', bestSigma, ...
        'Epsilon', bestEpsilon, ...
        'Standardize', false);

    Y_test_pred = predict(finalModel, X_test);

    % Regression metrics
    errors = Y_test - Y_test_pred;
    mse  = mean(errors.^2);
    rmse = sqrt(mse);
    mae  = mean(abs(errors));
    r2   = 1 - sum(errors.^2) / sum((Y_test - mean(Y_test)).^2);

    fprintf('\nTest set regression metrics:\n');
    fprintf('  RMSE: %.4f\n', rmse);
    fprintf('  MAE : %.4f\n', mae);
    fprintf('  R^2 : %.4f\n', r2);

    % Store all results in struct
    results = struct();
    results.outerR2 = outerR2;
    results.bestParamsList = bestParamsList;
    results.allResults = allResults;
    results.finalParams = struct('C', bestC, 'sigma', bestSigma, 'epsilon', bestEpsilon);
    results.testRMSE = rmse;
    results.finalModel = finalModel;
    results.Y_test_pred = Y_test_pred;
    results.regression = struct('rmse', rmse, 'mae', mae, 'r2', r2);

    % Call the group jump reporter, if baseline available
    if nargin > 4 && ~isempty(Y_test_baseline)
        fprintf('\nTest set group-jump metrics (using last year MMSE as baseline):\n');
        group_jump_report(Y_test, Y_test_pred, Y_test_baseline);
    else
        fprintf('\n[Note: Group-jump metrics require baseline MMSE for the test set.]\n');
    end
end


% function results = run_nested_cv_svm(X_train, Y_train, X_test, Y_test, Y_test_baseline)
% % RUN_NESTED_CV_SVM SVM regression with nested CV and full reporting
% %
% % Inputs:
% %   X_train          - [n_train x p] normalized, imputed features (train set)
% %   Y_train          - [n_train x 1] MMSE target for training set
% %   X_test           - [n_test x p] normalized, imputed features (test set)
% %   Y_test           - [n_test x 1] MMSE target for test set (future)
% %   Y_test_baseline  - [n_test x 1] Baseline MMSE for test set (past year)
% %
% % Outputs:
% %   results: struct with metrics, models, and predictions
% 
%     % Set up outer and inner CV
%     outerK = 5;
%     innerK = 3;
%     outerCV = cvpartition(size(X_train, 1), 'KFold', outerK);
% 
%     % Parameter grid
%     C_values = [0.001 0.01 0.1 1 10 20 50 80 100 1000];
%     sigma_values = [0.001 0.01 0.1 1 10 20 50 80 100 1000];
%     epsilon_values = [0.1, 0.2, 0.5, 1];
% 
%     outerRMSE = zeros(outerK, 1);
%     bestParamsList = cell(outerK, 1);
%     allResults = cell(outerK, 1);
% 
%     for i = 1:outerK
%         fprintf('\n====== Outer Fold %d/%d ======\n', i, outerK);
% 
%         trainIdx = training(outerCV, i);
%         testIdx = test(outerCV, i);
%         X_outerTrain = X_train(trainIdx, :);
%         Y_outerTrain = Y_train(trainIdx);
%         X_outerTest = X_train(testIdx, :);
%         Y_outerTest = Y_train(testIdx);
% 
%         innerCV = cvpartition(size(X_outerTrain, 1), 'KFold', innerK);
% 
%         bestRMSE = Inf;
%         hyperparamTable = [];
% 
%         % Grid search over all parameter combinations
%         for c = C_values
%             for sigma = sigma_values
%                 for eps = epsilon_values
%                     rmse_inner = zeros(innerK, 1);
%                     for j = 1:innerK
%                         trIdx = training(innerCV, j);
%                         valIdx = test(innerCV, j);
% 
%                         X_in = X_outerTrain(trIdx, :);
%                         Y_in = Y_outerTrain(trIdx);
%                         X_val = X_outerTrain(valIdx, :);
%                         Y_val = Y_outerTrain(valIdx);
% 
%                         % Train SVM
%                         model = fitrsvm(X_in, Y_in, ...
%                             'KernelFunction', 'rbf', ...
%                             'BoxConstraint', c, ...
%                             'KernelScale', sigma, ...
%                             'Epsilon', eps, ...
%                             'Standardize', false);
% 
%                         Y_pred = predict(model, X_val);
%                         rmse_inner(j) = sqrt(mean((Y_val - Y_pred).^2));
%                     end
%                     mean_rmse = mean(rmse_inner);
% 
%                     % Log parameters and RMSE for this combo
%                     newrow = table(c, sigma, eps, mean_rmse, ...
%                         'VariableNames', {'C', 'Sigma', 'Epsilon', 'RMSE'});
%                     if isempty(hyperparamTable)
%                         hyperparamTable = newrow;
%                     else
%                         hyperparamTable = [hyperparamTable; newrow];
%                     end
% 
%                     % Update best params if improved
%                     if mean_rmse < bestRMSE
%                         bestRMSE = mean_rmse;
%                         bestParams = struct('C', c, 'sigma', sigma, 'epsilon', eps);
%                     end
%                 end
%             end
%         end
% 
%         allResults{i} = hyperparamTable;
% 
%         % Train best model on all outer train
%         finalModel = fitrsvm(X_outerTrain, Y_outerTrain, ...
%             'KernelFunction', 'rbf', ...
%             'BoxConstraint', bestParams.C, ...
%             'KernelScale', bestParams.sigma, ...
%             'Epsilon', bestParams.epsilon, ...
%             'Standardize', false);
% 
%         Y_outerPred = predict(finalModel, X_outerTest);
%         outerRMSE(i) = sqrt(mean((Y_outerTest - Y_outerPred).^2));
%         bestParamsList{i} = bestParams;
% 
%         % Print summary for this fold
%         fprintf('Best params for fold %d: C=%.4g, sigma=%.4g, epsilon=%.3f, RMSE=%.4f\n', ...
%             bestParams.C, bestParams.sigma, bestParams.epsilon, bestRMSE);
%         disp(sortrows(hyperparamTable, 'RMSE')); % Show all sorted by RMSE
%     end
% 
%     % Find most common/frequent best hyperparameters
%     Cs = cellfun(@(s) s.C, bestParamsList);
%     sigmas = cellfun(@(s) s.sigma, bestParamsList);
%     epsilons = cellfun(@(s) s.epsilon, bestParamsList);
% 
%     % Use mode of best params (most frequently chosen)
%     bestC = mode(Cs);
%     bestSigma = mode(sigmas);
%     bestEpsilon = mode(epsilons);
% 
%     fprintf('\n*** Most common selected params across folds: C=%.4g, sigma=%.4g, epsilon=%.3f ***\n', ...
%         bestC, bestSigma, bestEpsilon);
% 
%     % Retrain final model on all training data
%     finalModel = fitrsvm(X_train, Y_train, ...
%         'KernelFunction', 'rbf', ...
%         'BoxConstraint', bestC, ...
%         'KernelScale', bestSigma, ...
%         'Epsilon', bestEpsilon, ...
%         'Standardize', false);
% 
%     % Predict on final test set
%     Y_test_pred = predict(finalModel, X_test);
% 
%     % Regression metrics
%     errors = Y_test - Y_test_pred;
%     mse  = mean(errors.^2);
%     rmse = sqrt(mse);
%     mae  = mean(abs(errors));
%     r2   = 1 - sum(errors.^2) / sum((Y_test - mean(Y_test)).^2);
% 
%     fprintf('\nTest set regression metrics:\n');
%     fprintf('  RMSE: %.4f\n', rmse);
%     fprintf('  MAE : %.4f\n', mae);
%     fprintf('  R^2 : %.4f\n', r2);
% 
%     % Store all results in struct
%     results = struct();
%     results.outerRMSE = outerRMSE;
%     results.bestParamsList = bestParamsList;
%     results.allResults = allResults;
%     results.finalParams = struct('C', bestC, 'sigma', bestSigma, 'epsilon', bestEpsilon);
%     results.testRMSE = rmse;
%     results.finalModel = finalModel;
%     results.Y_test_pred = Y_test_pred;
%     results.regression = struct('rmse', rmse, 'mae', mae, 'r2', r2);
% 
%     % Call the group jump reporter, if baseline available
%     if nargin > 4 && ~isempty(Y_test_baseline)
%         fprintf('\nTest set group-jump metrics (using last year MMSE as baseline):\n');
%         group_jump_report(Y_test, Y_test_pred, Y_test_baseline);
%     else
%         fprintf('\n[Note: Group-jump metrics require baseline MMSE for the test set.]\n');
%     end
% end




%% this last section is kept for functions

function [filledData, imputerInfo] = impute_data_simple(data, method, varargin)
%IMPUTE_DATA_SIMPLE Imputes missing values in a numeric matrix using simple methods.
%
%   [filledData, imputerInfo] = impute_data_simple(data, method)
%   [filledData, imputerInfo] = impute_data_simple(data, method, trainInfo)
%
%   Inputs:
%       data        : Numeric matrix with missing values (NaNs).
%       method      : String specifying the imputation method:
%                       'mean' - fills each column's NaNs with its mean
%                       'knn'  - uses KNN imputation (requires Bioinformatics Toolbox)
%       trainInfo   : Optional. For 'mean' method, supply a vector of means
%                     (used for imputing test data using training statistics).
%
%   Outputs:
%       filledData  : Data matrix with NaNs imputed.
%       imputerInfo : For 'mean' method, this is a vector of column means used.
%                     For 'knn', this is empty ([]).
%
%   Example:
%       [trainImputed, meanVec] = impute_data_simple(trainData, 'mean');
%       testImputed = impute_data_simple(testData, 'mean', meanVec);

    % Check inputs
    if nargin > 2
        info = varargin{1};  % Precomputed mean (for test set)
    else
        info = [];
    end

    switch lower(method)
        case 'mean'
            if isempty(info)
                % Compute column-wise mean excluding NaNs
                colMeans = mean(data, 'omitnan');
            else
                colMeans = info;  % Use supplied means (e.g., from training data)
            end

            filledData = data;
            for j = 1:size(filledData,2)
                nanIdx = isnan(filledData(:,j));
                filledData(nanIdx,j) = colMeans(j);
            end
            imputerInfo = colMeans;

        case 'knn'
            % KNN imputation using Bioinformatics Toolbox
            try
                filledData = knnimpute(data')';  % knnimpute works on features in rows
            catch ME
                error('KNN imputation requires the Bioinformatics Toolbox.\nError: %s', ME.message);
            end
            imputerInfo = [];  % No extra info needed for KNN reuse

        otherwise
            error('Unsupported method: %s. Use ''mean'' or ''knn''.', method);
    end
end




function group_jump_report(Y_true, Y_pred, Y_baseline, cutpoints)
%GROUP_JUMP_REPORT - Print group and jump metrics for predicted MMSE
% cutpoints: vector of group thresholds (e.g. [0 24 27 30.1])
% Group labels: e.g. {'Significant','Mild','Normal'}

    if nargin < 4
        cutpoints = [0 24 27 30.1]; % Default for MMSE
    end
    grouplabels = {'Significant','Mild','Normal'};
    true_group = discretize(Y_true, cutpoints, 'categorical', grouplabels);
    pred_group = discretize(Y_pred, cutpoints, 'categorical', grouplabels);
    base_group = discretize(Y_baseline, cutpoints, 'categorical', grouplabels);

    % Overall group accuracy
    group_acc = mean(true_group == pred_group);
    fprintf('Group accuracy: %.2f%%\n', 100*group_acc);

    % Confusion matrix
    fprintf('Confusion matrix (true vs predicted group):\n');
    disp(crosstab(true_group, pred_group));

    % Jumps: declining to a lower group
    actual_jump = double(true_group) < double(base_group);
    pred_jump  = double(pred_group) < double(base_group);

    % Sensitivity and precision for detecting jumpers
    sensitivity = sum(actual_jump & pred_jump) / max(sum(actual_jump),1); % avoid /0
    precision   = sum(actual_jump & pred_jump) / max(sum(pred_jump),1);

    fprintf('Sensitivity (recall) for group decline: %.2f%%\n', 100*sensitivity);
    fprintf('Precision (PPV) for group decline: %.2f%%\n', 100*precision);

    % Optionally return metrics, confusion matrix, etc.
end