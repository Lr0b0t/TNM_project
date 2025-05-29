close all; clear;


% rng(42, 'twister');
% TODO: write about ε, test teh jumping groups function

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

results_rbf = run_nested_cv_svm(X_train_norm, Y_train, X_test_norm, Y_test, Y_test_baseline, 'rbf');
results_linear = run_nested_cv_svm(X_train_norm, Y_train, X_test_norm, Y_test, Y_test_baseline, 'linear');
results_poly = run_nested_cv_svm(X_train_norm, Y_train, X_test_norm, Y_test, Y_test_baseline, 'polynomial');

%%
function results = run_nested_cv_svm(X_train, Y_train, X_test, Y_test, Y_test_baseline, kernelType)
% RUN_NESTED_CV_SVM
%   Trains SVM regression with nested CV and reports detailed metrics.
%   Selects hyperparameters by maximizing mean R² in inner CV.

    if nargin < 6 || isempty(kernelType)
        kernelType = 'rbf';
    end

    outerK = 5;
    innerK = 3;
    outerCV = cvpartition(size(X_train, 1), 'KFold', outerK);

    C_values = [0.01, 0.1, 1, 10, 50, 100, 1000];
    epsilon_values = [ 0.05, 0.1, 0.2, 0.5, 1];

    if strcmp(kernelType, 'rbf')
        sigma_values = logspace(-3, 3, 7);
    else
        sigma_values = 1;
    end

    if strcmp(kernelType, 'polynomial')
        poly_orders = [2 3 4 5];
    else
        poly_orders = 3;
    end

    outerR2 = zeros(outerK, 1);
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
        bestR2 = -Inf;
        hyperparamTable = [];

        for c = C_values
            for eps = epsilon_values
                switch kernelType
                    case 'rbf'
                        for sigma = sigma_values
                            [mean_rmse, mean_r2] = svm_inner_cv_r2(X_outerTrain, Y_outerTrain, innerCV, kernelType, c, sigma, [], eps);
                            newrow = table(c, sigma, eps, mean_rmse, mean_r2, ...
                                'VariableNames', {'C', 'Sigma', 'Epsilon', 'RMSE', 'R2'});
                            hyperparamTable = [hyperparamTable; newrow];
                            if mean_r2 > bestR2
                                bestR2 = mean_r2;
                                bestParams = struct('C', c, 'sigma', sigma, 'epsilon', eps);
                            end
                        end
                    case 'linear'
                        [mean_rmse, mean_r2] = svm_inner_cv_r2(X_outerTrain, Y_outerTrain, innerCV, kernelType, c, [], [], eps);
                        newrow = table(c, eps, mean_rmse, mean_r2, ...
                            'VariableNames', {'C', 'Epsilon', 'RMSE', 'R2'});
                        hyperparamTable = [hyperparamTable; newrow];
                        if mean_r2 > bestR2
                            bestR2 = mean_r2;
                            bestParams = struct('C', c, 'epsilon', eps);
                        end
                    case 'polynomial'
                        for order = poly_orders
                            [mean_rmse, mean_r2] = svm_inner_cv_r2(X_outerTrain, Y_outerTrain, innerCV, kernelType, c, [], order, eps);
                            newrow = table(c, order, eps, mean_rmse, mean_r2, ...
                                'VariableNames', {'C', 'PolyOrder', 'Epsilon', 'RMSE', 'R2'});
                            hyperparamTable = [hyperparamTable; newrow];
                            if mean_r2 > bestR2
                                bestR2 = mean_r2;
                                bestParams = struct('C', c, 'PolyOrder', order, 'epsilon', eps);
                            end
                        end
                end
            end
        end

        allResults{i} = hyperparamTable;

        % Train best model on all outer train
        switch kernelType
            case 'rbf'
                finalModel = fitrsvm(X_outerTrain, Y_outerTrain, ...
                    'KernelFunction', 'rbf', ...
                    'BoxConstraint', bestParams.C, ...
                    'KernelScale', bestParams.sigma, ...
                    'Epsilon', bestParams.epsilon, ...
                    'Standardize', false);
            case 'linear'
                finalModel = fitrsvm(X_outerTrain, Y_outerTrain, ...
                    'KernelFunction', 'linear', ...
                    'BoxConstraint', bestParams.C, ...
                    'Epsilon', bestParams.epsilon, ...
                    'Standardize', false);
            case 'polynomial'
                finalModel = fitrsvm(X_outerTrain, Y_outerTrain, ...
                    'KernelFunction', 'polynomial', ...
                    'PolynomialOrder', bestParams.PolyOrder, ...
                    'BoxConstraint', bestParams.C, ...
                    'Epsilon', bestParams.epsilon, ...
                    'Standardize', false);
        end

        Y_outerPred = predict(finalModel, X_outerTest);
        errors = Y_outerTest - Y_outerPred;
        outer_r2 = 1 - sum(errors.^2) / sum((Y_outerTest - mean(Y_outerTest)).^2);
        outerR2(i) = outer_r2;
        bestParamsList{i} = bestParams;
        continue;
        % Print best R2 and parameters for this fold
        switch kernelType
            case 'rbf'
                fprintf('Best params for fold %d: C=%.4g, sigma=%.4g, epsilon=%.3f, R2=%.4f\n', ...
                    bestParams.C, bestParams.sigma, bestParams.epsilon, bestR2);
            case 'linear'
                fprintf('Best params for fold %d: C=%.4g, epsilon=%.3f, R2=%.4f\n', ...
                    bestParams.C, bestParams.epsilon, bestR2);
            case 'polynomial'
                fprintf('Best params for fold %d: C=%.4g, PolyOrder=%d, epsilon=%.3f, R2=%.4f\n', ...
                    bestParams.C, bestParams.PolyOrder, bestParams.epsilon, bestR2);
        end
        disp(sortrows(hyperparamTable, 'R2', 'descend'));
    end

    % Determine best hyperparameters
    Cs = cellfun(@(s) s.C, bestParamsList);
    epsilons = cellfun(@(s) s.epsilon, bestParamsList);

    if strcmp(kernelType, 'rbf')
        sigmas = cellfun(@(s) s.sigma, bestParamsList);
        bestC = mode(Cs);
        bestEpsilon = mode(epsilons);
        bestSigma = mode(sigmas);
    elseif strcmp(kernelType, 'linear')
        bestC = mode(Cs);
        bestEpsilon = mode(epsilons);
        bestSigma = [];
    elseif strcmp(kernelType, 'polynomial')
        polyorders = cellfun(@(s) s.PolyOrder, bestParamsList);
        bestC = mode(Cs);
        bestEpsilon = mode(epsilons);
        bestSigma = [];
        bestPolyOrder = mode(polyorders);
    end

    % Retrain final model on full training data
    switch kernelType
        case 'rbf'
            finalModel = fitrsvm(X_train, Y_train, ...
                'KernelFunction', 'rbf', ...
                'BoxConstraint', bestC, ...
                'KernelScale', bestSigma, ...
                'Epsilon', bestEpsilon, ...
                'Standardize', false);
        case 'linear'
            finalModel = fitrsvm(X_train, Y_train, ...
                'KernelFunction', 'linear', ...
                'BoxConstraint', bestC, ...
                'Epsilon', bestEpsilon, ...
                'Standardize', false);
        case 'polynomial'
            finalModel = fitrsvm(X_train, Y_train, ...
                'KernelFunction', 'polynomial', ...
                'PolynomialOrder', bestPolyOrder, ...
                'BoxConstraint', bestC, ...
                'Epsilon', bestEpsilon, ...
                'Standardize', false);
    end

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

    % Save results
    results = struct();
    results.outerR2 = outerR2;
    results.bestParamsList = bestParamsList;
    results.allResults = allResults;
    if strcmp(kernelType, 'rbf')
        results.finalParams = struct('C', bestC, 'sigma', bestSigma, 'epsilon', bestEpsilon);
    elseif strcmp(kernelType, 'linear')
        results.finalParams = struct('C', bestC, 'epsilon', bestEpsilon);
    else
        results.finalParams = struct('C', bestC, 'PolyOrder', bestPolyOrder, 'epsilon', bestEpsilon);
    end
    results.testRMSE = rmse;
    results.finalModel = finalModel;
    results.Y_test_pred = Y_test_pred;
    results.regression = struct('rmse', rmse, 'mae', mae, 'r2', r2);

    % Group jump reporting
    if nargin > 4 && ~isempty(Y_test_baseline)
        fprintf('\nTest set group-jump metrics (using last year MMSE as baseline):\n');
        group_jump_report(Y_test, Y_test_pred, Y_test_baseline);
    else
        fprintf('\n[Note: Group-jump metrics require baseline MMSE for the test set.]\n');
    end
end

function [mean_rmse, mean_r2] = svm_inner_cv_r2(X, Y, cv, kernel, C, sigma, polyorder, eps)
    k = cv.NumTestSets;
    rmse_inner = zeros(k, 1);
    r2_inner = zeros(k, 1);
    for j = 1:k
        trainIdx = training(cv, j);
        valIdx = test(cv, j);
        X_in = X(trainIdx, :);
        Y_in = Y(trainIdx);
        X_val = X(valIdx, :);
        Y_val = Y(valIdx);

        switch kernel
            case 'rbf'
                model = fitrsvm(X_in, Y_in, 'KernelFunction', 'rbf', ...
                    'BoxConstraint', C, 'KernelScale', sigma, 'Epsilon', eps, 'Standardize', false);
            case 'linear'
                model = fitrsvm(X_in, Y_in, 'KernelFunction', 'linear', ...
                    'BoxConstraint', C, 'Epsilon', eps, 'Standardize', false);
            case 'polynomial'
                model = fitrsvm(X_in, Y_in, 'KernelFunction', 'polynomial', ...
                    'PolynomialOrder', polyorder, 'BoxConstraint', C, 'Epsilon', eps, 'Standardize', false);
        end
        Y_pred = predict(model, X_val);
        rmse_inner(j) = sqrt(mean((Y_val - Y_pred).^2));
        r2_inner(j) = 1 - sum((Y_val - Y_pred).^2) / sum((Y_val - mean(Y_val)).^2);
    end
    mean_rmse = mean(rmse_inner);
    mean_r2 = mean(r2_inner);
end


% Helper for inner CV
function mean_rmse = svm_inner_cv(X, Y, cv, kernel, C, sigma, polyorder, eps)
    k = cv.NumTestSets;
    rmse_inner = zeros(k, 1);
    for j = 1:k
        trainIdx = training(cv, j);
        valIdx = test(cv, j);
        X_in = X(trainIdx, :);
        Y_in = Y(trainIdx);
        X_val = X(valIdx, :);
        Y_val = Y(valIdx);

        switch kernel
            case 'rbf'
                model = fitrsvm(X_in, Y_in, 'KernelFunction', 'rbf', ...
                    'BoxConstraint', C, 'KernelScale', sigma, 'Epsilon', eps, 'Standardize', false);
            case 'linear'
                model = fitrsvm(X_in, Y_in, 'KernelFunction', 'linear', ...
                    'BoxConstraint', C, 'Epsilon', eps, 'Standardize', false);
            case 'polynomial'
                model = fitrsvm(X_in, Y_in, 'KernelFunction', 'polynomial', ...
                    'PolynomialOrder', polyorder, 'BoxConstraint', C, 'Epsilon', eps, 'Standardize', false);
        end
        Y_pred = predict(model, X_val);
        rmse_inner(j) = sqrt(mean((Y_val - Y_pred).^2));
    end
    mean_rmse = mean(rmse_inner);
end






function group_jump_report(Y_true, Y_pred, Y_baseline, cutpoints)
    if nargin < 4
        cutpoints = [0 24 27 30.1];
    end
    grouplabels = {'Significant','Mild','Normal'};
    true_group = discretize(Y_true, cutpoints, 'categorical', grouplabels);
    pred_group = discretize(Y_pred, cutpoints, 'categorical', grouplabels);
    base_group = discretize(Y_baseline, cutpoints, 'categorical', grouplabels);

    group_acc = mean(true_group == pred_group);
    fprintf('Group accuracy: %.2f%%\n', 100*group_acc);

    fprintf('Confusion matrix (true vs predicted group):\n');
    disp(crosstab(true_group, pred_group));

    actual_jump = double(true_group) < double(base_group);
    pred_jump  = double(pred_group) < double(base_group);

    sensitivity = sum(actual_jump & pred_jump) / max(sum(actual_jump),1);
    precision   = sum(actual_jump & pred_jump) / max(sum(pred_jump),1);

    fprintf('Sensitivity (recall) for group decline: %.2f%%\n', 100*sensitivity);
    fprintf('Precision (PPV) for group decline: %.2f%%\n', 100*precision);
end





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



