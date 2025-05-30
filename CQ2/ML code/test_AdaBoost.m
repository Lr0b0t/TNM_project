close all; clear;


rng(42, 'twister');
% TODO: write about ε, test teh jumping groups function

%% Step 0: Load Training and Test Data

% % Define base path to data files
% baseDir = fileparts(pwd);                     % e.g., if in /utilities/
% dataDir = fullfile(baseDir, 'final files');
% 
% % Define file paths
% trainFile = fullfile(dataDir, 'train_features_Q2_imputed.csv');
% testFile  = fullfile(dataDir, 'test_features_Q2_imputed.csv');
% 
% % Load training and test data tables
% trainTable = readtable(trainFile);
% testTable  = readtable(testFile);
% 
% % Remove ID column (assumed to be the first column)
% trainTable(:,1) = [];
% testTable(:,1) = [];
% 
% % Identify numeric columns (e.g., skip categorical if present)
% numericVarsTrain = varfun(@isnumeric, trainTable, 'OutputFormat', 'uniform');
% numericVarsTest  = varfun(@isnumeric, testTable,  'OutputFormat', 'uniform');
% 
% % Extract numeric matrices for modeling
% trainFilled = trainTable{:, numericVarsTrain};
% testFilled  = testTable{:, numericVarsTest};  % No missing values assumed
% 
% 
% %% Step 2: Define Target Columns and Extract MMSE
% numTargets = 3;
% 
% % Training set
% X_train = trainFilled(:, 1:end - numTargets);        % Features
% Y_all_train = trainFilled(:, end - numTargets + 1:end);
% Y_train = Y_all_train(:, 2);                         % MMSE (second-to-last)
% 
% % Test set
% X_test = testFilled(:, 1:end - numTargets);          % Features only
% Y_all_test = testFilled(:, end - numTargets + 1:end);
% Y_test = Y_all_test(:, 2);                           % MMSE



baseDir = fullfile('..', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(baseDir, 'test_features_Q2_imputed.csv');

train_tbl = readtable(trainFile);
test_tbl  = readtable(testFile);

% ---- 2. Identify feature and target columns ----

target_names = {'MMSCORE_followUp', 'CDSOB_followUp', 'GDTOTAL_followUp'};
id_col = 1;
target_col = find(strcmp(train_tbl.Properties.VariableNames, 'MMSCORE_followUp'));
target_cols = find(ismember(train_tbl.Properties.VariableNames, target_names));

feature_cols = setdiff(1:width(train_tbl), [id_col, target_cols]);
feature_names = train_tbl.Properties.VariableNames(feature_cols);

X_train = train_tbl{:, feature_cols};
Y_train = train_tbl{:, target_col};
X_test  = test_tbl{:, feature_cols};
Y_test  = test_tbl{:, target_col};


Y_test_baseline = X_test(:, 3);
Y_train_baseline = X_train(:, 3);

%% Step 3: Normalize Features (Z-score using training stats only)
[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

%% Step 4: Run Nested Cross-Validation and Final Test

results_ada = run_nested_cv_adaboost(X_train_norm, Y_train, X_test_norm, Y_test, Y_test_baseline);

%%

function results = run_nested_cv_adaboost(X_train, Y_train, X_test, Y_test, Y_test_baseline)
% RUN_NESTED_CV_ADABOOST - AdaBoost (LSBoost) regression with nested CV and group-jump reporting
%
%   X_train, Y_train: training features and targets (numeric)
%   X_test, Y_test: test features and targets (numeric)
%   Y_test_baseline: baseline MMSE for test set (for group jumps)
%
%   Returns: results struct

    outerK = 5;
    innerK = 3;
    outerCV = cvpartition(size(X_train, 1), 'KFold', outerK);

    % Parameter grids
    numCycles_values = [30, 50, 100, 200];
    learnRate_values = [0.01, 0.05, 0.1, 0.2];
    maxSplits_values = [2, 4, 8];

    outerRMSE = zeros(outerK, 1);
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
        bestRMSE = Inf;
        hyperparamTable = [];

        for numCycles = numCycles_values
            for learnRate = learnRate_values
                for maxSplits = maxSplits_values
                    rmse_inner = zeros(innerK, 1);
                    for j = 1:innerK
                        trIdx = training(innerCV, j);
                        valIdx = test(innerCV, j);

                        X_in = X_outerTrain(trIdx, :);
                        Y_in = Y_outerTrain(trIdx);
                        X_val = X_outerTrain(valIdx, :);
                        Y_val = Y_outerTrain(valIdx);

                        tTree = templateTree('MaxNumSplits', maxSplits);
                        mdl = fitrensemble(X_in, Y_in, ...
                            'Method', 'LSBoost', ...
                            'Learners', tTree, ...
                            'NumLearningCycles', numCycles, ...
                            'LearnRate', learnRate);

                        Y_pred = predict(mdl, X_val);
                        rmse_inner(j) = sqrt(mean((Y_val - Y_pred).^2));
                    end
                    mean_rmse = mean(rmse_inner);

                    newrow = table(numCycles, learnRate, maxSplits, mean_rmse, ...
                        'VariableNames', {'NumCycles', 'LearnRate', 'MaxSplits', 'RMSE'});
                    hyperparamTable = [hyperparamTable; newrow];

                    if mean_rmse < bestRMSE
                        bestRMSE = mean_rmse;
                        bestParams = struct('NumCycles', numCycles, ...
                                            'LearnRate', learnRate, ...
                                            'MaxSplits', maxSplits);
                    end
                end
            end
        end

        allResults{i} = hyperparamTable;

        % Train best model on all outer train
        tTree = templateTree('MaxNumSplits', bestParams.MaxSplits);
        finalModel = fitrensemble(X_outerTrain, Y_outerTrain, ...
            'Method', 'LSBoost', ...
            'Learners', tTree, ...
            'NumLearningCycles', bestParams.NumCycles, ...
            'LearnRate', bestParams.LearnRate);

        Y_outerPred = predict(finalModel, X_outerTest);
        outerRMSE(i) = sqrt(mean((Y_outerTest - Y_outerPred).^2));
        bestParamsList{i} = bestParams;

        fprintf('Best params for fold %d: Cycles=%d, LearnRate=%.3f, MaxSplits=%d, RMSE=%.4f\n', ...
            bestParams.NumCycles, bestParams.LearnRate, bestParams.MaxSplits, bestRMSE);
        disp(sortrows(hyperparamTable, 'RMSE'));
    end

    % Determine best/frequent hyperparameters
    cycles = cellfun(@(s) s.NumCycles, bestParamsList);
    rates = cellfun(@(s) s.LearnRate, bestParamsList);
    splits = cellfun(@(s) s.MaxSplits, bestParamsList);

    bestCycles = mode(cycles);
    bestRate = mode(rates);
    bestSplits = mode(splits);

    % Retrain final model on full training data
    tTree = templateTree('MaxNumSplits', bestSplits);
    finalModel = fitrensemble(X_train, Y_train, ...
        'Method', 'LSBoost', ...
        'Learners', tTree, ...
        'NumLearningCycles', bestCycles, ...
        'LearnRate', bestRate);

    Y_test_pred = predict(finalModel, X_test);

    % Regression metrics
    errors = Y_test - Y_test_pred;
    mse  = mean(errors.^2);
    rmse = sqrt(mse);
    mae  = mean(abs(errors));
    r2   = 1 - sum(errors.^2) / sum((Y_test - mean(Y_test)).^2);

    % Print regression metrics
    fprintf('\nAdaBoost Test set regression metrics:\n');
    fprintf('  RMSE: %.4f\n', rmse);
    fprintf('  MAE : %.4f\n', mae);
    fprintf('  R^2 : %.4f\n', r2);

    % Save results
    results = struct();
    results.outerRMSE = outerRMSE;
    results.bestParamsList = bestParamsList;
    results.allResults = allResults;
    results.finalParams = struct('NumCycles', bestCycles, 'LearnRate', bestRate, 'MaxSplits', bestSplits);
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




