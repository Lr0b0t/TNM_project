%--------------------------------------------------------------------------
% Script: random_forest_features.m
%
% Purpose:
%   - Predict MMSE using feature dataset and Random Forest regression.
%   - Optimizes key hyperparameters via cross-validation.
%   - Reports best parameters and test set performance.
%
% Inputs:
%   - train_features_Q2_imputed.csv (features, with missing data imputed)
%   - test_features_Q2_imputed.csv  (test set, with missing data imputed)
%
% Output:
%   - Console output: best parameters, test set R^2, RMSE, MAE.
%   - (Optional) Plots true vs predicted scores.
%--------------------------------------------------------------------------

clc; clear; close all;

% ---- 1. Load Data ----

baseDir = fullfile('..', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q3_imputed.csv');
testFile  = fullfile(baseDir, 'test_features_Q3_imputed.csv');

train_tbl = readtable(trainFile);
test_tbl  = readtable(testFile);

% ---- 2. Identify feature and target columns ----

target_names = {'MMSCORE_followUp', 'CDSOB_followUp', 'GDTOTAL_followUp'};

% Find indices
target_cols = find(ismember(train_tbl.Properties.VariableNames, target_names));
id_col = 1; % Still assuming first column is ID

% Only use as features those columns not in targets or ID
feature_cols = setdiff(1:width(train_tbl), [id_col, target_cols]);

X_train = train_tbl{:, feature_cols};
Y_train = train_tbl{:, strcmp(train_tbl.Properties.VariableNames, 'MMSCORE_followUp')};
X_test  = test_tbl{:, feature_cols};
Y_test  = test_tbl{:, strcmp(test_tbl.Properties.VariableNames, 'MMSCORE_followUp')};

% ---- 3. Standardize features (optional for trees, but keeps comparability) ----

[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

% ---- 4. Set up hyperparameter grids ----

numTrees_grid = [40, 80, 150, 200];      % Number of trees in the forest
minLeaf_grid = [1, 3, 5, 8, 12];         % Minimum leaf size
maxNumSplits_grid = [10, 50, 100, 200];  % Maximum number of splits

outerK = 5;
innerK = 3;
outerCV = cvpartition(size(X_train_norm,1), 'KFold', outerK);

all_outer_r2 = zeros(outerK,1);
bestParamsList = cell(outerK,1);

fprintf('\n==== Starting Random Forest Nested Cross-Validation ====\n');
for i = 1:outerK
    fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
    trainIdx = training(outerCV, i);
    valIdx   = test(outerCV, i);
    Xtr_outer = X_train_norm(trainIdx, :);
    Ytr_outer = Y_train(trainIdx);
    Xval_outer = X_train_norm(valIdx, :);
    Yval_outer = Y_train(valIdx);

    best_r2 = -Inf;

    for ntree = numTrees_grid
        for minLeaf = minLeaf_grid
            for maxSplit = maxNumSplits_grid
                % Train ensemble
                model = fitrensemble(Xtr_outer, Ytr_outer, ...
                    'Method', 'Bag', ...
                    'NumLearningCycles', ntree, ...
                    'Learners', templateTree(...
                        'MinLeafSize', minLeaf, ...
                        'MaxNumSplits', maxSplit));

                Ypred_val = predict(model, Xval_outer);
                r2 = 1 - sum((Yval_outer - Ypred_val).^2) / sum((Yval_outer - mean(Yval_outer)).^2);

                fprintf('  Trees=%d, MinLeaf=%d, MaxSplit=%d | R^2=%.4f\n', ntree, minLeaf, maxSplit, r2);

                if r2 > best_r2
                    best_r2 = r2;
                    best_param = struct('NumTrees', ntree, 'MinLeaf', minLeaf, 'MaxNumSplits', maxSplit);
                end
            end
        end
    end

    all_outer_r2(i) = best_r2;
    bestParamsList{i} = best_param;
    fprintf('>> Fold %d best: Trees=%d, MinLeaf=%d, MaxSplit=%d | Outer R^2=%.4f\n', ...
        i, best_param.NumTrees, best_param.MinLeaf, best_param.MaxNumSplits, best_r2);
end

fprintf('\n=== Nested CV complete! ===\n');
fprintf('Outer fold R^2s: '); disp(all_outer_r2');

%% Find the mode (most frequent) best parameters
numTrees_cv = cellfun(@(s) s.NumTrees, bestParamsList);
minLeaf_cv = cellfun(@(s) s.MinLeaf, bestParamsList);
maxSplit_cv = cellfun(@(s) s.MaxNumSplits, bestParamsList);

bestNumTrees = mode(numTrees_cv);
bestMinLeaf = mode(minLeaf_cv);
bestMaxSplit = mode(maxSplit_cv);

fprintf('\nBest parameters (mode): Trees=%d, MinLeaf=%d, MaxSplit=%d\n', bestNumTrees, bestMinLeaf, bestMaxSplit);

%% Retrain on all training data with best params
finalModel = fitrensemble(X_train_norm, Y_train, ...
    'Method', 'Bag', ...
    'NumLearningCycles', bestNumTrees, ...
    'Learners', templateTree(...
        'MinLeafSize', bestMinLeaf, ...
        'MaxNumSplits', bestMaxSplit));

Y_pred_test = predict(finalModel, X_test_norm);

rmse_test = sqrt(mean((Y_test - Y_pred_test).^2));
mae_test = mean(abs(Y_test - Y_pred_test));
sse_test = sum((Y_test - Y_pred_test).^2);
sst_test = sum((Y_test - mean(Y_test)).^2);
r2_test = 1 - sse_test / sst_test;

fprintf('\nFinal test set: RMSE = %.3f, MAE = %.3f, R^2 = %.3f\n', rmse_test, mae_test, r2_test);

% Baseline (predict mean)
baseline_pred = mean(Y_train) * ones(size(Y_test));
sse_base = sum((Y_test - baseline_pred).^2);
sst = sum((Y_test - mean(Y_test)).^2);
r2_base = 1 - sse_base / sst;
fprintf('Null-model R^2 (test): %.3f\n', r2_base);

% Optional: scatter plot
figure;
scatter(Y_test, Y_pred_test, 'filled');
xlabel('Actual MMSE (FollowUp)');
ylabel('Predicted MMSE');
title(sprintf('Random Forest Regression (R^2 = %.3f)', r2_test));
grid on; refline(1,0);



%%


imp = predictorImportance(finalModel);

% The features used as input:
feature_names = train_tbl.Properties.VariableNames;
feature_names = feature_names(setdiff(1:width(train_tbl), [id_col, target_cols(2)]));

% Sort importance in descending order
[~, idx] = sort(imp, 'descend');

% Display the top 10 features and their importance
disp('Top 10 most important features in Random Forest:');
for k = 1:min(10, numel(idx))
    fprintf('%2d. %-30s  Importance: %.4f\n', k, feature_names{idx(k)}, imp(idx(k)));
end
