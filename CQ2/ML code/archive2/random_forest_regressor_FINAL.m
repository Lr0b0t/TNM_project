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
rng(6, 'twister');
% ---- 1. Load Data ----

baseDir = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(baseDir, 'test_features_Q2_imputed.csv');

train_tbl = readtable(trainFile);
test_tbl  = readtable(testFile);

% ---- 2. Identify feature and target columns ----

target_names = {'MMSCORE_followUp', 'CDSOB_followUp', 'GDTOTAL_followUp'};

% Find indices
target_cols = find(ismember(train_tbl.Properties.VariableNames, target_names));
id_col = 1; % first column is ID

% Only use as features those columns not in targets or ID
feature_cols = setdiff(1:width(train_tbl), [id_col, target_cols]);

X_train = train_tbl{:, feature_cols};
Y_train = train_tbl{:, strcmp(train_tbl.Properties.VariableNames, 'MMSCORE_followUp')};
X_test  = test_tbl{:, feature_cols};
Y_test  = test_tbl{:, strcmp(test_tbl.Properties.VariableNames, 'MMSCORE_followUp')};

% ---- 3. Standardize features (optional for trees, but keeps comparability) ----

[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

%%
outerK = 5; innerK = 3;
 [all_outer_r2_rf, mean_outer_r2, std_outer_r2, bestParamsList, ...
 bestParamsMode] = run_Random_Forest_Regression(X_train, Y_train, outerK, innerK)

%% Retrain on all training data with best params
finalModel = fitrensemble(X_train_norm, Y_train, ...
    'Method', 'Bag', ...
    'NumLearningCycles', bestParamsMode.NumTrees, ...
    'Learners', templateTree(...
        'MinLeafSize', bestParamsMode.MinLeaf, ...
        'MaxNumSplits', bestParamsMode.MaxNumSplits));



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



