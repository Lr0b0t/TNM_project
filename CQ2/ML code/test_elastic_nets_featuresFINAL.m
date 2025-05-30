%%--------------------------------------------------------------------------
% Script: elastic_net_nested_cv.m
%
% Purpose:
%   - Predict MMSE (or other target) using feature dataset and Elastic Net.
%   - Performs **nested cross-validation** to select optimal Alpha and Lambda.
%   - Reports unbiased nested-CV performance and test set performance.
%
% Inputs:
%   - train_features_Q2_imputed.csv (features, with missing data imputed)
%   - test_features_Q2_imputed.csv  (test set, with missing data imputed)
%
% Output:
%   - Console output: best parameters, nested CV R^2/RMSE/MAE, test set metrics.
%   - Optional: plot true vs predicted scores.
%--------------------------------------------------------------------------

clc; clear; close all;
rng(42, 'twister');


%% ---- 1. Load Data ----

baseDir = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(baseDir, 'test_features_Q2_imputed.csv');

train_tbl = readtable(trainFile);
test_tbl  = readtable(testFile);

%% ---- 2. Identify feature and target columns ----

target_names = {'MMSCORE_followUp', 'CDSOB_followUp', 'GDTOTAL_followUp'};

% Find indices
target_cols = find(ismember(train_tbl.Properties.VariableNames, target_names));
id_col = 1; % First column is assumed to be ID

feature_cols = setdiff(1:width(train_tbl), [id_col, target_cols]);

X_train = train_tbl{:, feature_cols};
Y_train = train_tbl{:, strcmp(train_tbl.Properties.VariableNames, 'MMSCORE_followUp')};
X_test  = test_tbl{:, feature_cols};
Y_test  = test_tbl{:, strcmp(test_tbl.Properties.VariableNames, 'MMSCORE_followUp')};


[all_outer_r2, all_outer_rmse, all_outer_mae, bestParamsList, bestAlpha, bestLambda] = run_Elastic_Net_Regression(X_train, Y_train);


fprintf('\n===== FINAL MODEL EVALUATION =====\n');
fprintf('Retraining with Alpha=%.2f, Lambda=%.5f\n', bestAlpha, bestLambda);

% Standardize training set
[X_train_norm, mu_final, sigma_final] = zscore(X_train);
X_test_norm = (X_test - mu_final) ./ sigma_final;

[B, FitInfo] = lassoglm(X_train_norm, Y_train, 'normal', ...
                        'Alpha', bestAlpha, 'Lambda', bestLambda, ...
                        'Standardize', false);
coef = [FitInfo.Intercept; B];

X_test_aug = [ones(size(X_test_norm,1),1), X_test_norm];
Y_pred_test = X_test_aug * coef;

rmse_test = sqrt(mean((Y_test - Y_pred_test).^2));
mae_test = mean(abs(Y_test - Y_pred_test));
r2_test = 1 - sum((Y_test - Y_pred_test).^2) / sum((Y_test - mean(Y_test)).^2);

fprintf('Test set: RMSE = %.3f, MAE = %.3f, R^2 = %.3f\n', rmse_test, mae_test, r2_test);

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
title(sprintf('Elastic Net Regression (Test R^2 = %.3f)', r2_test));
grid on; refline(1,0);

