%--------------------------------------------------------------------------
% Script: linear_regression_features.m
%
% Purpose:
%   - Predict MMSE using feature dataset and ordinary least squares (OLS) linear regression.
%   - Reports test set performance and displays top coefficients.
%
% Inputs:
%   - train_features_Q2_imputed.csv (with missing data imputed)
%   - test_features_Q2_imputed.csv  (test set)
%
% Output:
%   - Console: test set R^2, RMSE, MAE; top coefficients.
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
id_col = 1;
target_col = find(strcmp(train_tbl.Properties.VariableNames, 'MMSCORE_followUp'));
target_cols = find(ismember(train_tbl.Properties.VariableNames, target_names));

feature_cols = setdiff(1:width(train_tbl), [id_col, target_cols]);

X_train = train_tbl{:, feature_cols};
Y_train = train_tbl{:, target_col};
X_test  = test_tbl{:, feature_cols};
Y_test  = test_tbl{:, target_col};
feature_names = train_tbl.Properties.VariableNames(feature_cols);

% ---- 3. Standardize features (not necessary, but good for coefficient comparison) ----

[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

% ---- 4. Fit OLS Linear Regression ----

tbl_train_lm = array2table(X_train_norm, 'VariableNames', feature_names);
tbl_train_lm.MMSE = Y_train;
model = fitlm(tbl_train_lm, 'MMSE');
% ---- 5. Predict and Evaluate ----

tbl_test_lm = array2table(X_test_norm, 'VariableNames', feature_names);
Y_pred_test = predict(model, tbl_test_lm);

rmse_test = sqrt(mean((Y_test - Y_pred_test).^2));
mae_test = mean(abs(Y_test - Y_pred_test));
sse_test = sum((Y_test - Y_pred_test).^2);
sst_test = sum((Y_test - mean(Y_test)).^2);
r2_test = 1 - sse_test / sst_test;

fprintf('\nLinear Regression Test set: RMSE = %.3f, MAE = %.3f, R^2 = %.3f\n', rmse_test, mae_test, r2_test);

% Baseline (predict mean)
baseline_pred = mean(Y_train) * ones(size(Y_test));
sse_base = sum((Y_test - baseline_pred).^2);
sst = sum((Y_test - mean(Y_test)).^2);
r2_base = 1 - sse_base / sst;
fprintf('Null-model R^2 (test): %.3f\n', r2_base);

% ---- 6. Show top coefficients ----

coefs = model.Coefficients.Estimate(2:end); % skip intercept
[~, idx] = sort(abs(coefs), 'descend');
disp('Top 10 features by absolute coefficient value:');
for k = 1:min(10, numel(coefs))
    fprintf('%2d. %-30s  Coefficient: %.4f\n', k, feature_names{idx(k)}, coefs(idx(k)));
end

% Optional: scatter plot
figure;
scatter(Y_test, Y_pred_test, 'filled');
xlabel('Actual MMSE (FollowUp)');
ylabel('Predicted MMSE');
title(sprintf('Linear Regression (R^2 = %.3f)', r2_test));
grid on; refline(1,0);
