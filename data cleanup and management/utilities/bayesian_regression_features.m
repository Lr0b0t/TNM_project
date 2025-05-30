%--------------------------------------------------------------------------
% Script: bayesian_regression_features.m
%
% Purpose:
%   - Predict MMSE using Bayesian linear regression.
%   - Reports test set performance and shows posterior intervals for coefficients.
%
% Inputs:
%   - train_features_Q2_imputed.csv (features, imputed)
%   - test_features_Q2_imputed.csv  (test set, imputed)
%
% Output:
%   - Console: test set R^2, RMSE, MAE, and some info on uncertainty.
%   - (Optional) Plots posterior mean and intervals for top coefficients.
%--------------------------------------------------------------------------

clc; clear; close all;

% ---- 1. Load Data ----

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

% Standardize features (recommended for Bayesian regression)
[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

% ---- 3. Fit Bayesian Linear Regression ----

% Prior: Automatic (default, "conjugate normal-inverse-gamma" prior)
Mdl = bayeslm(size(X_train_norm,2), 'ModelType', 'conjugate');

% Estimate posterior
PosteriorMdl = estimate(Mdl, X_train_norm, Y_train);

% Posterior mean coefficients
betaMean = PosteriorMdl.MuBeta;
intercept = PosteriorMdl.MuIntercept;

% Predict on test set (posterior mean prediction)
Y_pred_test = X_test_norm * betaMean + intercept;

rmse_test = sqrt(mean((Y_test - Y_pred_test).^2));
mae_test = mean(abs(Y_test - Y_pred_test));
sse_test = sum((Y_test - Y_pred_test).^2);
sst_test = sum((Y_test - mean(Y_test)).^2);
r2_test = 1 - sse_test / sst_test;

fprintf('\nBayesian Regression Test set: RMSE = %.3f, MAE = %.3f, R^2 = %.3f\n', rmse_test, mae_test, r2_test);

% Show top 10 features by absolute value of mean coefficient
[~, idx] = sort(abs(betaMean), 'descend');
disp('Top 10 features by absolute Bayesian mean coefficient:');
for k = 1:min(10, numel(idx))
    fprintf('%2d. %-30s  Mean Coef: %.4f\n', k, feature_names{idx(k)}, betaMean(idx(k)));
end

% Optional: plot predicted vs actual
figure;
scatter(Y_test, Y_pred_test, 'filled');
xlabel('Actual MMSE (FollowUp)');
ylabel('Predicted MMSE');
title(sprintf('Bayesian Regression (R^2 = %.3f)', r2_test));
grid on; refline(1,0);

% Optional: plot posterior intervals for top features
try
    CI = coefCI(PosteriorMdl, 0.05); % 95% intervals
    figure;
    errorbar(1:10, betaMean(idx(1:10)), ...
        betaMean(idx(1:10)) - CI(idx(1:10), 1), ...
        CI(idx(1:10), 2) - betaMean(idx(1:10)), 'o');
    set(gca, 'XTick', 1:10, 'XTickLabel', feature_names(idx(1:10)), 'XTickLabelRotation', 45);
    ylabel('Posterior Mean Coefficient');
    title('Top 10 Bayesian Regression Coefficients with 95% CI');
    grid on;
catch
    disp('Coefficient intervals not available in this MATLAB version.');
end
