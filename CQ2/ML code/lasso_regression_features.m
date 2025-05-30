% LASSO regression with cross-validation for Lambda (Alpha = 1 = LASSO)

clc; clear; close all;

% Load data as before
baseDir = fullfile('..', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q3_imputed.csv');
testFile  = fullfile(baseDir, 'test_features_Q3_imputed.csv');

train_tbl = readtable(trainFile);
test_tbl  = readtable(testFile);

% Define features/targets as before (excluding all target vars and ID)
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

% Standardize features (lasso expects this!)
[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

% LASSO regression (Alpha = 1 = pure LASSO)
[B, FitInfo] = lasso(X_train_norm, Y_train, ...
    'Alpha', 1, ...                 % LASSO
    'CV', 5, ...                    % 5-fold CV for Lambda
    'Standardize', false);          % Already standardized

% Find the best lambda (lowest MSE in CV)
idxLambdaMinMSE = FitInfo.IndexMinMSE;
bestB = B(:, idxLambdaMinMSE);
intercept = FitInfo.Intercept(idxLambdaMinMSE);

% Predict on test set
Y_pred_test = X_test_norm * bestB + intercept;

rmse_test = sqrt(mean((Y_test - Y_pred_test).^2));
mae_test = mean(abs(Y_test - Y_pred_test));
sse_test = sum((Y_test - Y_pred_test).^2);
sst_test = sum((Y_test - mean(Y_test)).^2);
r2_test = 1 - sse_test / sst_test;

fprintf('\nLASSO Regression Test set: RMSE = %.3f, MAE = %.3f, R^2 = %.3f\n', rmse_test, mae_test, r2_test);

% Show top 10 absolute coefficients
[~, idx] = sort(abs(bestB), 'descend');
disp('Top 10 features by absolute LASSO coefficient:');
for k = 1:min(10, numel(idx))
    fprintf('%2d. %-30s  Coefficient: %.4f\n', k, feature_names{idx(k)}, bestB(idx(k)));
end

% Optional: plot predicted vs actual
figure;
scatter(Y_test, Y_pred_test, 'filled');
xlabel('Actual MMSE (FollowUp)');
ylabel('Predicted MMSE');
title(sprintf('LASSO Regression (R^2 = %.3f)', r2_test));
grid on; refline(1,0);
