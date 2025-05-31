clc; clear; close all;
rng(42, 'twister');
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

%--- training‐set target vector
y = Y_train;  

%--- basic summaries
mean_y   = mean(y);
median_y = median(y);
std_y    = std(y);
min_y    = min(y);
max_y    = max(y);
q25      = prctile(y,25);
q75      = prctile(y,75);

fprintf('Training MMSCORE: mean=%.3f, median=%.3f, std=%.3f\n', mean_y, median_y, std_y);
fprintf('  range [%.3f .. %.3f], 25th=%.3f, 75th=%.3f\n', min_y, max_y, q25, q75);

%%

%--- null‐predictions: always the training mean
y_pred_null_train = mean_y * ones(size(Y_train));
y_pred_null_test  = mean_y * ones(size(Y_test));


%--- mean squared error
mse_train = mean((Y_train - y_pred_null_train).^2);
mse_test  = mean((Y_test  - y_pred_null_test ).^2);

%--- root mean squared error
rmse_train = sqrt(mse_train);
rmse_test  = sqrt(mse_test);

%--- mean absolute error
mae_train = mean(abs(Y_train - y_pred_null_train));
mae_test  = mean(abs(Y_test  - y_pred_null_test ));

%--- R^2 (coefficient of determination)
%    For train data this will be exactly 0 by definition.
sst_train = sum((Y_train - mean_y).^2);
ssr_train = sum((Y_train - y_pred_null_train).^2);
r2_train  = 1 - ssr_train/sst_train;  % → 0

sst_test = sum((Y_test -  mean(Y_test)).^2);
ssr_test = sum((Y_test - y_pred_null_test ).^2);
r2_test  = 1 - ssr_test/sst_test;  % may be ≤0


fprintf('NULL model performance:\n');
fprintf('  Train: MSE=%.3f, RMSE=%.3f, MAE=%.3f,  R^2=%.3f\n', ...
        mse_train, rmse_train, mae_train, r2_train);
fprintf('  Test : MSE=%.3f, RMSE=%.3f, MAE=%.3f,  R^2=%.3f\n', ...
        mse_test,  rmse_test,  mae_test,  r2_test);
