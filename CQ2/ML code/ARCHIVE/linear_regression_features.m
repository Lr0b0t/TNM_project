%--------------------------------------------------------------------------
% Script: linear_regression_nestedcv.m
%
% Purpose:
%   - Predict MMSE using feature dataset and OLS linear regression.
%   - Uses nested cross-validation for unbiased performance estimate.
%   - Reports test set performance and top coefficients.
%--------------------------------------------------------------------------

clc; clear; close all;
%rng(24, 'twister')

% ---- 1. Load Data ----

baseDir = fullfile('..','..', 'data cleanup and management', 'final files');
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

X_train = train_tbl{:, feature_cols};
Y_train = train_tbl{:, target_col};
X_test  = test_tbl{:, feature_cols};
Y_test  = test_tbl{:, target_col};
feature_names = train_tbl.Properties.VariableNames(feature_cols);

%% ---- 3. Nested Cross-Validation ----

outerK = 5;
innerK = 3; % For example, 3-fold inner CV
outerCV = cvpartition(length(Y_train), 'KFold', outerK);

all_outer_r2 = zeros(outerK,1);
all_outer_rmse = zeros(outerK,1);
all_outer_mae = zeros(outerK,1);
all_outer_nfeat = zeros(outerK,1);

fprintf('\n===== NESTED CROSS-VALIDATION =====\n');

% Number of top features to try (grid search)
feature_grid = [5, 10, 20, 30, 50, 100, size(X_train,2)];

for i = 1:outerK
    fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
    trainIdx = training(outerCV, i);
    valIdx   = test(outerCV, i);

    Xtr_outer = X_train(trainIdx, :);
    Ytr_outer = Y_train(trainIdx);
    Xval_outer = X_train(valIdx, :);
    Yval_outer = Y_train(valIdx);

    % ---- Standardize using only outer-train statistics
    [Xtr_outer_norm, mu_outer, sigma_outer] = zscore(Xtr_outer);
    Xval_outer_norm = (Xval_outer - mu_outer) ./ sigma_outer;

    % ---- INNER CV: Select best #features by univariate correlation ----
    innerCV = cvpartition(length(Ytr_outer), 'KFold', innerK);
    mean_inner_r2 = zeros(length(feature_grid),1);

    for j = 1:length(feature_grid)
        nFeat = feature_grid(j);
        inner_r2 = zeros(innerK,1);

        for k = 1:innerK
            inner_train_idx = training(innerCV, k);
            inner_val_idx   = test(innerCV, k);

            Xtr_inner = Xtr_outer_norm(inner_train_idx,:);
            Ytr_inner = Ytr_outer(inner_train_idx);
            Xval_inner = Xtr_outer_norm(inner_val_idx,:);
            Yval_inner = Ytr_outer(inner_val_idx);

            % Select features by univariate correlation
            corrvals = abs(corr(Xtr_inner, Ytr_inner));
            [~, sort_idx] = sort(corrvals, 'descend');
            selected = sort_idx(1:min(nFeat, size(Xtr_inner,2)));

            % Train model
            tbl_inner = array2table(Xtr_inner(:,selected), 'VariableNames', feature_names(selected));
            tbl_inner.MMSE = Ytr_inner;
            mdl_inner = fitlm(tbl_inner, 'MMSE');
            tbl_inner_val = array2table(Xval_inner(:,selected), 'VariableNames', feature_names(selected));
            Ypred_inner = predict(mdl_inner, tbl_inner_val);

            % R^2 on validation fold
            r2_inner = 1 - sum((Yval_inner - Ypred_inner).^2) / sum((Yval_inner - mean(Yval_inner)).^2);
            inner_r2(k) = r2_inner;
        end

        mean_inner_r2(j) = mean(inner_r2);
    end

    % Choose nFeat (feature count) that gave best inner CV R^2
    [~, best_idx] = max(mean_inner_r2);
    best_nfeat = feature_grid(best_idx);

    % ---- Retrain on full outer-train with best n features ----
    corrvals = abs(corr(Xtr_outer_norm, Ytr_outer));
    [~, sort_idx] = sort(corrvals, 'descend');
    selected = sort_idx(1:min(best_nfeat, size(Xtr_outer_norm,2)));

    tbl_train_lm = array2table(Xtr_outer_norm(:,selected), 'VariableNames', feature_names(selected));
    tbl_train_lm.MMSE = Ytr_outer;
    model = fitlm(tbl_train_lm, 'MMSE');

    tbl_val_lm = array2table(Xval_outer_norm(:,selected), 'VariableNames', feature_names(selected));
    Y_pred_val = predict(model, tbl_val_lm);

    rmse_val = sqrt(mean((Yval_outer - Y_pred_val).^2));
    mae_val = mean(abs(Yval_outer - Y_pred_val));
    sse_val = sum((Yval_outer - Y_pred_val).^2);
    sst_val = sum((Yval_outer - mean(Yval_outer)).^2);
    r2_val = 1 - sse_val / sst_val;

    all_outer_r2(i) = r2_val;
    all_outer_rmse(i) = rmse_val;
    all_outer_mae(i) = mae_val;
    all_outer_nfeat(i) = best_nfeat;

    fprintf('Outer fold R^2 = %.3f, RMSE = %.3f, MAE = %.3f | Best nFeat = %d\n', ...
        r2_val, rmse_val, mae_val, best_nfeat);
end

fprintf('\n===== Nested CV Performance =====\n');
fprintf('Mean Outer R^2: %.3f (std %.3f)\n', mean(all_outer_r2), std(all_outer_r2));
fprintf('Mean Outer RMSE: %.3f (std %.3f)\n', mean(all_outer_rmse), std(all_outer_rmse));
fprintf('Mean Outer MAE: %.3f (std %.3f)\n', mean(all_outer_mae), std(all_outer_mae));
fprintf('Mean selected nFeat: %.1f\n', mean(all_outer_nfeat));


%% ---- 4. Final model: Retrain on full train set and evaluate on test set ----

[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

tbl_train_lm = array2table(X_train_norm, 'VariableNames', feature_names);
tbl_train_lm.MMSE = Y_train;
model = fitlm(tbl_train_lm, 'MMSE');

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

% ---- 5. Show top coefficients ----

coefs = model.Coefficients.Estimate(2:end); % skip intercept
[~, idx] = sort(abs(coefs), 'descend');
disp('Top 10 features by absolute coefficient value:');
for k = 1:min(10, numel(coefs))
    fprintf('%2d. %-30s  Coefficient: %.4f\n', k, feature_names{idx(k)}, coefs(idx(k)));
end

figure;
scatter(Y_test, Y_pred_test, 'filled');
xlabel('Actual MMSE (FollowUp)');
ylabel('Predicted MMSE');
title(sprintf('Linear Regression (R^2 = %.3f)', r2_test));
grid on; refline(1,0);
