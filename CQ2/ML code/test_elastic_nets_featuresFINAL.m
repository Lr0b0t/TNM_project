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

%% ---- 3. Define hyperparameter grid ----

alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1];
lambdas = logspace(-4, 1, 8);

%% ---- 4. Nested Cross-validation setup ----

outerK = 5;
innerK = 3;
outerCV = cvpartition(size(X_train,1), 'KFold', outerK);

all_outer_r2 = zeros(outerK,1);
all_outer_rmse = zeros(outerK,1);
all_outer_mae = zeros(outerK,1);
bestParamsList = cell(outerK,1);

fprintf('\n===== NESTED CROSS-VALIDATION =====\n');

for i = 1:outerK
    fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
    trainIdx = training(outerCV, i);
    valIdx   = test(outerCV, i);

    Xtr_outer = X_train(trainIdx, :);
    Ytr_outer = Y_train(trainIdx);
    Xval_outer = X_train(valIdx, :);
    Yval_outer = Y_train(valIdx);

    % --- 4.1 Standardize: Fit scaler on outer-train, apply to val
    [Xtr_outer_norm, mu_outer, sigma_outer] = zscore(Xtr_outer);
    Xval_outer_norm = (Xval_outer - mu_outer) ./ sigma_outer;
    
    % --- 4.2 Inner CV for hyperparameter tuning ---
    innerCV = cvpartition(size(Xtr_outer_norm,1), 'KFold', innerK);

    best_inner_r2 = -Inf;
    best_inner_alpha = NaN;
    best_inner_lambda = NaN;

    % Grid search for both alpha and lambda
    for a = alphas
        for l = lambdas
            inner_r2s = zeros(innerK,1);
            for j = 1:innerK
                inner_trainIdx = training(innerCV, j);
                inner_valIdx = test(innerCV, j);

                Xtr_inner = Xtr_outer_norm(inner_trainIdx,:);
                Ytr_inner = Ytr_outer(inner_trainIdx);
                Xval_inner = Xtr_outer_norm(inner_valIdx,:);
                Yval_inner = Ytr_outer(inner_valIdx);

                % No standardize, already done
                [B, FitInfo] = lassoglm(Xtr_inner, Ytr_inner, 'normal', ...
                    'Alpha', a, 'Lambda', l, ...
                    'Standardize', false);

                coef = [FitInfo.Intercept; B];
                Xval_aug = [ones(size(Xval_inner,1),1), Xval_inner];
                Ypred_val = Xval_aug * coef;

                r2 = 1 - sum((Yval_inner - Ypred_val).^2) / sum((Yval_inner - mean(Yval_inner)).^2);
                inner_r2s(j) = r2;
            end
            mean_r2 = mean(inner_r2s);
            if mean_r2 > best_inner_r2
                best_inner_r2 = mean_r2;
                best_inner_alpha = a;
                best_inner_lambda = l;
            end
        end
    end

    fprintf('  Best inner params: Alpha=%.2f, Lambda=%.5f (Inner CV R^2=%.4f)\n', ...
        best_inner_alpha, best_inner_lambda, best_inner_r2);

    % --- 4.3 Fit on all outer-train with best inner params, predict on outer-val
    [B, FitInfo] = lassoglm(Xtr_outer_norm, Ytr_outer, 'normal', ...
        'Alpha', best_inner_alpha, 'Lambda', best_inner_lambda, ...
        'Standardize', false);
    coef = [FitInfo.Intercept; B];

    Xval_aug = [ones(size(Xval_outer_norm,1),1), Xval_outer_norm];
    Ypred_val = Xval_aug * coef;

    r2 = 1 - sum((Yval_outer - Ypred_val).^2) / sum((Yval_outer - mean(Yval_outer)).^2);
    rmse = sqrt(mean((Yval_outer - Ypred_val).^2));
    mae = mean(abs(Yval_outer - Ypred_val));

    all_outer_r2(i) = r2;
    all_outer_rmse(i) = rmse;
    all_outer_mae(i) = mae;
    bestParamsList{i} = struct('Alpha', best_inner_alpha, 'Lambda', best_inner_lambda);

    fprintf('>> Fold %d: Outer R^2=%.4f, RMSE=%.4f, MAE=%.4f\n', i, r2, rmse, mae);
end

fprintf('\n===== Nested CV Performance =====\n');
fprintf('Mean Outer R^2: %.4f (std %.4f)\n', mean(all_outer_r2), std(all_outer_r2));
fprintf('Mean Outer RMSE: %.4f (std %.4f)\n', mean(all_outer_rmse), std(all_outer_rmse));
fprintf('Mean Outer MAE: %.4f (std %.4f)\n', mean(all_outer_mae), std(all_outer_mae));

%% ---- 5. Final model: Retrain on full train set with best params and test on test set ----

% Choose most frequent best alpha and lambda, or use mean if tie
alphas_cv = cellfun(@(s) s.Alpha, bestParamsList);
lambdas_cv = cellfun(@(s) s.Lambda, bestParamsList);

% If ties, mode returns the smallest value
bestAlpha = mode(alphas_cv);
bestLambda = mode(lambdas_cv);




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

