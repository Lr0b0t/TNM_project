%%--------------------------------------------------------------------------
% Script: elastic_net_features.m
%
% Purpose:
%   - Predict MMSE (or other target) using feature dataset and Elastic Net.
%   - Performs nested cross-validation to select optimal Alpha (mix) and Lambda (reg. strength).
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

% note: Elastic net and lasso/ridge regression are regularized versions of linear regression.
% 
% If you set Alpha = 0 and Lambda = 0 in lassoglm or fitrlinear, you get standard (ordinary) least squares linear regression—without regularization.
% 
% fitlm is MATLAB's classic function for OLS (ordinary least squares) linear regression and is a good choice for a pure, unregularized model.

clc; clear; close all;

% ---- 1. Load Data ----

baseDir = fullfile('..', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(baseDir, 'test_features_Q2_imputed.csv');

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


% ---- 3. Standardize features (z-score) ----

[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

% ---- 4. Set up hyperparameter grids ----

% alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1];
alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1];
lambdas = logspace(-4, 1, 8);

outerK = 5;
outerCV = cvpartition(size(X_train_norm,1), 'KFold', outerK);
all_outer_r2 = zeros(outerK,1);
bestParamsList = cell(outerK,1);

for i = 1:outerK
    fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
    trainIdx = training(outerCV, i);
    valIdx   = test(outerCV, i);
    Xtr_outer = X_train_norm(trainIdx, :);
    Ytr_outer = Y_train(trainIdx);
    Xval_outer = X_train_norm(valIdx, :);
    Yval_outer = Y_train(valIdx);

    best_r2 = -Inf;

    for a = alphas
        [B, FitInfo] = lassoglm(Xtr_outer, Ytr_outer, 'normal', ...
                                'Alpha', a, 'Lambda', lambdas, ...
                                'Standardize', false, 'CV', 3);

        % Pick the best lambda from lassoglm's inner CV
        [~, idxLambdaMinDeviance] = min(FitInfo.Deviance);
        bestLambda = FitInfo.Lambda(idxLambdaMinDeviance);
        coef = [FitInfo.Intercept(idxLambdaMinDeviance); B(:, idxLambdaMinDeviance)];
        
        % Predict on validation fold
        Xval_aug = [ones(size(Xval_outer,1),1), Xval_outer];
        Ypred_val = Xval_aug * coef;

        % Compute R^2
        r2 = 1 - sum((Yval_outer - Ypred_val).^2) / sum((Yval_outer - mean(Yval_outer)).^2);
        fprintf('  Alpha=%.2f, Lambda=%.5f | R^2=%.4f\n', a, bestLambda, r2);
        if r2 > best_r2
            best_r2 = r2;
            best_param = struct('Alpha', a, 'Lambda', bestLambda, 'Coef', coef);
        end
    end

    all_outer_r2(i) = best_r2;
    bestParamsList{i} = best_param;
    fprintf('>> Fold %d best: Alpha=%.2f, Lambda=%.5f | Outer R^2=%.4f\n', ...
        i, best_param.Alpha, best_param.Lambda, best_r2);
end

% Mode selection, retrain on all data with best params
alphas_cv = cellfun(@(s) s.Alpha, bestParamsList);
lambdas_cv = cellfun(@(s) s.Lambda, bestParamsList);
bestAlpha = mode(alphas_cv);
bestLambda = mode(lambdas_cv);

[B, FitInfo] = lassoglm(X_train_norm, Y_train, 'normal', ...
                        'Alpha', bestAlpha, 'Lambda', bestLambda, ...
                        'Standardize', false);

coef = [FitInfo.Intercept; B];
X_test_aug = [ones(size(X_test_norm,1),1), X_test_norm];
Y_pred_test = X_test_aug * coef;

rmse_test = sqrt(mean((Y_test - Y_pred_test).^2));
mae_test = mean(abs(Y_test - Y_pred_test));
r2_test = 1 - sum((Y_test - Y_pred_test).^2) / sum((Y_test - mean(Y_test)).^2);

fprintf('\nTest set: RMSE = %.3f, MAE = %.3f, R^2 = %.3f\n', rmse_test, mae_test, r2_test);


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
title(sprintf('Elastic Net Regression (R^2 = %.3f)', r2_test));
grid on; refline(1,0);

