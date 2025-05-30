%--------------------------------------------------------------------------
% Script: adaboost_features.m
%
% Purpose:
%   - Predict MMSE using the feature dataset and Adaboost (ensemble) regression.
%   - Optimizes key hyperparameters via nested cross-validation.
%   - Reports best parameters and test set performance.
%
% Inputs:
%   - train_features_Q2_imputed.csv (features, imputed)
%   - test_features_Q2_imputed.csv  (test set, imputed)
%
% Output:
%   - Console: best parameters, test set R^2, RMSE, MAE.
%   - (Optional) Plots true vs predicted scores.
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

% ---- 3. Standardize features (optional for trees) ----

[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

% ---- 4. Set up hyperparameter grids ----

numLearners_grid = [20, 40, 80, 120];  % Number of boosting iterations
learnRate_grid = [0.05, 0.1, 0.2, 0.4]; % Learning rate
minLeaf_grid = [1, 3, 6, 10];           % Min leaf size for weak learners

outerK = 5;
innerK = 3;
outerCV = cvpartition(size(X_train_norm,1), 'KFold', outerK);

all_outer_r2 = zeros(outerK,1);
bestParamsList = cell(outerK,1);

fprintf('\n==== Starting AdaBoost Nested Cross-Validation ====\n');
for i = 1:outerK
    fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
    trainIdx = training(outerCV, i);
    valIdx   = test(outerCV, i);
    Xtr_outer = X_train_norm(trainIdx, :);
    Ytr_outer = Y_train(trainIdx);
    Xval_outer = X_train_norm(valIdx, :);
    Yval_outer = Y_train(valIdx);

    best_r2 = -Inf;

    for nL = numLearners_grid
        for lr = learnRate_grid
            for minLeaf = minLeaf_grid
                % Train ensemble
                t = templateTree('MinLeafSize', minLeaf);
                model = fitrensemble(Xtr_outer, Ytr_outer, ...
                    'Method', 'LSBoost', ...
                    'NumLearningCycles', nL, ...
                    'LearnRate', lr, ...
                    'Learners', t);

                Ypred_val = predict(model, Xval_outer);
                r2 = 1 - sum((Yval_outer - Ypred_val).^2) / sum((Yval_outer - mean(Yval_outer)).^2);

                fprintf('  Learners=%d, LR=%.3f, MinLeaf=%d | R^2=%.4f\n', nL, lr, minLeaf, r2);

                if r2 > best_r2
                    best_r2 = r2;
                    best_param = struct('NumLearners', nL, 'LearnRate', lr, 'MinLeaf', minLeaf);
                end
            end
        end
    end

    all_outer_r2(i) = best_r2;
    bestParamsList{i} = best_param;
    fprintf('>> Fold %d best: Learners=%d, LR=%.3f, MinLeaf=%d | Outer R^2=%.4f\n', ...
        i, best_param.NumLearners, best_param.LearnRate, best_param.MinLeaf, best_r2);
end

fprintf('\n=== Nested CV complete! ===\n');
fprintf('Outer fold R^2s: '); disp(all_outer_r2');

%% Find the mode (most frequent) best parameters
numLearners_cv = cellfun(@(s) s.NumLearners, bestParamsList);
learnRate_cv = cellfun(@(s) s.LearnRate, bestParamsList);
minLeaf_cv = cellfun(@(s) s.MinLeaf, bestParamsList);

bestNumLearners = mode(numLearners_cv);
bestLearnRate = mode(learnRate_cv);
bestMinLeaf = mode(minLeaf_cv);

fprintf('\nBest parameters (mode): Learners=%d, LR=%.3f, MinLeaf=%d\n', ...
    bestNumLearners, bestLearnRate, bestMinLeaf);

%% Retrain on all training data with best params
finalTree = templateTree('MinLeafSize', bestMinLeaf);
finalModel = fitrensemble(X_train_norm, Y_train, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', bestNumLearners, ...
    'LearnRate', bestLearnRate, ...
    'Learners', finalTree);

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
title(sprintf('AdaBoost Regression (R^2 = %.3f)', r2_test));
grid on; refline(1,0);

% Feature importance (optional)
imp = predictorImportance(finalModel);
[~, idx] = sort(imp, 'descend');
disp('Top 10 most important features in AdaBoost:');
for k = 1:min(10, numel(idx))
    fprintf('%2d. %-30s  Importance: %.4f\n', k, feature_names{idx(k)}, imp(idx(k)));
end
