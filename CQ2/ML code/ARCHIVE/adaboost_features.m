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
feature_names = train_tbl.Properties.VariableNames(feature_cols);

X_train = train_tbl{:, feature_cols};
Y_train = train_tbl{:, target_col};
X_test  = test_tbl{:, feature_cols};
Y_test  = test_tbl{:, target_col};

% ---- 3. Standardize features (optional for trees) ----

[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

% ---- 4. Set up hyperparameter grids ----

% Nested CV for AdaBoost (LSBoost)
% Inputs: X_train_norm, Y_train
% Fixed grids and CV settings
numLearners_grid = [20, 40, 80, 120];  % Number of boosting iterations
learnRate_grid   = [0.05, 0.1, 0.2, 0.4]; % Learning rate
minLeaf_grid     = [1, 3, 6, 10];           % Min leaf size for weak learners
outerK = 5;
innerK = 3;
outerCV = cvpartition(size(X_train_norm,1), 'KFold', outerK);

all_outer_r2    = zeros(outerK,1);
bestParamsList  = cell(outerK,1);

fprintf('\n==== Starting AdaBoost Nested Cross-Validation ====\n');
for i = 1:outerK
    fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
    trainIdx = training(outerCV, i);
    valIdx   = test(outerCV, i);
    Xtr_outer = X_train_norm(trainIdx, :);
    Ytr_outer = Y_train(trainIdx);
    Xval_outer = X_train_norm(valIdx, :);
    Yval_outer = Y_train(valIdx);

    % Inner CV for hyperparameter tuning
    innerCV = cvpartition(size(Xtr_outer,1), 'KFold', innerK);
    best_inner_r2 = -Inf;
    best_param = struct('NumLearners',[], 'LearnRate',[], 'MinLeaf',[]);

    for nL = numLearners_grid
        for lr = learnRate_grid
            for ml = minLeaf_grid
                inner_r2s = zeros(innerK,1);
                for j = 1:innerK
                    trIdx = training(innerCV, j);
                    teIdx = test(innerCV, j);

                    Xtr_inner = Xtr_outer(trIdx, :);
                    Ytr_inner = Ytr_outer(trIdx);
                    Xval_inner = Xtr_outer(teIdx, :);
                    Yval_inner = Ytr_outer(teIdx);

                    t = templateTree('MinLeafSize', ml);
                    mdl = fitrensemble(Xtr_inner, Ytr_inner, ...
                        'Method','LSBoost', ...
                        'NumLearningCycles', nL, ...
                        'LearnRate', lr, ...
                        'Learners', t);

                    Ypred_inner = predict(mdl, Xval_inner);
                    inner_r2s(j) = 1 - sum((Yval_inner - Ypred_inner).^2) / sum((Yval_inner - mean(Yval_inner)).^2);
                end
                avg_r2 = mean(inner_r2s);
                if avg_r2 > best_inner_r2
                    best_inner_r2 = avg_r2;
                    best_param.NumLearners = nL;
                    best_param.LearnRate   = lr;
                    best_param.MinLeaf     = ml;
                end
            end
        end
    end

    % Train on full outer training with best hyperparams
    t_final = templateTree('MinLeafSize', best_param.MinLeaf);
    model_final = fitrensemble(Xtr_outer, Ytr_outer, ...
        'Method','LSBoost', ...
        'NumLearningCycles', best_param.NumLearners, ...
        'LearnRate', best_param.LearnRate, ...
        'Learners', t_final);

    % Evaluate on outer validation
    Ypred_outer = predict(model_final, Xval_outer);
    outer_r2 = 1 - sum((Yval_outer - Ypred_outer).^2) / sum((Yval_outer - mean(Yval_outer)).^2);

    all_outer_r2(i) = outer_r2;
    bestParamsList{i} = best_param;

    fprintf('>> Fold %d best: Learners=%d, LR=%.3f, MinLeaf=%d | Outer R^2=%.4f\n', ...
        i, best_param.NumLearners, best_param.LearnRate, best_param.MinLeaf, outer_r2);
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
