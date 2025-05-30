%--------------------------------------------------------------------------
% Script: random_forest_features.m
%
% Purpose:
%   - Predict MMSE using feature dataset and Random Forest regression.
%   - Optimizes key hyperparameters via cross-validation.
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

clc; clear; close all;
rng(42, 'twister');
% ---- 1. Load Data ----

baseDir = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q3_imputed.csv');
testFile  = fullfile(baseDir, 'test_features_Q3_imputed.csv');

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

% ---- 4. Set up hyperparameter grids ----

numTrees_grid = [40, 80, 150, 200];      % Number of trees in the forest
minLeaf_grid = [1, 3, 5, 8, 12];         % Minimum leaf size
maxNumSplits_grid = [10, 50, 100, 200];  % Maximum number of splits

outerK = 5;
innerK = 3;
outerCV = cvpartition(size(X_train_norm,1), 'KFold', outerK);

all_outer_r2 = zeros(outerK,1);
bestParamsList = cell(outerK,1);

fprintf('\n==== Starting Random Forest Nested Cross-Validation ====\n');
for i = 1:outerK
    fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
    trainIdx = training(outerCV, i);
    valIdx   = test(outerCV, i);
    Xtr_outer = X_train_norm(trainIdx, :);
    Ytr_outer = Y_train(trainIdx);
    Xval_outer = X_train_norm(valIdx, :);
    Yval_outer = Y_train(valIdx);

    % --- Nested: Inner CV for hyperparameter selection ---
    best_inner_r2 = -Inf;
    best_inner_param = struct();

    % Grid search over hyperparameters
    for ntree = numTrees_grid
        for minLeaf = minLeaf_grid
            for maxSplit = maxNumSplits_grid
                % Prepare inner CV
                innerCV = cvpartition(length(Ytr_outer), 'KFold', innerK);
                inner_r2s = zeros(innerK,1);

                for j = 1:innerK
                    inner_trainIdx = training(innerCV, j);
                    inner_valIdx   = test(innerCV, j);

                    Xtr_inner = Xtr_outer(inner_trainIdx,:);
                    Ytr_inner = Ytr_outer(inner_trainIdx);
                    Xval_inner = Xtr_outer(inner_valIdx,:);
                    Yval_inner = Ytr_outer(inner_valIdx);

                    % Train model on inner train
                    model = fitrensemble(Xtr_inner, Ytr_inner, ...
                        'Method', 'Bag', ...
                        'NumLearningCycles', ntree, ...
                        'Learners', templateTree(...
                            'MinLeafSize', minLeaf, ...
                            'MaxNumSplits', maxSplit));

                    % Predict on inner val
                    Ypred_inner = predict(model, Xval_inner);

                    % Compute R^2 for this inner fold
                    r2_inner = 1 - sum((Yval_inner - Ypred_inner).^2) / sum((Yval_inner - mean(Yval_inner)).^2);
                    inner_r2s(j) = r2_inner;
                end

                avg_inner_r2 = mean(inner_r2s);

                % Save best hyperparams if this is the best avg R^2 so far
                if avg_inner_r2 > best_inner_r2
                    best_inner_r2 = avg_inner_r2;
                    best_inner_param = struct('NumTrees', ntree, ...
                                              'MinLeaf', minLeaf, ...
                                              'MaxNumSplits', maxSplit);
                end
                fprintf('  (Inner CV) Trees=%d, MinLeaf=%d, MaxSplit=%d | Mean R^2=%.4f\n', ...
                        ntree, minLeaf, maxSplit, avg_inner_r2);
            end
        end
    end

    % --- Train on full outer train set with best inner params ---
    model_final = fitrensemble(Xtr_outer, Ytr_outer, ...
        'Method', 'Bag', ...
        'NumLearningCycles', best_inner_param.NumTrees, ...
        'Learners', templateTree(...
            'MinLeafSize', best_inner_param.MinLeaf, ...
            'MaxNumSplits', best_inner_param.MaxNumSplits));

    % Evaluate on outer validation set
    Ypred_outer = predict(model_final, Xval_outer);
    outer_r2 = 1 - sum((Yval_outer - Ypred_outer).^2) / sum((Yval_outer - mean(Yval_outer)).^2);

    all_outer_r2(i) = outer_r2;
    bestParamsList{i} = best_inner_param;

    fprintf('>> Fold %d best params: Trees=%d, MinLeaf=%d, MaxSplit=%d | Outer R^2=%.4f\n', ...
        i, best_inner_param.NumTrees, best_inner_param.MinLeaf, best_inner_param.MaxNumSplits, outer_r2);
end

fprintf('\n==== Nested CV finished ====\n');
fprintf('Outer fold R^2 scores: '); disp(all_outer_r2');
fprintf('Mean Outer R^2: %.4f | Std: %.4f\n', mean(all_outer_r2), std(all_outer_r2));



%% Find the mode (most frequent) best parameters
numTrees_cv = cellfun(@(s) s.NumTrees, bestParamsList);
minLeaf_cv = cellfun(@(s) s.MinLeaf, bestParamsList);
maxSplit_cv = cellfun(@(s) s.MaxNumSplits, bestParamsList);

bestNumTrees = mode(numTrees_cv);
bestMinLeaf = mode(minLeaf_cv);
bestMaxSplit = mode(maxSplit_cv);

fprintf('\nBest parameters (mode): Trees=%d, MinLeaf=%d, MaxSplit=%d\n', bestNumTrees, bestMinLeaf, bestMaxSplit);





%% Retrain on all training data with best params
finalModel = fitrensemble(X_train_norm, Y_train, ...
    'Method', 'Bag', ...
    'NumLearningCycles', bestNumTrees, ...
    'Learners', templateTree(...
        'MinLeafSize', bestMinLeaf, ...
        'MaxNumSplits', bestMaxSplit));



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
title(sprintf('Random Forest Regression (R^2 = %.3f)', r2_test));
grid on; refline(1,0);



%%


% imp = predictorImportance(finalModel);
% 
% % The features used as input:
% feature_names = train_tbl.Properties.VariableNames;
% feature_names = feature_names(setdiff(1:width(train_tbl), [id_col, target_cols(2)]));
% 
% % Sort importance in descending order
% [~, idx] = sort(imp, 'descend');
% 
% % Display the top 10 features and their importance
% disp('Top 10 most important features in Random Forest:');
% for k = 1:min(10, numel(idx))
%     fprintf('%2d. %-30s  Importance: %.4f\n', k, feature_names{idx(k)}, imp(idx(k)));
% end
