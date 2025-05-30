%--------------------------------------------------------------------------
% Script: randomforest_fc_regression.m
%
% Purpose:
%   - Predict MMSE using Random Forest regression, using FC matrices as input.
%   - Dimensionality reduction: vectorized, PCA, or graph-theory features.
%   - Nested CV (R^2 selection, NO leakage). Final model tested on external test set.
%
% Inputs:
%   - train_features_Q2_imputed.csv, test_features_Q2_imputed.csv
%   - connectivity_n88/<ID>/rdcm_connectivity.mat (with 'output_m_all')
%
% Usage:
%   Set 'dim_reduc' to 'vectorized', 'pca', or 'graph' below.
%--------------------------------------------------------------------------

clc; clear; close all;

%% ---- USER OPTION: choose dim reduction type ----
dim_reduc = 'vectorized';  % 'vectorized', 'pca', or 'graph'

%% ---- PATH HANDLING ----
baseDir = pwd;
featuresDir = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(featuresDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(featuresDir, 'test_features_Q2_imputed.csv');

connDir = fullfile(baseDir, 'connectivity_n88');
if ~exist(connDir, 'dir')
    connDir = fullfile(baseDir, '..','..', 'connectivity_n88');
end
if ~exist(connDir, 'dir')
    error('Could not find connectivity_n88 folder.');
end

%% ---- LOAD DATA ----
trainData = readtable(trainFile);
testData  = readtable(testFile);

id_col = 1;
mmse_col = find(strcmp(trainData.Properties.VariableNames, 'MMSCORE_followUp'));
train_ids = trainData{:, id_col};
test_ids  = testData{:, id_col};
Y_train = trainData{:, mmse_col};
Y_test  = testData{:, mmse_col};
pad_id = @(id) sprintf('%07d', id);

fprintf('\nLoading TRAIN FC matrices...\n');
fc_train = load_fc_matrices(train_ids, connDir, pad_id);
fprintf('Loading TEST FC matrices...\n');
fc_test  = load_fc_matrices(test_ids,  connDir, pad_id);

%% ---- FEATURE EXTRACTION ----
switch lower(dim_reduc)
    case 'vectorized'
        [X_train, feature_labels, idx_mask] = vectorize_rdcm_train(fc_train);
        X_test  = vectorize_rdcm_test(fc_test, idx_mask);
    case 'pca'
        [X_train, feature_labels, idx_mask] = vectorize_rdcm_train(fc_train);
        X_test  = vectorize_rdcm_test(fc_test, idx_mask);
        [X_train_norm, mu, sigma] = zscore(X_train);
        X_test_norm = (X_test - mu) ./ sigma;
        [coeff, score_train, ~, ~, explained] = pca(X_train_norm);
        cumExplained = cumsum(explained);
        numPC = find(cumExplained >= 95, 1, 'first');
        if isempty(numPC), numPC = length(explained); end
        fprintf('Retaining %d PCs (%.1f%% variance)\n', numPC, cumExplained(numPC));
        X_train = score_train(:, 1:numPC);
        X_test  = (X_test_norm * coeff(:,1:numPC));
        feature_labels = arrayfun(@(i) sprintf('PC%d',i), 1:numPC, 'UniformOutput', false);
    case 'graph'
        X_train = extract_graph_features(fc_train);
        X_test  = extract_graph_features(fc_test);
        feature_labels = graph_feature_labels(size(fc_train{1},1));
    otherwise
        error('Unknown dim_reduc: %s', dim_reduc);
end

%% === NESTED CROSS-VALIDATION (Random Forest) ===
outerK = 5;
innerK = 3;
outerCV = cvpartition(length(Y_train), 'KFold', outerK);

numTrees_vals = [30, 60, 100];
minLeaf_vals = [1, 3, 5, 10];
fprintf('\n===== Starting Nested CV (Random Forest, R^2 selection) =====\n');
all_outer_r2 = zeros(outerK, 1);
all_outer_rmse = zeros(outerK, 1);
all_params = cell(outerK, 1);

for i = 1:outerK
    fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
    trainIdx = training(outerCV, i);
    valIdx = test(outerCV, i);

    Xtr_outer = X_train(trainIdx,:);
    Ytr_outer = Y_train(trainIdx);
    Xval_outer = X_train(valIdx,:);
    Yval_outer = Y_train(valIdx);

    % --- Standardize if PCA, else no need (trees handle scale) ---
    if strcmp(dim_reduc,'pca')
        [Xtr_outer_norm, mu, sigma] = zscore(Xtr_outer);
        Xval_outer_norm = (Xval_outer - mu) ./ sigma;
    else
        Xtr_outer_norm = Xtr_outer;
        Xval_outer_norm = Xval_outer;
    end

    % --- Inner CV for hyperparams ---
    bestR2 = -Inf;
    for nt = numTrees_vals
        for ml = minLeaf_vals
            innerCV = cvpartition(length(Ytr_outer), 'KFold', innerK);
            inner_r2s = zeros(innerK,1);
            for j = 1:innerK
                trIdx = training(innerCV, j);
                vaIdx = test(innerCV, j);
                Xtr_in = Xtr_outer_norm(trIdx,:);
                Ytr_in = Ytr_outer(trIdx);
                Xva_in = Xtr_outer_norm(vaIdx,:);
                Yva_in = Ytr_outer(vaIdx);

                % Train random forest regression
                rf = TreeBagger(nt, Xtr_in, Ytr_in, ...
                    'Method', 'regression', ...
                    'MinLeafSize', ml, ...
                    'OOBPrediction','off', ...
                    'OOBPredictorImportance', 'on', ...  
                    'PredictorSelection','allsplits','NumPrint',0);


                Y_pred_in = predict(rf, Xva_in);
                sse = sum((Yva_in - Y_pred_in).^2);
                sst = sum((Yva_in - mean(Yva_in)).^2);
                r2 = 1 - sse / sst;
                inner_r2s(j) = r2;
            end
            mean_r2 = mean(inner_r2s);
            if mean_r2 > bestR2
                bestR2 = mean_r2;
                bestNumTrees = nt;
                bestMinLeaf = ml;
            end
        end
    end
    fprintf('Best inner params: NumTrees=%d, MinLeaf=%d (Inner mean R^2=%.4f)\n', bestNumTrees, bestMinLeaf, bestR2);

    % --- Retrain on all outer train with best params ---
    rf_final = TreeBagger(bestNumTrees, Xtr_outer_norm, Ytr_outer, ...
    'Method', 'regression', ...
    'MinLeafSize', bestMinLeaf, ...
    'OOBPrediction','off', ...
    'OOBPredictorImportance', 'on', ...   
    'PredictorSelection','allsplits','NumPrint',0);


    Ypred_outer = predict(rf_final, Xval_outer_norm);

    % --- Outer fold metrics ---
    rmse_outer = sqrt(mean((Yval_outer - Ypred_outer).^2));
    sse = sum((Yval_outer - Ypred_outer).^2);
    sst = sum((Yval_outer - mean(Yval_outer)).^2);
    r2_outer = 1 - sse / sst;
    fprintf('  Outer fold RMSE = %.3f, R^2 = %.3f\n', rmse_outer, r2_outer);

    all_outer_r2(i) = r2_outer;
    all_outer_rmse(i) = rmse_outer;
    all_params{i} = struct('NumTrees', bestNumTrees, 'MinLeaf', bestMinLeaf);
end

fprintf('\n===== Nested CV Complete =====\n');
fprintf('Mean outer RMSE: %.3f\n', mean(all_outer_rmse));
fprintf('Mean outer R^2: %.3f\n', mean(all_outer_r2));
fprintf('Summary (held-out folds):\n');
disp(all_outer_r2');

%% === FINAL TRAIN/TEST FIT AND REPORT ===
% Use most frequent hyperparams from outer folds
bestNumTrees = mode(cellfun(@(s) s.NumTrees, all_params));
bestMinLeaf = mode(cellfun(@(s) s.MinLeaf, all_params));
fprintf('\n--- FINAL MODEL: NumTrees=%d, MinLeaf=%d ---\n', bestNumTrees, bestMinLeaf);

if strcmp(dim_reduc,'pca')
    [Xz, mu, sigma] = zscore(X_train);
    Xz_test = (X_test - mu) ./ sigma;
else
    Xz = X_train;
    Xz_test = X_test;
end

rf = TreeBagger(bestNumTrees, Xz, Y_train, ...
    'Method', 'regression', ...
    'MinLeafSize', bestMinLeaf, ...
    'OOBPrediction','off', ...
    'OOBPredictorImportance', 'on', ...  
    'PredictorSelection','allsplits','NumPrint',0);

Y_pred_test = predict(rf, Xz_test);

rmse_test = sqrt(mean((Y_test - Y_pred_test).^2));
mae_test = mean(abs(Y_test - Y_pred_test));
sse_test = sum((Y_test - Y_pred_test).^2);
sst_test = sum((Y_test - mean(Y_test)).^2);
r2_test = 1 - sse_test / sst_test;

fprintf('\n--- TEST SET PERFORMANCE (external) ---\n');
fprintf('Test RMSE: %.3f\n', rmse_test);
fprintf('Test MAE : %.3f\n', mae_test);
fprintf('Test R^2 : %.3f\n', r2_test);

% Baseline/null model for test set
Y_test_baseline = mean(Y_train) * ones(size(Y_test));
rmse_base = sqrt(mean((Y_test - Y_test_baseline).^2));
r2_base = 1 - sum((Y_test - Y_test_baseline).^2) / sum((Y_test - mean(Y_test)).^2);
fprintf('Null-model R^2 (test): %.3f\n', r2_base);

% Optional: plot predicted vs actual
figure;
scatter(Y_test, Y_pred_test, 'filled');
xlabel('Actual MMSE (FollowUp)');
ylabel('Predicted MMSE');
title(sprintf('Random Forest Regression (R^2 = %.3f)', r2_test));
grid on; refline(1,0);

% Optional: Show OOB variable importance (if not using PCA)
if ~strcmp(dim_reduc, 'pca')
    imp = rf.OOBPermutedPredictorDeltaError;
    [~, idx] = sort(imp, 'descend');
    topK = min(10, length(idx));
    fprintf('\nTop %d features by OOB importance:\n', topK);
    for k = 1:topK
        fprintf('%2d. %-30s Importance: %.4f\n', k, feature_labels{idx(k)}, imp(idx(k)));
    end
    figure;
    bar(imp(idx(1:topK)));
    set(gca, 'XTickLabel', feature_labels(idx(1:topK)), 'XTick', 1:topK, 'XTickLabelRotation', 45);
    ylabel('OOB Importance');
    title('Top Features (Random Forest)');
end

%% --- Helper Functions: Same as in your code ---
% Use your previous implementations for load_fc_matrices, vectorize_rdcm_train, etc.

% ... (copy your previous helper functions here)



function matrices = load_fc_matrices(ids, connDir, pad_id)
    matrices = cell(length(ids),1);
    for i = 1:length(ids)
        folderName = pad_id(ids(i));
        matFile = fullfile(connDir, folderName, 'rdcm_connectivity.mat');
        if exist(matFile, 'file')
            data = load(matFile);
            if isfield(data, 'output_m_all')
                matrices{i} = data.output_m_all;
            else
                error('output_m_all variable not found in %s', matFile);
            end
        else
            error('File not found: %s', matFile);
        end
    end
end

function [X, feature_labels, idx_mask] = vectorize_rdcm_train(matrices)
    % Only use train matrices to create the mask!
    N = size(matrices{1}, 1);
    mask = ~eye(N);  % Off-diagonal elements only
    % Find any entry that's nonzero across the TRAINING SET
    nonzero_mask = false(N,N);
    for i = 1:length(matrices)
        nonzero_mask = nonzero_mask | (matrices{i} ~= 0);
    end
    mask = mask & nonzero_mask; % Only keep nonzero off-diag elements
    idx_mask = find(mask);
    X = zeros(length(matrices), length(idx_mask));
    for i = 1:length(matrices)
        X(i,:) = matrices{i}(idx_mask);
    end
    [row, col] = ind2sub([N,N], idx_mask);
    feature_labels = arrayfun(@(i,j) sprintf('Conn_%d_%d',i,j), row, col, 'UniformOutput', false);
end

function X = vectorize_rdcm_test(matrices, idx_mask)
    % Apply the mask from the TRAINING SET!
    X = zeros(length(matrices), length(idx_mask));
    for i = 1:length(matrices)
        X(i,:) = matrices{i}(idx_mask);
    end
end

function labels = vectorized_labels(N)
    mask = triu(true(N), 1);
    [row, col] = find(mask);
    labels = arrayfun(@(i,j) sprintf('Conn_%d_%d',i,j), row, col, 'UniformOutput', false);
end

function Xg = extract_graph_features(matrices)
    N = size(matrices{1}, 1);
    Xg = zeros(length(matrices), N*4 + 2); % strength, clustering, local eff, degree, + 2 global
    for i = 1:length(matrices)
        A = matrices{i};
        A(A < 0) = 0;
        if exist('strengths_und', 'file')
            node_strength = strengths_und(A); % 1 x N
        else
            node_strength = sum(A,2)';
        end
        if exist('clustering_coef_wu', 'file')
            clustering = clustering_coef_wu(A);
        else
            clustering = zeros(1,N);
        end
        if exist('efficiency_wei', 'file')
            local_eff = efficiency_wei(A, 1);
            global_eff = efficiency_wei(A);
        else
            local_eff = zeros(1,N);
            global_eff = 0;
        end
        degree = sum(A > 0, 2)';
        mean_strength = mean(node_strength);
        Xg(i,:) = [node_strength, clustering, local_eff, degree, global_eff, mean_strength];
    end
end

function labels = graph_feature_labels(N)
    labels = [arrayfun(@(i) sprintf('Strength_%d', i), 1:N, 'UniformOutput',false), ...
              arrayfun(@(i) sprintf('Clustering_%d', i), 1:N, 'UniformOutput',false), ...
              arrayfun(@(i) sprintf('LocalEff_%d', i), 1:N, 'UniformOutput',false), ...
              arrayfun(@(i) sprintf('Degree_%d', i), 1:N, 'UniformOutput',false), ...
              {'GlobalEff', 'MeanStrength'}];
end
