%--------------------------------------------------------------------------
% Script: elasticnet_fc_regression.m
%
% Purpose:
%   - Predict MMSE using elastic net regression, using FC matrices as input.
%   - Dimensionality reduction: vectorized, PCA, or graph-theory features.
%   - Nested CV (R^2 selection, NO leakage). Final model tested on external test set.
%
% Inputs:
%   - train_features_Q2_imputed.csv, test_features_Q2_imputed.csv
%   - connectivity_n88/<ID>/func_connectivity.mat (with 'fc_mat')
%
% Usage:
%   Set 'dim_reduc' to 'vectorized', 'pca', or 'graph' below.
%--------------------------------------------------------------------------

clc; clear; close all;

%% ---- USER OPTION: choose dim reduction type ----
dim_reduc = 'graph';  
% 'vectorized', 'pca', or 'graph'

%% ---- PATH HANDLING ----
% Find baseDir (where this script lives)
baseDir = pwd;

% Path to features CSVs (relative to here)
featuresDir = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(featuresDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(featuresDir, 'test_features_Q2_imputed.csv');

% Path to connectivity
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
        X_train = vectorize_fc(fc_train);
        X_test  = vectorize_fc(fc_test);
        feature_labels = vectorized_labels(size(fc_train{1}));
    case 'pca'
        X_train = vectorize_fc(fc_train);
        X_test  = vectorize_fc(fc_test);
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

%% === NESTED CROSS-VALIDATION ===
outerK = 5;
innerK = 3;
outerCV = cvpartition(length(Y_train), 'KFold', outerK);

alpha_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1]; % Elastic net/LASSO
fprintf('\n===== Starting Nested CV (Elastic Net, R^2 selection) =====\n');
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

    % --- Standardize using ONLY outer train (for each fold) ---
    [Xtr_outer_norm, mu, sigma] = zscore(Xtr_outer);
    Xval_outer_norm = (Xval_outer - mu) ./ sigma;

    % --- Inner CV for hyperparams ---
    bestR2 = -Inf;
    for a = alpha_vals
        [B, FitInfo] = lassoglm(Xtr_outer_norm, Ytr_outer, 'normal', ...
            'CV', innerK, 'Alpha', a, 'Standardize', false);
        idx = FitInfo.IndexMinDeviance;
        Y_pred_cv = Xtr_outer_norm * B(:,idx) + FitInfo.Intercept(idx);
        sse = sum((Ytr_outer - Y_pred_cv).^2);
        sst = sum((Ytr_outer - mean(Ytr_outer)).^2);
        r2_cv = 1 - sse / sst;
        if r2_cv > bestR2
            bestR2 = r2_cv;
            bestB = B(:,idx);
            bestIntercept = FitInfo.Intercept(idx);
            bestAlpha = a;
            bestLambda = FitInfo.Lambda(idx);
        end
    end
    fprintf('Best inner params: alpha=%.2f, lambda=%.4g (Inner mean R^2=%.4f)\n', bestAlpha, bestLambda, bestR2);

    % --- Retrain on all outer train with best params ---
    [Bfinal, FitInfoFinal] = lassoglm(Xtr_outer_norm, Ytr_outer, 'normal', ...
        'Alpha', bestAlpha, 'Lambda', bestLambda, 'Standardize', false);
    coef = Bfinal;
    intercept = FitInfoFinal.Intercept;

    % --- Predict on held-out fold (never seen in any inner CV) ---
    Ypred_outer = Xval_outer_norm * coef + intercept;

    % --- Outer fold metrics ---
    rmse_outer = sqrt(mean((Yval_outer - Ypred_outer).^2));
    sse = sum((Yval_outer - Ypred_outer).^2);
    sst = sum((Yval_outer - mean(Yval_outer)).^2);
    r2_outer = 1 - sse / sst;
    fprintf('  Outer fold RMSE = %.3f, R^2 = %.3f\n', rmse_outer, r2_outer);

    all_outer_r2(i) = r2_outer;
    all_outer_rmse(i) = rmse_outer;
    all_params{i} = struct('alpha', bestAlpha, 'lambda', bestLambda);
end

fprintf('\n===== Nested CV Complete =====\n');
fprintf('Mean outer RMSE: %.3f\n', mean(all_outer_rmse));
fprintf('Mean outer R^2: %.3f\n', mean(all_outer_r2));
fprintf('Summary: (these numbers are from **completely held-out folds**)\n');
disp(all_outer_r2');

%% === FINAL TRAIN/TEST FIT AND REPORT ===
% Use most frequent alpha/lambda from outer folds
bestAlpha = mode(cellfun(@(s) s.alpha, all_params));
bestLambda = mean(cellfun(@(s) s.lambda, all_params));
fprintf('\n--- FINAL MODEL: alpha=%.2f, lambda=%.4g ---\n', bestAlpha, bestLambda);

[Xz, mu, sigma] = zscore(X_train);
[B, FitInfo] = lassoglm(Xz, Y_train, 'normal', ...
    'Alpha', bestAlpha, 'Lambda', bestLambda, 'Standardize', false);
coef = B;
intercept = FitInfo.Intercept;

% --- Prepare test set (using ONLY train mu/sigma) ---
Xz_test = (X_test - mu) ./ sigma;

% --- Predict and report ---
Y_pred_test = Xz_test * coef + intercept;

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
title(sprintf('Elastic Net Regression (R^2 = %.3f)', r2_test));
grid on; refline(1,0);

% Show top 10 features by |coef|
[~, idx] = sort(abs(coef), 'descend');
fprintf('\nTop 10 features by |coef| in final (all-data) model:\n');
for k = 1:min(10, numel(idx))
    fprintf('%2d. %-30s Coef: %.4f\n', k, feature_labels{idx(k)}, coef(idx(k)));
end

%% --- Helper Functions ---

function matrices = load_fc_matrices(ids, connDir, pad_id)
    matrices = cell(length(ids),1);
    for i = 1:length(ids)
        folderName = pad_id(ids(i));
        matFile = fullfile(connDir, folderName, 'func_connectivity.mat');
        if exist(matFile, 'file')
            data = load(matFile);
            if isfield(data, 'fc_mat')
                matrices{i} = data.fc_mat;
            else
                error('fc_mat variable not found in %s', matFile);
            end
        else
            error('File not found: %s', matFile);
        end
    end
end

function X = vectorize_fc(matrices)
    N = size(matrices{1}, 1);
    mask = triu(true(N), 1);
    X = zeros(length(matrices), sum(mask(:)));
    for i = 1:length(matrices)
        X(i, :) = matrices{i}(mask);
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
