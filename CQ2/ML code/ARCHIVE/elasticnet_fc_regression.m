clc; clear; close all;
% Find baseDir (where this script lives)
baseDir = pwd;
dataDir = fullfile(baseDir, 'final files');

% Path to features CSVs (relative to here)
baseDir   = fullfile('..','..','data cleanup and management', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q2_imputed.csv');
% % 
trainData = readtable(trainFile);
connDir = fullfile(dataDir, 'connectivity_n88');

id_col = 1;
mmse_col = find(strcmp(trainData.Properties.VariableNames, 'MMSCORE_followUp'));
train_ids = trainData{:, id_col};
% test_ids  = testData{:, id_col};
Y_train = trainData{:, mmse_col};
% Helper function to zero-pad patient IDs to 7 digits
pad_id = @(id) sprintf('%07d', id);


fprintf('\nLoading TRAIN FC matrices...\n');
fc_train = load_fc_matrices(train_ids, connDir, pad_id);
fprintf('Loading TEST FC matrices...\n');
% fc_test  = load_fc_matrices(test_ids,  connDir, pad_id);


X_train = extract_graph_features(fc_train);
X_test  = extract_graph_features(fc_test);
feature_labels = graph_feature_labels(size(fc_train{1},1));


%%
[ all_outer_r2_elnet, all_outer_rmse_elnet, all_outer_mae_elnet, bestParamsList_elnet, bestAlpha, bestLambda ] = ...
    run_Elastic_Net_Regression( X_train, Y_train, outerK, innerK );

% summary 
fprintf('\nElastic Net Nested CV Results:\n');
fprintf('Per-fold R2 scores: [ %s ]\n', sprintf('%.4f ', all_outer_r2_elnet));
fprintf('Per-fold RMSE: [ %s ]\n', sprintf('%.4f ', all_outer_rmse_elnet));
fprintf('Per-fold MAE: [ %s ]\n', sprintf('%.4f ', all_outer_mae_elnet));
fprintf('Mean R2 = %.4f\n', mean(all_outer_r2_elnet));
fprintf('Mean RMSE = %.4f\n', mean(all_outer_rmse_elnet));
fprintf('Mean MAE = %.4f\n', mean(all_outer_mae_elnet));
fprintf('Most frequent hyperparameters:\n');
fprintf('Alpha = %.2f\n', bestAlpha);
fprintf('Lambda = %.5f\n\n', bestLambda);

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
