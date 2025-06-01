%% =========================================================================
% Purpose:
%   This script runs nested cross-validation regression analyses using 
%   generative latent features (from a VAE) for FC, on selected follow-up scores.
%   It evaluates Random Forest, Elastic Net, and SVM models with different kernels,
%   reporting all metrics for direct model comparison.
%
% Folder structure expected:
%   - This script is placed alongside the VAE embedding files
%   - Patient IDs and training targets are loaded from "data cleanup and management/final files"
%
% Key Notes:
%   - Only the training set is used for all model evaluation to avoid leakage;
%     test data are not referenced here.
%   - Features are VAE latent variables matched by patient ID.
%
%% =========================================================================

clc; clear; close all
rng(6, "twister")

% IMPORTANT:
% The three target scores are 'CDSOB_followUp', 'GDTOTAL_followUp',
% 'MMSCORE_followUp'. Select one for running this script. 'MMSCORE_followUp'
% is used by default. You can set it below

targetScore = 'GDTOTAL_followUp'; % or'GDTOTAL_followUp' or 'MMSCORE_followUp'


%    We go two levels up to reach the “latent_results/vae_results” directory, where FC_dim10.mat
%    resides. That MAT-file must contain a structure fcData.vae_data.mu_latent,
%    which is an [n_latent_ids × latent_dim] matrix of embeddings.

scriptDir     = fileparts(mfilename('fullpath'));   % folder containing this script
parentLevel1  = fileparts(scriptDir);                % one level up
parentLevel2  = fileparts(parentLevel1);             % two levels up
vaeResultsDir = fullfile(parentLevel2, 'latent_results', 'vae_results');

if ~isfolder(vaeResultsDir)
    error('Directory not found: %s\nMake sure "latent_results/vae_results" is two levels above.', vaeResultsDir)
end

FC_path = fullfile(vaeResultsDir, 'FC_dim10.mat');
if ~isfile(FC_path)
    error('File not found: %s', FC_path)
end

fcData = load(FC_path);  % load the .mat file into a struct
mu_latent = fcData.vae_data.mu_latent;  % extract [n_latent_ids × latent_dim]

%  Load the list of unique patient IDs corresponding to mu_latent

idsDir   = fullfile('..', '..', 'data cleanup and management', 'utilities');
idsFile  = fullfile(idsDir, 'unique_patient_ids.csv');


%
% Read the CSV without assuming a header so we get exactly the first column
idsTbl    = readtable(idsFile, 'ReadVariableNames', false);
uniqueIDs = idsTbl{:,1};   % [n_latent_ids × 1] vector of patient identifiers

% Ensure the number of IDs matches the number of rows in mu_latent
nLatentIDs = numel(uniqueIDs);
nLatents   = size(mu_latent, 1);
if nLatentIDs ~= nLatents
    error('Mismatch between %d IDs and %d rows in mu_latent.', nLatentIDs, nLatents)
end


% Load the training feature file to obtain MMSCORE_followUp outcomes
featuresDir = fullfile('..', '..', 'data cleanup and management', 'final files');
trainFile   = fullfile(featuresDir, 'train_features_Q2_imputed.csv');

trainData = readtable(trainFile);     % read entire training table

% Extract the ID column (first col)
trainIDs = trainData{:,1};   % [n_train_ids × 1]


trainData = sortrows(trainData, 1); %sort rows in ascending order to match 
% the order of the ids in latent table
mu_latent_withIds = [uniqueIDs mu_latent]; % the order that the letents are stored
% is the same are the order in the unique ids list

% keep only the ids in the train set
mu_latent_withIds = mu_latent_withIds(ismember(mu_latent_withIds(:,1), trainIDs), :);

% keep only the col with the score for Y_train
scoreIdx = strcmp(trainData.Properties.VariableNames, targetScore);

Y_train_withID = trainData{:, [1, find(scoreIdx)]};

% final sets
X_train = mu_latent_withIds(:,2:end);
Y_train = Y_train_withID(:,2);


%%  Standardize features
%    In nested cross‐validation, feature scaling is performed within each
%    outer fold of the Elastic Net and SVM routines (each computes its own μ/σ).
%    Random Forests do not require any scaling, so we omit global z‐scoring here.

%%  nested-CV folds
outerK = 5;   
innerK = 3;   

%% Run Random Forest regression
fprintf('========== Random Forest Regression (nested CV: outerK=%d, innerK=%d) ==========\n', outerK, innerK);

[ all_outer_r2_rf, mean_outer_r2_rf, std_outer_r2_rf, bestParamsList_rf, bestParamsMode_rf ] = ...
    run_Random_Forest_Regression( X_train, Y_train, outerK, innerK );

%  summary
fprintf('\nRandom Forest Nested CV Results:\n');
fprintf('Per-fold R2 scores: [ %s ]\n', sprintf('%.4f ', all_outer_r2_rf));
fprintf('Mean R2 = %.4f, Std R2 = %.4f\n', mean_outer_r2_rf, std_outer_r2_rf);
fprintf('Best hyperparameters:\n');
fprintf('NumTrees = %d\n', bestParamsMode_rf.NumTrees);
fprintf('MinLeafSize = %d\n', bestParamsMode_rf.MinLeaf);
fprintf('MaxNumSplits = %d\n\n', bestParamsMode_rf.MaxNumSplits);
%%  Run Elastic Net regression
fprintf(' ========== Elastic Net Regression (nested CV: outerK=%d, innerK=%d)==========\n', outerK, innerK);

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
fprintf('Best hyperparameters:\n');
fprintf('Alpha = %.2f\n', bestAlpha);
fprintf('Lambda = %.5f\n\n', bestLambda);
%% Run SVM RBF 
fprintf(' ========== SVM Regression (RBF kernel, nested CV: outerK=%d, innerK=%d) ==========\n', outerK, innerK);

results_rbf = run_nested_cv_SVM( X_train, Y_train, 'rbf', outerK, innerK );

%  summary
fprintf('\nSVM (RBF) Nested CV Results:\n');
fprintf('Per-fold R2: [ %s ]\n', sprintf('%.4f ', results_rbf.outerR2));
fprintf('Per-fold RMSE: [ %s ]\n', sprintf('%.4f ', results_rbf.outerRMSE));
fprintf('Per-fold MAE: [ %s ]\n', sprintf('%.4f ', results_rbf.outerMAE));
fprintf('Mean R2 = %.4f\n', results_rbf.meanR2);
fprintf('Mean RMSE = %.4f\n', results_rbf.meanRMSE);
fprintf('Mean MAE = %.4f\n', results_rbf.meanMAE);
fprintf('Hyperparameters (mode):\n');
fprintf('C = %.4g\n', results_rbf.bestParamsMode.C);
fprintf('Epsilon = %.3f\n', results_rbf.bestParamsMode.epsilon);
fprintf('Sigma = %.4g\n\n', results_rbf.bestParamsMode.sigma);

%%  Run SVM with Polynomial kernel
fprintf(' ========== SVM Regression (Polynomial kernel, nested CV: outerK=%d, innerK=%d) ==========\n', outerK, innerK);

results_poly = run_nested_cv_SVM( X_train, Y_train, 'polynomial', outerK, innerK );

% final summary
fprintf('\nSVM (Polynomial) Nested CV Results:\n');
fprintf('Per-fold R2: [ %s ]\n', sprintf('%.4f ', results_poly.outerR2));
fprintf('Per-fold RMSE: [ %s ]\n', sprintf('%.4f ', results_poly.outerRMSE));
fprintf('Per-fold MAE: [ %s ]\n', sprintf('%.4f ', results_poly.outerMAE));
fprintf('Mean R2 = %.4f\n', results_poly.meanR2);
fprintf('Mean RMSE = %.4f\n', results_poly.meanRMSE);
fprintf('Mean MAE = %.4f\n', results_poly.meanMAE);
fprintf('Hyperparameters (mode):\n');
fprintf('C = %.4g\n', results_poly.bestParamsMode.C);
fprintf('Epsilon = %.3f\n', results_poly.bestParamsMode.epsilon);
fprintf('PolyOrder = %d\n\n', results_poly.bestParamsMode.PolyOrder);

%% Run SVM with Linear kernel
fprintf('========== SVM Regression (Linear kernel, nested CV: outerK=%d, innerK=%d) ==========\n', outerK, innerK);

results_lin = run_nested_cv_SVM( X_train, Y_train, 'linear', outerK, innerK );

%  summary 
fprintf('\nSVM (Linear) Nested CV Results:\n');
fprintf('Per-fold R2: [ %s ]\n', sprintf('%.4f ', results_lin.outerR2));
fprintf('Per-fold RMSE: [ %s ]\n', sprintf('%.4f ', results_lin.outerRMSE));
fprintf('Per-fold MAE: [ %s ]\n', sprintf('%.4f ', results_lin.outerMAE));
fprintf('Mean R2 = %.4f\n', results_lin.meanR2);
fprintf('Mean RMSE = %.4f\n', results_lin.meanRMSE);
fprintf('Mean MAE = %.4f\n', results_lin.meanMAE);
fprintf('Hyperparameters (mode):\n');
fprintf('C = %.4g\n', results_lin.bestParamsMode.C);
fprintf('Epsilon = %.3f\n\n', results_lin.bestParamsMode.epsilon);

%% 
fprintf('All models runs completed.\n');

%% Final comparison of mean R^2 for each model
fprintf('===== Model Comparison (Mean R^2) =====\n');
fprintf('Random Forest     Mean R2: %.4f\n', mean_outer_r2_rf);
fprintf('Elastic Net       Mean R2: %.4f\n', mean(all_outer_r2_elnet));
fprintf('SVM (RBF)         Mean R2: %.4f\n', results_rbf.meanR2);
fprintf('SVM (Polynomial)  Mean R2: %.4f\n', results_poly.meanR2);
fprintf('SVM (Linear)      Mean R2: %.4f\n\n', results_lin.meanR2);

