%%==========================================================================%
% Purpose:
%   For the questionnaire-derived features—this script loads
%   preprocessed training data, standardizes it, and then runs the set of
%   machine learning regressors (Random Forest, Elastic Net, and SVM with
%   RBF, Polynomial, and Linear kernels) using nested cross-validation.
%   At the end, it prints some summaries of each model’s performance
%   and the most final hyperparameters.
%
%   This file assumes that the folder structure is:
%        ├─ (current folder)
%        │   └─ run_all_models_questionnaire.m
%        └─ data cleanup and management
%            └─ final files
%                ├─ train_features_Q2_imputed.csv
%                └─ *(test file is not loaded here)*
%
%
%  This file executes nested CV for:
%             – Random Forest regression
%             – Elastic Net regression
%             – SVM (RBF), SVM (Polynomial), and SVM (Linear)
%
%  Notes:
%   - We use “MMSCORE_followUp” as the default outcome. To switch targets, adjust
%     the “target_names” list or manually set Y_train to a different column.
%   - There in no trace of any test data in this file, to prevent any data
%     leakage. The test files will only be used at the very end.
%
%%==========================================================================

clc; clear; close all;
%Set random seed for reproducibility
rng(2, 'twister');

%%  Load and parse 
%    We assume “data cleanup and management/final files” is two levels up.
baseDir   = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q2_imputed.csv');

% Read tables (test set is loaded but not used in nested CV)
train_tbl = readtable(trainFile);

% Identify columns: ID is first column, targets in this study are three
target_names = {'MMSCORE_followUp','CDSOB_followUp','GDTOTAL_followUp'};
target_cols  = find( ismember(train_tbl.Properties.VariableNames, target_names) );
id_col       = 1;  % first column is ID

% Define feature columns as everything except ID and any of the target columns
feature_cols = setdiff(1:width(train_tbl), [id_col, target_cols]);

% Extract feature matrix and the chosen target 
X_train = train_tbl{:, feature_cols};
Y_train = train_tbl{:, strcmp(train_tbl.Properties.VariableNames, 'MMSCORE_followUp')};

%%  Standardize features
%    Although Random Forests do not strictly need feature scaling, we
%    standardize to keep all models (especially SVM/Elastic Net) comparable.
[X_train_norm, mu, sigma] = zscore(X_train);

%%  Specify nested-CV folds
outerK = 5;   % number of outer folds
innerK = 3;   % number of inner folds

%% Run Random Forest regression
fprintf('========== Random Forest Regression (nested CV: outerK=%d, innerK=%d) ==========\n', outerK, innerK);

[ all_outer_r2_rf, mean_outer_r2_rf, std_outer_r2_rf, bestParamsList_rf, bestParamsMode_rf ] = ...
    run_Random_Forest_Regression( X_train_norm, Y_train, outerK, innerK );

%  summary
fprintf('\nRandom Forest Nested CV Results:\n');
fprintf('Per-fold R2 scores: [ %s ]\n', sprintf('%.4f ', all_outer_r2_rf));
fprintf('Mean R2 = %.4f, Std R2 = %.4f\n', mean_outer_r2_rf, std_outer_r2_rf);
fprintf('Most frequent hyperparameters:\n');
fprintf('NumTrees = %d\n', bestParamsMode_rf.NumTrees);
fprintf('MinLeafSize = %d\n', bestParamsMode_rf.MinLeaf);
fprintf('MaxNumSplits = %d\n\n', bestParamsMode_rf.MaxNumSplits);

%%  Run Elastic Net regression
fprintf(' ========== Elastic Net Regression (nested CV: outerK=%d, innerK=%d)==========\n', outerK, innerK);

[ all_outer_r2_elnet, all_outer_rmse_elnet, all_outer_mae_elnet, bestParamsList_elnet, bestAlpha, bestLambda ] = ...
    run_Elastic_Net_Regression( X_train_norm, Y_train, outerK, innerK );

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
%% Run SVM RBF 
fprintf(' ========== SVM Regression (RBF kernel, nested CV: outerK=%d, innerK=%d) ==========\n', outerK, innerK);

results_rbf = run_nested_cv_SVM( X_train_norm, Y_train, 'rbf', outerK, innerK );

%  summary
fprintf('\nSVM (RBF) Nested CV Results:\n');
fprintf('Per-fold R2: [ %s ]\n', sprintf('%.4f ', results_rbf.outerR2));
fprintf('Per-fold RMSE: [ %s ]\n', sprintf('%.4f ', results_rbf.outerRMSE));
fprintf('Per-fold MAE: [ %s ]\n', sprintf('%.4f ', results_rbf.outerMAE));
fprintf('Mean R2 = %.4f\n', results_rbf.meanR2);
fprintf('Mean RMSE = %.4f\n', results_rbf.meanRMSE);
fprintf('Mean MAE = %.4f\n', results_rbf.meanMAE);
fprintf('Most frequent hyperparameters:\n');
fprintf('C = %.4g\n', results_rbf.bestParamsMode.C);
fprintf('Epsilon = %.3f\n', results_rbf.bestParamsMode.epsilon);
fprintf('Sigma = %.4g\n\n', results_rbf.bestParamsMode.sigma);

%%  Run SVM with Polynomial kernel
fprintf(' ========== SVM Regression (Polynomial kernel, nested CV: outerK=%d, innerK=%d) ==========\n', outerK, innerK);

results_poly = run_nested_cv_SVM( X_train_norm, Y_train, 'polynomial', outerK, innerK );

% final summary
fprintf('\nSVM (Polynomial) Nested CV Results:\n');
fprintf('Per-fold R2: [ %s ]\n', sprintf('%.4f ', results_poly.outerR2));
fprintf('Per-fold RMSE: [ %s ]\n', sprintf('%.4f ', results_poly.outerRMSE));
fprintf('Per-fold MAE: [ %s ]\n', sprintf('%.4f ', results_poly.outerMAE));
fprintf('Mean R2 = %.4f\n', results_poly.meanR2);
fprintf('Mean RMSE = %.4f\n', results_poly.meanRMSE);
fprintf('Mean MAE = %.4f\n', results_poly.meanMAE);
fprintf('Most frequent hyperparameters:\n');
fprintf('C = %.4g\n', results_poly.bestParamsMode.C);
fprintf('Epsilon = %.3f\n', results_poly.bestParamsMode.epsilon);
fprintf('PolyOrder = %d\n\n', results_poly.bestParamsMode.PolyOrder);

%% Run SVM with Linear kernel
fprintf('========== SVM Regression (Linear kernel, nested CV: outerK=%d, innerK=%d) ==========\n', outerK, innerK);

results_lin = run_nested_cv_SVM( X_train_norm, Y_train, 'linear', outerK, innerK );

%  summary 
fprintf('\nSVM (Linear) Nested CV Results:\n');
fprintf('Per-fold R2: [ %s ]\n', sprintf('%.4f ', results_lin.outerR2));
fprintf('Per-fold RMSE: [ %s ]\n', sprintf('%.4f ', results_lin.outerRMSE));
fprintf('Per-fold MAE: [ %s ]\n', sprintf('%.4f ', results_lin.outerMAE));
fprintf('Mean R2 = %.4f\n', results_lin.meanR2);
fprintf('Mean RMSE = %.4f\n', results_lin.meanRMSE);
fprintf('Mean MAE = %.4f\n', results_lin.meanMAE);
fprintf('Most frequent hyperparameters:\n');
fprintf('C = %.4g\n', results_lin.bestParamsMode.C);
fprintf('Epsilon = %.3f\n\n', results_lin.bestParamsMode.epsilon);

%% 
fprintf('All models runs completed.\n');

