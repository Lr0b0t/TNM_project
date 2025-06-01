%% =========================================================================
% Purpose:
%   This script assesses the predictive value of APOE genotype variables (APGEN1, APGEN2)
%   for clinical follow-up outcomes using nested cross-validation with
%   Random Forest, Elastic Net, and SVM regressors.
%
% Workflow and Data:
%   - Loads only APOE features (APGEN1, APGEN2) from the imputed training set.
%   - Regression targets are one-year follow-up scores: 'CDSOB_followUp',
%     'GDTOTAL_followUp', or 'MMSCORE_followUp'. Select target at the top.
%   - Only training data are used in nested CV to prevent data leakage.
% =========================================================================

clc; clear; close all;
%Set random seed for reproducibility
rng(6, 'twister');
% IMPORTANT:
% The three target scores are 'CDSOB_followUp', 'GDTOTAL_followUp',
% 'MMSCORE_followUp'. Select one for running this script. 'MMSCORE_followUp'
% is used by default. You can set it below

targetScore = 'GDTOTAL_followUp'; % or'GDTOTAL_followUp' or 'MMSCORE_followUp'

%%  Load and parse 
%    We assume “data cleanup and management/final files” is two levels up.
baseDir   = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q3_imputed.csv');

% Read tables (test set is loaded but not used in nested CV)
train_tbl = readtable(trainFile);

% Identify columns: ID is first column, targets in this study are three
target_names = {'MMSCORE_followUp','CDSOB_followUp','GDTOTAL_followUp'};
target_cols  = find( ismember(train_tbl.Properties.VariableNames, target_names) );
id_col       = 1;  % first column is ID

% we only need to keep the columns that have the APGEN1, APGEN2. These two
% will be our features. In the 'train_features_Q3_imputed.csv' there are
% also the rest of the features for completeness, but they will not be used
% in this script.
apgen1_col = find( ismember(train_tbl.Properties.VariableNames, 'APGEN1') );

% Define feature columns as just the APGEN columns.
feature_cols = setdiff(1:width(train_tbl), [id_col:(apgen1_col-1), target_cols]);

% Extract feature matrix and the chosen target 
X_train = train_tbl{:, feature_cols};
Y_train = train_tbl{:, strcmp(train_tbl.Properties.VariableNames, targetScore)};

%  Standardize features
% In nested cross‐validation, feature scaling is performed within each
% outer fold of the Elastic Net and SVM routines (each computes its own μ/σ).
% Random Forests do not require any scaling, so we omit global z‐scoring here.

%%  Specify nested-CV folds
outerK = 5;   % number of outer folds
innerK = 3;   % number of inner folds

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


%% Final comparison of mean R^2 for each model
fprintf('===== Model Comparison (Mean R^2) =====\n');
fprintf('Random Forest     Mean R2: %.4f\n', mean_outer_r2_rf);
fprintf('Elastic Net       Mean R2: %.4f\n', mean(all_outer_r2_elnet));
fprintf('SVM (RBF)         Mean R2: %.4f\n', results_rbf.meanR2);
fprintf('SVM (Polynomial)  Mean R2: %.4f\n', results_poly.meanR2);
fprintf('SVM (Linear)      Mean R2: %.4f\n\n', results_lin.meanR2);


fprintf('All models runs completed.\n');

