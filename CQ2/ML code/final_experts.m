clc; clear; close all;
rng(6, 'twister');

% the target are officially named as 'MMSCORE_followUp', 'CDSOB_followUp', 'GDTOTAL_followUp'


function interpret_elastic_net(feature_names, coefs, target_score)

% Display and plot all nonzero feature importances for an elastic net model
%
% Inputs:
%   feature_names : cell array of feature names (e.g., train_tbl.Properties.VariableNames(feature_cols))
%   coefs         : vector of model coefficients (excluding intercept; e.g., B from lassoglm)
%   target_score  : string, name of the target score (e.g., 'MMSCORE_followUp')
%

    nonzero_idx = find(coefs ~= 0);
    nNonZero = numel(nonzero_idx);
    if nNonZero == 0
        fprintf('No nonzero coefficients in the model for %s.\n', target_score);
        return
    end

    % Sort by absolute value, descending
    [~, sort_idx] = sort(abs(coefs(nonzero_idx)), 'descend');
    top_idx = nonzero_idx(sort_idx);

    fprintf('\nAll Nonzero Features (by |Elastic Net Coefficient|) for %s:\n', target_score);
    for i = 1:nNonZero
        fprintf('%2d. %-30s  Coefficient: %.4f\n', i, feature_names{top_idx(i)}, coefs(top_idx(i)));
    end

    % Horizontal bar plot
    figure('Name', ['ENET Importances: ', target_score]);
    barh(coefs(top_idx));
    set(gca, 'ytick', 1:nNonZero, 'yticklabel', feature_names(top_idx));
    xlabel('Elastic Net Coefficient');
    title(sprintf('Elastic Net Feature Importances for %s', strrep(target_score,'_','\_')));
    grid on;

end

function bootstrap_test_metrics(Y_test, Y_pred_test, target_score, nBoot)
% Calculate bootstrap CIs for RMSE, MAE, and R^2 on a test set
%
% Inputs:
%   Y_test      : Vector of true target values (test set)
%   Y_pred_test : Vector of predicted values (test set)
%   target_score: String, name of the target score (for reporting)
%   nBoot       : (Optional) Number of bootstrap iterations (default: 4k)
%

    if nargin < 4
        nBoot = 4000;
    end
    N = length(Y_test);

    boot_rmse = zeros(nBoot,1);
    boot_mae  = zeros(nBoot,1);
    boot_r2   = zeros(nBoot,1);

    for b = 1:nBoot
        idx = randsample(N, N, true);
        Y_true_b = Y_test(idx);
        Y_pred_b = Y_pred_test(idx);

        boot_rmse(b) = sqrt(mean((Y_true_b - Y_pred_b).^2));
        boot_mae(b)  = mean(abs(Y_true_b - Y_pred_b));
        boot_r2(b)   = 1 - sum((Y_true_b - Y_pred_b).^2) / sum((Y_true_b - mean(Y_true_b)).^2);
    end

    fprintf('\nBootstrap 95%% CI for RMSE (%s, n=%d): %.3f - %.3f\n', ...
        target_score, N, prctile(boot_rmse,2.5), prctile(boot_rmse,97.5));
    fprintf('Bootstrap 95%% CI for MAE  (%s, n=%d): %.3f - %.3f\n', ...
        target_score, N, prctile(boot_mae,2.5), prctile(boot_mae,97.5));
    fprintf('Bootstrap 95%% CI for R^2  (%s, n=%d): %.3f - %.3f\n', ...
        target_score, N, prctile(boot_r2,2.5), prctile(boot_r2,97.5));
    fprintf('Bootstrap means for %s (n=%d): RMSE = %.3f, MAE = %.3f, R^2 = %.3f\n', ...
        target_score, N, mean(boot_rmse), mean(boot_mae), mean(boot_r2));

end

function plot_predictions(Y_test, Y_pred_test, r2_test, target_score)
% Scatter plot of actual vs. predicted values with regression and reference lines.
%
% Inputs:
%   Y_test        : Vector of true target values (test set)
%   Y_pred_test   : Vector of predicted values (test set)
%   r2_test       : R^2 statistic on test set
%   target_score  : String, name of the target score (for labeling)
%


    figure;
    scatter(Y_test, Y_pred_test, 60, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;

    % Axis limits for clean plotting
    minVal = min([Y_test; Y_pred_test]);
    maxVal = max([Y_test; Y_pred_test]);
    xlim([minVal, maxVal]);
    ylim([minVal, maxVal]);

    % Perfect prediction line
    plot([minVal, maxVal], [minVal, maxVal], 'k--', 'LineWidth', 1.5);

    % Regression line (fit on test data)
    p = polyfit(Y_test, Y_pred_test, 1);
    xfit = linspace(minVal, maxVal, 100);
    yfit = polyval(p, xfit);
    plot(xfit, yfit, 'r-', 'LineWidth', 1.5);

    % Pearson correlation coefficient
    r_corr = corr(Y_test, Y_pred_test);

    % Annotate r and R^2
    text(0.05, 0.95, sprintf('r = %.2f\nR^2 = %.3f', r_corr, r2_test), ...
        'Units', 'normalized', 'FontSize', 13, 'FontWeight', 'bold', ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
        'BackgroundColor', 'white', 'Margin', 4, 'EdgeColor', [0.8 0.8 0.8]);

    xlabel(sprintf('Actual %s', strrep(target_score,'_','\_')), 'FontSize', 14, 'FontWeight', 'bold');
    ylabel(sprintf('Predicted %s', strrep(target_score,'_','\_')), 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('Elastic Net Regression for %s\n(Test set: n = %d)', ...
        strrep(target_score,'_','\_'), numel(Y_test)), 'FontSize', 15, 'FontWeight', 'bold');

    legend({'Predictions', 'Perfect (y = x)', 'Regression fit'}, 'Location', 'Best', 'FontSize', 12);
    grid on;
    set(gca, 'FontSize', 13, 'LineWidth', 1.2);

    hold off;
end

%% ================== case 1: MMSCORE_followUp ==================


% ---- Load Data ----
baseDir = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(baseDir, 'test_features_Q2_imputed.csv');

train_tbl = readtable(trainFile);
test_tbl  = readtable(testFile);

% ---- Identify feature and target columns --
target_names = {'MMSCORE_followUp', 'CDSOB_followUp', 'GDTOTAL_followUp'}; % this os need in
% order to remove the score cols from the raining features% Find indices
target_cols = find(ismember(train_tbl.Properties.VariableNames, target_names));
id_col = 1; % First column is assumed to be ID

feature_cols = setdiff(1:width(train_tbl), [id_col, target_cols]);


target_score = 'MMSCORE_followUp'; % !!! careful here!!!
X_train = train_tbl{:, feature_cols};
X_test  = test_tbl{:, feature_cols};
Y_train = train_tbl{:, strcmp(train_tbl.Properties.VariableNames, target_score)};
Y_test  = test_tbl{:, strcmp(test_tbl.Properties.VariableNames, target_score)};



% We take the parameters that were found best in the Nested CV, using the
% training set only. These parameters should be inputted manually.

bestAlpha = 1.00;
bestLambda = 0.37276;
fprintf('\n===== FINAL MODEL EVALUATION for MMSCORE_followUp =====\n');
fprintf('Retraining with Alpha=%.2f, Lambda=%.5f\n', bestAlpha, bestLambda);

% Standardize training set
[X_train_norm, mu_final, sigma_final] = zscore(X_train);
X_test_norm = (X_test - mu_final) ./ sigma_final;

[B, FitInfo] = lassoglm(X_train_norm, Y_train, 'normal', ...
                        'Alpha', bestAlpha, 'Lambda', bestLambda, ...
                        'Standardize', false);
coef = [FitInfo.Intercept; B];

X_test_aug = [ones(size(X_test_norm,1),1), X_test_norm];
Y_pred_test = X_test_aug * coef;

rmse_test = sqrt(mean((Y_test - Y_pred_test).^2));
mae_test = mean(abs(Y_test - Y_pred_test));
r2_test = 1 - sum((Y_test - Y_pred_test).^2) / sum((Y_test - mean(Y_test)).^2);

fprintf('Test set: RMSE = %.3f, MAE = %.3f, R^2 = %.3f\n', rmse_test, mae_test, r2_test);

% Baseline (predict mean)
baseline_pred = mean(Y_train) * ones(size(Y_test));
sse_base = sum((Y_test - baseline_pred).^2);
sst = sum((Y_test - mean(Y_test)).^2);
r2_base = 1 - sse_base / sst;
fprintf('Null-model R^2 (test): %.3f\n', r2_base);

feature_names = train_tbl.Properties.VariableNames(feature_cols);
% apply functions to for interpreting the results
interpret_elastic_net(feature_names, B, target_score)
bootstrap_test_metrics(Y_test, Y_pred_test, target_score);
plot_predictions(Y_test, Y_pred_test, r2_test, target_score);


clear; % clean the workspace to make sure there is no data pollution.
%% ================== case 2: CDSOB_followUp ==================


% ---- Load Data ----
baseDir = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(baseDir, 'test_features_Q2_imputed.csv');

train_tbl = readtable(trainFile);
test_tbl  = readtable(testFile);

% ---- Identify feature and target columns --
target_names = {'MMSCORE_followUp', 'CDSOB_followUp', 'GDTOTAL_followUp'}; % this os need in
% order to remove the score cols from the raining features

% Find indices
target_cols = find(ismember(train_tbl.Properties.VariableNames, target_names));
id_col = 1; % First column is assumed to be ID

feature_cols = setdiff(1:width(train_tbl), [id_col, target_cols]);



target_score = 'CDSOB_followUp'; % !!! careful here!!!
X_train = train_tbl{:, feature_cols};
X_test  = test_tbl{:, feature_cols};
Y_train = train_tbl{:, strcmp(train_tbl.Properties.VariableNames, target_score)};
Y_test  = test_tbl{:, strcmp(test_tbl.Properties.VariableNames, target_score)};


% We take the parameters that were found best in the Nested CV, using the
% training set only. These parameters should be inputted manually.

bestAlpha = 0.70;
bestLambda = 0.37276; 


fprintf('\n===== FINAL MODEL EVALUATION for CDSOB_followUp =====\n');
fprintf('Retraining with Alpha=%.2f, Lambda=%.5f\n', bestAlpha, bestLambda);

% Standardize training set
[X_train_norm, mu_final, sigma_final] = zscore(X_train);
X_test_norm = (X_test - mu_final) ./ sigma_final;

[B, FitInfo] = lassoglm(X_train_norm, Y_train, 'normal', ...
                        'Alpha', bestAlpha, 'Lambda', bestLambda, ...
                        'Standardize', false);
coef = [FitInfo.Intercept; B];

X_test_aug = [ones(size(X_test_norm,1),1), X_test_norm];
Y_pred_test = X_test_aug * coef;

rmse_test = sqrt(mean((Y_test - Y_pred_test).^2));
mae_test = mean(abs(Y_test - Y_pred_test));
r2_test = 1 - sum((Y_test - Y_pred_test).^2) / sum((Y_test - mean(Y_test)).^2);

fprintf('Test set: RMSE = %.3f, MAE = %.3f, R^2 = %.3f\n', rmse_test, mae_test, r2_test);

% Baseline (predict mean)
baseline_pred = mean(Y_train) * ones(size(Y_test));
sse_base = sum((Y_test - baseline_pred).^2);
sst = sum((Y_test - mean(Y_test)).^2);
r2_base = 1 - sse_base / sst;
fprintf('Null-model R^2 (test): %.3f\n', r2_base);

feature_names = train_tbl.Properties.VariableNames(feature_cols);
% apply functions to for interpreting the results
interpret_elastic_net(feature_names, B, target_score)
bootstrap_test_metrics(Y_test, Y_pred_test, target_score);
plot_predictions(Y_test, Y_pred_test, r2_test, target_score);




clear; % clean the workspace to make sure there is no data pollution.
%%  ================== case 3 : GDTOTAL_followUp ==================
% ---- 1. Load Data ----
baseDir = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(baseDir, 'test_features_Q2_imputed.csv');

train_tbl = readtable(trainFile);
test_tbl  = readtable(testFile);

% ----  Identify feature and target columns --
target_names = {'MMSCORE_followUp', 'CDSOB_followUp', 'GDTOTAL_followUp'}; % this os need in
% order to remove the score cols from the raining features
% Find indices
target_cols = find(ismember(train_tbl.Properties.VariableNames, target_names));
id_col = 1; % First column is assumed to be ID

feature_cols = setdiff(1:width(train_tbl), [id_col, target_cols]);



target_score = 'GDTOTAL_followUp'; % !!!
X_train = train_tbl{:, feature_cols};
X_test  = test_tbl{:, feature_cols};
Y_train = train_tbl{:, strcmp(train_tbl.Properties.VariableNames, target_score)};
Y_test  = test_tbl{:, strcmp(test_tbl.Properties.VariableNames, target_score)};



% We take the parameters that were found best in the Nested CV, using the
% training set only. These parameters should be inputted manually.

bestAlpha = 0.90;
bestLambda = 0.37276;
fprintf('\n===== FINAL MODEL EVALUATION for GDTOTAL_followUp =====\n');
fprintf('Retraining with Alpha=%.2f, Lambda=%.5f\n', bestAlpha, bestLambda);

% Standardize training set
[X_train_norm, mu_final, sigma_final] = zscore(X_train);
X_test_norm = (X_test - mu_final) ./ sigma_final;

[B, FitInfo] = lassoglm(X_train_norm, Y_train, 'normal', ...
                        'Alpha', bestAlpha, 'Lambda', bestLambda, ...
                        'Standardize', false);
coef = [FitInfo.Intercept; B];

X_test_aug = [ones(size(X_test_norm,1),1), X_test_norm];
Y_pred_test = X_test_aug * coef;

rmse_test = sqrt(mean((Y_test - Y_pred_test).^2));
mae_test = mean(abs(Y_test - Y_pred_test));
r2_test = 1 - sum((Y_test - Y_pred_test).^2) / sum((Y_test - mean(Y_test)).^2);

fprintf('Test set: RMSE = %.3f, MAE = %.3f, R^2 = %.3f\n', rmse_test, mae_test, r2_test);

% Baseline (predict mean)
baseline_pred = mean(Y_train) * ones(size(Y_test));
sse_base = sum((Y_test - baseline_pred).^2);
sst = sum((Y_test - mean(Y_test)).^2);
r2_base = 1 - sse_base / sst;
fprintf('Null-model R^2 (test): %.3f\n', r2_base);

feature_names = train_tbl.Properties.VariableNames(feature_cols);
% apply functions to for interpreting the results
interpret_elastic_net(feature_names, B, target_score)
bootstrap_test_metrics(Y_test, Y_pred_test, target_score);
plot_predictions(Y_test, Y_pred_test, r2_test, target_score);


