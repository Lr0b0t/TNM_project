%--------------------------------------------------------------------------
% Script: svm_rbf_nested_cv_classification.m
%
% Purpose:
%   - Binary classification with SVM (RBF kernel), nested cross-validation for hyperparameter selection.
%   - Reports accuracy and AUC for each outer fold and test set.
%   - Includes bootstrap confidence interval and p-value for test accuracy.
%
% Inputs:
%   - train_features_Q2_imputed_and_categorized.csv (features + binary class label in last column)
%   - test_features_Q2_impute_and_categorized_.csv
%--------------------------------------------------------------------------

clc; clear; close all;

% ---- 1. Load Data ----
baseDir = fullfile('..', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q2_imputed_and_categorized.csv');
testFile  = fullfile(baseDir, 'test_features_Q2_impute_and_categorized_.csv');

train_tbl = readtable(trainFile);
test_tbl  = readtable(testFile);

% ---- 2. Identify features and class columns ----
id_col = 1;
class_col = width(train_tbl); % last column
target_names = {'MMSCORE_followUp', 'CDSOB_followUp', 'GDTOTAL_followUp'};
target_cols = find(ismember(train_tbl.Properties.VariableNames, target_names));
feature_cols = setdiff(1:width(train_tbl), [id_col, target_cols, class_col]);
feature_names = train_tbl.Properties.VariableNames(feature_cols);

X_train = train_tbl{:, feature_cols};
Y_train = train_tbl{:, class_col};
X_test = test_tbl{:, feature_cols};
Y_test = test_tbl{:, class_col};

% ---- 3. Prepare labels ----
if iscell(Y_train), Y_train = cellfun(@str2double, Y_train); end
if iscell(Y_test), Y_test = cellfun(@str2double, Y_test); end
Y_train = double(Y_train(:));
Y_test = double(Y_test(:));
Y_train_categorical = categorical(Y_train);

% ---- 4. Normalize features ----
[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

% ---- 5. Nested Cross-Validation (Outer + Inner) for RBF SVM ----
outerK = 5; innerK = 3;
outerCV = cvpartition(Y_train, 'KFold', outerK);

% Hyperparameter grid
C_values = logspace(-2,2,5);    % [0.01, 0.1, 1, 10, 100]
gamma_values = logspace(-2,2,5); % SVM "KernelScale" in MATLAB = 1/sqrt(2*gamma)
bestParamsList = cell(outerK, 1);
outerAcc = zeros(outerK,1);
outerAUC = zeros(outerK,1);

fprintf('\n=== Nested Cross-Validation (SVM RBF) ===\n');
for i = 1:outerK
    fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
    trainIdx = training(outerCV, i);
    valIdx = test(outerCV, i);

    X_outerTrain = X_train_norm(trainIdx, :);
    Y_outerTrain = Y_train(trainIdx);
    X_outerVal = X_train_norm(valIdx, :);
    Y_outerVal = Y_train(valIdx);

    best_acc = -Inf;
    for C = C_values
        for gamma = gamma_values
            innerCV = cvpartition(Y_outerTrain, 'KFold', innerK);
            acc_inner = zeros(innerK,1);

            for j = 1:innerK
                trIdx = training(innerCV, j);
                teIdx = test(innerCV, j);

                mdl = fitcsvm(X_outerTrain(trIdx,:), Y_outerTrain(trIdx), ...
                    'KernelFunction', 'rbf', ...
                    'BoxConstraint', C, ...
                    'KernelScale', 1/sqrt(2*gamma), ...
                    'Standardize', false, ...
                    'ClassNames', [0 1]);
                Yp = predict(mdl, X_outerTrain(teIdx,:));
                acc_inner(j) = mean(Yp == Y_outerTrain(teIdx));
            end

            mean_acc = mean(acc_inner);
            fprintf('  [Inner] C=%.3f, gamma=%.3f | Mean acc=%.4f\n', C, gamma, mean_acc);
            if mean_acc > best_acc
                best_acc = mean_acc;
                best_params = struct('C', C, 'gamma', gamma);
            end
        end
    end

    bestParamsList{i} = best_params;
    fprintf('>> Best inner params: C=%.3f, gamma=%.3f | Mean acc=%.4f\n', best_params.C, best_params.gamma, best_acc);

    % Retrain on outer train set with best params
    mdl_final = fitcsvm(X_outerTrain, Y_outerTrain, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', best_params.C, ...
        'KernelScale', 1/sqrt(2*best_params.gamma), ...
        'Standardize', false, ...
        'ClassNames', [0 1]);

    % Try probability calibration for AUC (if supported)
    scores_val = [];
    try
        mdl_final = fitPosterior(mdl_final, X_outerTrain, Y_outerTrain);
        [Yp_val, scores_val] = predict(mdl_final, X_outerVal);
    catch
        [Yp_val, scores_val] = predict(mdl_final, X_outerVal);
    end

    outerAcc(i) = mean(Yp_val == Y_outerVal);

    % Outer fold AUC (if probabilities available)
    try
        [~,~,~,auc] = perfcurve(Y_outerVal, scores_val(:,2), 1);
        outerAUC(i) = auc;
    catch
        outerAUC(i) = NaN;
    end
    fprintf('>> Fold %d Outer acc = %.4f, AUC = %.3f\n', i, outerAcc(i), outerAUC(i));
end

fprintf('\n--- Nested CV summary (SVM RBF) ---\n');
fprintf('Mean outer fold accuracy: %.4f (std=%.4f)\n', mean(outerAcc), std(outerAcc));
fprintf('Mean outer fold AUC: %.4f (std=%.4f)\n', nanmean(outerAUC), nanstd(outerAUC));

% ---- 6. Retrain on ALL train data with mode of best params, test on test set ----
Cs = cellfun(@(s) s.C, bestParamsList);
gammas = cellfun(@(s) s.gamma, bestParamsList);
bestC = mode(Cs);
bestGamma = mode(gammas);

fprintf('\n[Training final SVM (RBF) on ALL training data: C=%.3f, gamma=%.3f]\n', bestC, bestGamma);
final_mdl = fitcsvm(X_train_norm, Y_train, ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint', bestC, ...
    'KernelScale', 1/sqrt(2*bestGamma), ...
    'Standardize', false, ...
    'ClassNames', [0 1]);

% Try probability calibration for AUC (if supported)
Y_pred_test = [];
Y_pred_scores = [];
try
    final_mdl = fitPosterior(final_mdl, X_train_norm, Y_train);
    [Y_pred_test, Y_pred_scores] = predict(final_mdl, X_test_norm);
catch
    [Y_pred_test, Y_pred_scores] = predict(final_mdl, X_test_norm);
end
test_acc = mean(Y_pred_test == Y_test);

% Test set AUC (if supported)
try
    [~,~,~,auc_test] = perfcurve(Y_test, Y_pred_scores(:,2), 1);
catch
    auc_test = NaN;
end

fprintf('\nTest set accuracy (final model): %.4f\n', test_acc);
fprintf('Test set AUC (final model): %.4f\n', auc_test);

% Plot confusion matrix
figure;
confusionchart(Y_test, Y_pred_test, ...
    'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
title('Confusion Matrix (Test Set, Final SVM RBF)');

% Bootstrapped accuracy and p-value
nBoot = 1000;
acc_boot = zeros(nBoot,1);
N = numel(Y_test);
for i = 1:nBoot
    idx_boot = randi(N, N, 1);
    acc_boot(i) = mean(Y_pred_test(idx_boot) == Y_test(idx_boot));
end
acc_ci = prctile(acc_boot, [2.5 97.5]);
fprintf('\nBootstrapped 95%% Confidence Interval for Test Set Accuracy: [%.3f, %.3f]\n', acc_ci(1), acc_ci(2));
fprintf(['This interval means that if you were to repeat this experiment many times, ' ...
         'the test set accuracy would fall in this range 95%% of the time.\n']);

chance = 0.5;
pval = mean(acc_boot <= chance);
fprintf('\nBootstrap p-value for accuracy > chance level (%.2f): %.4f\n', chance, pval);
fprintf(['This is the estimated probability that a classifier no better than random guessing\n' ...
         'would perform as well or better than your model. p < 0.05 is considered significant.\n']);

figure;
histogram(acc_boot, 30);
xlabel('Accuracy');
ylabel('Bootstrap Sample Count');
title('Bootstrap Distribution of Test Set Accuracy');

% Optional: ROC curve (if AUC supported)
try
    figure;
    [~,~,~,auc_test] = perfcurve(Y_test, Y_pred_scores(:,2), 1);
    plotroc(categorical(Y_test)', Y_pred_scores(:,2)');
    title(sprintf('ROC Curve (AUC = %.3f)', auc_test));
catch
    disp('ROC curve not available (no probabilistic output)');
end
