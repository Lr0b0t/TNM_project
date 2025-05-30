%--------------------------------------------------------------------------
% Script: random_forest_nested_cv_classification.m
%
% Purpose:
%   - Binary classification using Random Forests with nested cross-validation
%   - Reports accuracy for each outer fold and the average
%
% Inputs:
%   - train_features_Q2_imputed_and_categorized.csv (contains imputed features + binary class label in last column)
%   - test_features_Q2_impute_and_categorized_.csv  (used only for final report, not for CV)
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
class_col = width(train_tbl); % class in last column
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
Y_train_str = categorical(Y_train);  % for TreeBagger
Y_test_str = categorical(Y_test);

% ---- 4. Normalize features ----
[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

% ---- 5. Nested Cross Validation (Outer + Inner) ----
outerK = 5; innerK = 3;
outerCV = cvpartition(Y_train, 'KFold', outerK);

% Hyperparameter grid
ntrees_grid = [50, 100, 200];
minleaf_grid = [1, 5, 10];

outerAcc = zeros(outerK, 1);
outerAUC = zeros(outerK, 1);

fprintf('\n=== Nested Cross-Validation (Random Forest) ===\n');
for i = 1:outerK
    fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
    trainIdx = training(outerCV, i);
    valIdx = test(outerCV, i);
    
    X_outerTrain = X_train_norm(trainIdx, :);
    Y_outerTrain = Y_train_str(trainIdx);
    X_outerVal = X_train_norm(valIdx, :);
    Y_outerVal = Y_train(valIdx);

    % Inner CV for hyperparameter tuning
    best_acc = -Inf;
    best_ntrees = ntrees_grid(1);
    best_minleaf = minleaf_grid(1);

    for ntrees = ntrees_grid
        for minleaf = minleaf_grid
            innerCV = cvpartition(Y_outerTrain, 'KFold', innerK);
            acc_inner = zeros(innerK,1);

            for j = 1:innerK
                trIdx = training(innerCV, j);
                teIdx = test(innerCV, j);
                rf_inner = TreeBagger(ntrees, X_outerTrain(trIdx,:), Y_outerTrain(trIdx), ...
                    'Method', 'classification', 'MinLeafSize', minleaf, ...
                    'OOBPrediction', 'off');
                Yp = predict(rf_inner, X_outerTrain(teIdx,:));
                acc_inner(j) = mean(str2double(Yp) == double(Y_outerTrain(teIdx)));
            end

            mean_acc = mean(acc_inner);
            fprintf('  [Inner] ntrees=%d, minleaf=%d | Mean acc=%.4f\n', ntrees, minleaf, mean_acc);
            if mean_acc > best_acc
                best_acc = mean_acc;
                best_ntrees = ntrees;
                best_minleaf = minleaf;
            end
        end
    end

    fprintf('>> Best inner params: ntrees=%d, minleaf=%d | Mean acc=%.4f\n', best_ntrees, best_minleaf, best_acc);

    % Retrain on full outer training with best params, test on outer val
    rf_final = TreeBagger(best_ntrees, X_outerTrain, Y_outerTrain, ...
        'Method', 'classification', 'MinLeafSize', best_minleaf, ...
        'OOBPrediction', 'off');
    [Yp_val, Yp_scores] = rf_final.predict(X_outerVal);
    Yp_val = str2double(Yp_val);
    outerAcc(i) = mean(Yp_val == Y_outerVal);

    % AUC for this fold
    if size(Yp_scores,2)==2
        posclass = find(categories(Y_outerTrain) == "1");
        scores = Yp_scores(:, posclass);
    else
        scores = Yp_scores(:,1); % fallback
    end
    [~,~,~,auc] = perfcurve(Y_outerVal, scores, 1);
    outerAUC(i) = auc;
    fprintf('>> Fold %d Outer acc = %.4f, AUC = %.3f\n', i, outerAcc(i), auc);
end

fprintf('\n--- Nested CV summary (Random Forest) ---\n');
fprintf('Mean outer fold accuracy: %.4f (std=%.4f)\n', mean(outerAcc), std(outerAcc));
fprintf('Mean outer fold AUC: %.4f (std=%.4f)\n', mean(outerAUC), std(outerAUC));

% ---- 6. Retrain on ALL train data with most frequent best params, test on test set ----
% (You can use mode or just use largest ntrees/smallest minleaf for demonstration)
[~, idx] = max(outerAcc);
final_ntrees = ntrees_grid(1); final_minleaf = minleaf_grid(1); % default/fallback
if exist('best_ntrees','var'), final_ntrees = best_ntrees; end
if exist('best_minleaf','var'), final_minleaf = best_minleaf; end

fprintf('\n[Training final Random Forest on ALL training data: ntrees=%d, minleaf=%d]\n', final_ntrees, final_minleaf);
rf_final = TreeBagger(final_ntrees, X_train_norm, categorical(Y_train), ...
    'Method', 'classification', 'MinLeafSize', final_minleaf, ...
    'OOBPrediction', 'on', 'OOBVarImp', 'on');

[Y_pred_test, Y_pred_scores] = rf_final.predict(X_test_norm);
Y_pred_test = str2double(Y_pred_test);
test_acc = mean(Y_pred_test == Y_test);

% Test set AUC
if size(Y_pred_scores,2)==2
    posclass = find(categories(categorical(Y_train)) == "1");
    scores = Y_pred_scores(:, posclass);
else
    scores = Y_pred_scores(:,1);
end
[~,~,~,auc_test] = perfcurve(Y_test, scores, 1);

fprintf('\nTest set accuracy (final model): %.4f\n', test_acc);
fprintf('Test set AUC (final model): %.4f\n', auc_test);

% Plot confusion matrix
figure;
confusionchart(Y_test, Y_pred_test, ...
    'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
title('Confusion Matrix (Test Set, Final Random Forest)');

% Feature importance
imp = rf_final.OOBPermutedPredictorDeltaError;
[~,idx] = sort(imp, 'descend');
disp('Top 10 features by random forest importance (full model):');
for k = 1:min(10, numel(idx))
    fprintf('%2d. %-30s  Importance: %.4f\n', k, feature_names{idx(k)}, imp(idx(k)));
end
figure;
bar(imp(idx(1:10)));
set(gca, 'XTickLabel', feature_names(idx(1:10)), 'XTickLabelRotation',45);
ylabel('OOB Permuted Predictor Importance');
title('Top 10 Feature Importances (Random Forest)');

% Bootstrap accuracy and p-value (as before)
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
