%--------------------------------------------------------------------------
% Script: elasticnet_classification_with_bootstrap.m
%
% Purpose:
%   - Binary classification using elastic net logistic regression (grid search).
%   - Reports test set accuracy, confusion matrix, AUC, bootstrapped 95% CI for accuracy,
%     and bootstrap p-value for accuracy vs. chance.
%
% Inputs:
%   - train_features_Q2_imputed_and_categorized.csv
%   - test_features_Q2_impute_and_categorized_.csv
%--------------------------------------------------------------------------

clc; clear; close all;

% ---- 1. Load Data ----
baseDir = fullfile('..','..', 'data cleanup and management', 'final files');

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
X_test  = test_tbl{:, feature_cols};
Y_train = train_tbl{:, class_col};
Y_test  = test_tbl{:, class_col};

% ---- 3. Prepare Labels (must be double 0/1) ----
if iscell(Y_train), Y_train = cellfun(@str2double, Y_train); end
if iscell(Y_test), Y_test = cellfun(@str2double, Y_test); end
if iscategorical(Y_train), Y_train = double(Y_train) - 1; end
if iscategorical(Y_test), Y_test = double(Y_test) - 1; end
Y_train = double(Y_train(:));
Y_test = double(Y_test(:));

% ---- 4. Normalize Features ----
[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

% ---- 5. Elastic Net Logistic Regression (Grid over Alpha/Lambda) ----
alphas = [0.1, 0.25, 0.5, 0.75, 1];
lambdas = logspace(-4, 1, 20);

bestDeviance = Inf;
bestAlpha = NaN; bestLambda = NaN; bestB = []; bestIntercept = NaN;

fprintf('\nElastic net grid search (logistic regression):\n');
for i = 1:length(alphas)
    a = alphas(i);
    [B, FitInfo] = lassoglm(X_train_norm, Y_train, 'binomial', ...
        'Alpha', a, ...
        'Lambda', lambdas, ...
        'CV', 5, ...
        'MaxIter', 1e4, ...
        'Standardize', false);

    [minDev, idx] = min(FitInfo.Deviance);
    fprintf('  Alpha = %.2f | best Lambda = %.5f | min CV deviance = %.5f\n', ...
            a, FitInfo.Lambda(idx), minDev);

    if minDev < bestDeviance
        bestDeviance = minDev;
        bestAlpha = a;
        bestLambda = FitInfo.Lambda(idx);
        bestB = B(:, idx);
        bestIntercept = FitInfo.Intercept(idx);
    end
end

fprintf('\nBest Elastic Net parameters found:\n');
fprintf('  Alpha (L1/L2 mixing): %.2f\n', bestAlpha);
fprintf('  Lambda (penalty): %.5f\n', bestLambda);

% ---- 6. Predict on Test Set ----
scores = X_test_norm * bestB + bestIntercept;
probs = 1 ./ (1 + exp(-scores));
Y_pred = double(probs > 0.5); % 0/1

accuracy = mean(Y_pred == Y_test);
fprintf('\nTest set accuracy: %.2f%% (raw accuracy: %.4f)\n', 100*accuracy, accuracy);

% ---- 7. Confusion Matrix (plot) ----
% figure;
% confusionchart(Y_test, Y_pred, ...
%     'RowSummary','row-normalized', ...
%     'ColumnSummary','column-normalized');
% title('Confusion Matrix (Test Set)');

% Print confusion matrix counts in console
cmat = confusionmat(Y_test, Y_pred);
fprintf('\nConfusion matrix (rows = true class, cols = predicted class):\n');
disp(cmat);

% ---- 8. ROC Curve and AUC ----
[Xroc, Yroc, T, AUC] = perfcurve(Y_test, probs, 1);
fprintf('\nTest set AUC (Area Under ROC Curve): %.3f\n', AUC);

figure;
plot(Xroc, Yroc, 'b-', 'LineWidth', 2); hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1); % Diagonal = chance
grid on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (Test Set) — AUC = %.3f', AUC));
legend('Model', 'Random Guess', 'Location', 'southeast');
hold off;

% ---- 9. Top Features by Absolute Coefficient ----
[~, idx] = sort(abs(bestB), 'descend');
disp('Top 10 features by absolute elastic net coefficient:');
for k = 1:min(10, numel(idx))
    fprintf('%2d. %-30s  Coefficient: %.4f\n', k, feature_names{idx(k)}, bestB(idx(k)));
end

% ---- 10. Bootstrap Accuracy, Confidence Interval, and p-value ----
nBoot = 1000;
acc_boot = zeros(nBoot,1);
N = numel(Y_test);

for i = 1:nBoot
    idx = randi(N, N, 1);  % Resample with replacement
    acc_boot(i) = mean(Y_pred(idx) == Y_test(idx));
end

acc_ci = prctile(acc_boot, [2.5 97.5]);
fprintf('\nBootstrapped 95%% Confidence Interval for Accuracy: [%.3f, %.3f]\n', acc_ci(1), acc_ci(2));
fprintf(['This interval means that if you were to repeat this experiment many times, ' ...
         'the test set accuracy would fall in this range 95%% of the time.\n']);

% p-value (test if accuracy > chance, e.g. 0.5)
chance = 0.5;
pval = mean(acc_boot <= chance);
fprintf('\nBootstrap p-value for accuracy > chance level (%.2f): %.4f\n', chance, pval);
fprintf(['This is the estimated probability that a classifier no better than random guessing\n' ...
         'would perform as well or better than your model. p < 0.05 is considered significant.\n']);

% Histogram of bootstrapped accuracies
figure;
histogram(acc_boot, 30);
xlabel('Accuracy');
ylabel('Bootstrap Sample Count');
title('Bootstrap Distribution of Test Set Accuracy');
