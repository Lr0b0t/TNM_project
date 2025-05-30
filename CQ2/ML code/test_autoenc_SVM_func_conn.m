close all; clear; clc;


% Paths
baseDir = fileparts(pwd);
dataDir = fullfile(baseDir, 'final files');
connDir = fullfile(dataDir, 'connectivity_n88');

% Load train/test splits (for IDs and MMSE follow-up)
trainFile = fullfile(dataDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(dataDir, 'test_features_Q2_imputed.csv');
trainData = readtable(trainFile);
testData  = readtable(testFile);

% Find column indices
id_col = 1; % usually first column
mmse_col = find(strcmp(trainData.Properties.VariableNames, 'MMSCORE_followUp')); % replace with exact col name if needed

train_ids = trainData{:, id_col};
Y_train = trainData{:, mmse_col};
test_ids = testData{:, id_col};
Y_test = testData{:, mmse_col};



% Helper function to zero-pad patient IDs to 7 digits
pad_id = @(id) sprintf('%07d', id);

% Load training matrices
train_matrices = cell(length(train_ids), 1);
for i = 1:length(train_ids)
    folderName = pad_id(train_ids(i));
    matFile = fullfile(connDir, folderName, 'func_connectivity.mat');
    if exist(matFile, 'file')
        data = load(matFile);
        if isfield(data, 'fc_mat')
            train_matrices{i} = data.fc_mat;
        else
            error('fc_mat variable not found in %s', matFile);
        end
       
    else
        error('File not found: %s', matFile);
    end
end


% Load test matrices
test_matrices = cell(length(test_ids), 1);
for i = 1:length(test_ids)
    folderName = pad_id(test_ids(i));
    matFile = fullfile(connDir, folderName, 'func_connectivity.mat');
    if exist(matFile, 'file')
        data = load(matFile);
        if isfield(data, 'fc_mat')
            test_matrices{i} = data.fc_mat;
        else
            error('fc_mat variable not found in %s', matFile);
        end
    else
        error('File not found: %s', matFile);
    end
end

%%  ------------------ plotting of all the functional connectivity matrices ----------------------------
% for k = 1:length(train_matrices)
%     figure; clf;
%     imagesc(train_matrices{k});
%     title(['Train subject ', num2str(train_ids(k)), ' (', num2str(k), '/', num2str(length(train_matrices)), ')']);
%     colorbar;
%     % caxis([-1 1]);
%     xlabel('Node');
%     ylabel('Node');
%     % Pause before showing next matrix
%     disp('Press any key to continue to the next matrix...');
%     pause;
% end
% 
% for k = 1:length(test_matrices)
%     figure; clf;
%     imagesc(test_matrices{k});
%     title(['Test subject ', num2str(test_ids(k)), ' (', num2str(k), '/', num2str(length(test_matrices)), ')']);
%     colorbar;
%     % caxis([-1 1]);
%     xlabel('Node');
%     ylabel('Node');
%     disp('Press any key to continue to the next matrix...');
%     pause;
% end


%%
% Assume all matrices are square
N = size(train_matrices{1}, 1);
upper_tri_mask = triu(true(N), 1); % upper triangle, no diagonal

% Vectorize train matrices
X_train = zeros(length(train_matrices), sum(upper_tri_mask(:)));
for i = 1:length(train_matrices)
    mat = train_matrices{i};
    X_train(i, :) = mat(upper_tri_mask);
end




% Vectorize test matrices
X_test = zeros(length(test_matrices), sum(upper_tri_mask(:)));
for i = 1:length(test_matrices)
    mat = test_matrices{i};
    X_test(i, :) = mat(upper_tri_mask);
end

fprintf('Train matrix shape: %d x %d\n', size(X_train, 1), size(X_train, 2));
fprintf('Test matrix shape: %d x %d\n', size(X_test, 1), size(X_test, 2));



%%
%% 1. Standardize features
[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;

%% 2. PCA on standardized training set
% [coeff, score_train, ~, ~, explained] = pca(X_train_norm);
% 
% cumExplained = cumsum(explained);
% numPC = find(cumExplained >= 65, 1, 'first');
% fprintf('Selected %d principal components for the inputted percentage of variance explained.\n', numPC);
% 
% X_train_pca = score_train(:, 1:numPC);
% X_test_pca = (X_test_norm * coeff(:, 1:numPC));



%% --- Autoencoder feature extraction ---

% hiddenSize = 20;  % Adjustable hyperparameter
% [X_train_ae, X_test_ae, autoenc] = autoencoder_features(X_train, X_test, hiddenSize);
% 

hiddenSizes = [10, 20, 30, 50];

for h = hiddenSizes
    [X_train_ae, X_test_ae] = autoencoder_features(X_train, X_test, h);
    %% --- Nested cross-validation setup ---
    
    outerK = 5;
    innerK = 3;
    outerCV = cvpartition(size(X_train_ae,1), 'KFold', outerK);
    
    results = struct();
    
    %% --- Hyperparameter grids ---
    
    % RBF kernel
    C_values_rbf = logspace(-2, 2, 5);
    sigma_values = logspace(-2, 2, 5);
    epsilon_values = [0.1, 0.5, 1];
    
    % Polynomial kernel
    C_values_poly = logspace(-2, 2, 5);
    polyorder_values = [2, 3, 4];
    epsilon_values_poly = [0.1, 0.5, 1];
    
    % Linear kernel
    C_values_lin = logspace(-2, 2, 5);
    epsilon_values_lin = [0.1, 0.5, 1];
    
    %% --- Nested CV for SVM Kernels ---
    
    % --- RBF SVM ---
    [meanR2_rbf, bestParams_rbf] = nested_cv_svm(X_train_ae, Y_train, outerCV, innerK, ...
        'rbf', struct('C', C_values_rbf, 'KernelScale', sigma_values, 'Epsilon', epsilon_values));
    
    results.rbf.meanR2 = meanR2_rbf;
    results.rbf.bestParams = bestParams_rbf;
    
    % --- Polynomial SVM ---
    [meanR2_poly, bestParams_poly] = nested_cv_svm(X_train_ae, Y_train, outerCV, innerK, ...
        'polynomial', struct('C', C_values_poly, 'PolyOrder', polyorder_values, 'Epsilon', epsilon_values_poly));
    
    results.poly.meanR2 = meanR2_poly;
    results.poly.bestParams = bestParams_poly;
    
    % --- Linear SVM ---
    [meanR2_lin, bestParams_lin] = nested_cv_svm(X_train_ae, Y_train, outerCV, innerK, ...
        'linear', struct('C', C_values_lin, 'Epsilon', epsilon_values_lin));
    
    results.linear.meanR2 = meanR2_lin;
    results.linear.bestParams = bestParams_lin;

    %% --- Select Best Model ---
    
    [~, bestIdx] = max([results.rbf.meanR2, results.poly.meanR2, results.linear.meanR2]);
    kernelTypes = {'rbf', 'polynomial', 'linear'};
    bestKernel = kernelTypes{bestIdx};
    fprintf('Best kernel type by mean R²: %s\n', bestKernel);
    
    %% --- Final Training & Evaluation ---
    
    switch bestKernel
        case 'rbf'
            params = results.rbf.bestParams;
            model = fitrsvm(X_train_ae, Y_train, 'KernelFunction', 'rbf', ...
                'BoxConstraint', params.C, 'KernelScale', params.KernelScale, 'Epsilon', params.Epsilon);
        case 'polynomial'
            params = results.poly.bestParams;
            model = fitrsvm(X_train_ae, Y_train, 'KernelFunction', 'polynomial', ...
                'PolynomialOrder', params.PolyOrder, 'BoxConstraint', params.C, 'Epsilon', params.Epsilon);
        case 'linear'
            params = results.linear.bestParams;
            model = fitrsvm(X_train_ae, Y_train, 'KernelFunction', 'linear', ...
                'BoxConstraint', params.C, 'Epsilon', params.Epsilon);
    end
    
    % Test predictions
    Y_pred_test = predict(model, X_test_ae);
    rmse_test = sqrt(mean((Y_test - Y_pred_test).^2));
    mae_test = mean(abs(Y_test - Y_pred_test));
    r2_test = 1 - sum((Y_test - Y_pred_test).^2) / sum((Y_test - mean(Y_test)).^2);
    
    fprintf('\nFinal Test Results:\n');
    fprintf('RMSE = %.3f, MAE = %.3f, R^2 = %.3f\n', rmse_test, mae_test, r2_test);
    
    % Baseline (mean prediction)
    baseline_pred = mean(Y_train) * ones(size(Y_test));
    r2_base = 1 - sum((Y_test - baseline_pred).^2) / sum((Y_test - mean(Y_test)).^2);
    fprintf('Baseline R^2 (mean prediction): %.3f\n', r2_base);

end



%% --- Function Definitions ---

function [X_train_ae, X_test_ae, autoenc] = autoencoder_features(X_train, X_test, hiddenSize)
    [X_train_norm, mu, sigma] = zscore(X_train);
    X_test_norm = (X_test - mu) ./ sigma;
    
    autoenc = trainAutoencoder(X_train_norm', hiddenSize, ...
        'MaxEpochs', 200, ...
        'L2WeightRegularization', 0.001, ...
        'SparsityRegularization', 1, ...
        'SparsityProportion', 0.1, ...
        'ScaleData', false);
    
    X_train_ae = encode(autoenc, X_train_norm')';
    X_test_ae  = encode(autoenc, X_test_norm')';
end

%% 5. Nested CV SVM function
function [mean_outer_r2, bestParams, all_outer_r2] = nested_cv_svm(X, Y, outerCV, innerK, kernelType, grid)
    outerK = outerCV.NumTestSets;
    all_outer_r2 = zeros(outerK,1);
    bestParamList = cell(outerK,1);

    for i = 1:outerK
        fprintf('\n--- [%s] Outer Fold %d/%d ---\n', kernelType, i, outerK);
        trainIdx = training(outerCV, i);
        valIdx   = test(outerCV, i);
        Xtr_outer = X(trainIdx, :);
        Ytr_outer = Y(trainIdx);
        Xval_outer = X(valIdx, :);
        Yval_outer = Y(valIdx);

        innerCV = cvpartition(size(Xtr_outer,1), 'KFold', innerK);

        best_r2 = -Inf;
        switch kernelType
            case 'rbf'
                for C = grid.C
                    for sigma = grid.KernelScale
                        for eps = grid.Epsilon
                            inner_r2s = zeros(innerK,1);
                            for j = 1:innerK
                                trIdx = training(innerCV, j);
                                valIdx = test(innerCV, j);
                                mdl = fitrsvm(Xtr_outer(trIdx,:), Ytr_outer(trIdx), ...
                                    'KernelFunction', 'rbf', ...
                                    'BoxConstraint', C, ...
                                    'KernelScale', sigma, ...
                                    'Epsilon', eps, ...
                                    'Standardize', false);
                                Ypred = predict(mdl, Xtr_outer(valIdx,:));
                                sse = sum((Ytr_outer(valIdx) - Ypred).^2);
                                sst = sum((Ytr_outer(valIdx) - mean(Ytr_outer(valIdx))).^2);
                                r2 = 1 - sse/sst;
                                inner_r2s(j) = r2;
                            end
                            mean_r2 = mean(inner_r2s);
                            % fprintf('  C=%.3f, sigma=%.3f, eps=%.3f | Inner mean R^2=%.4f\n', C, sigma, eps, mean_r2);
                            if mean_r2 > best_r2
                                best_r2 = mean_r2;
                                best_param = struct('C', C, 'KernelScale', sigma, 'Epsilon', eps);
                            end
                        end
                    end
                end
            case 'polynomial'
                for C = grid.C
                    for po = grid.PolyOrder
                        for eps = grid.Epsilon
                            inner_r2s = zeros(innerK,1);
                            for j = 1:innerK
                                trIdx = training(innerCV, j);
                                valIdx = test(innerCV, j);
                                mdl = fitrsvm(Xtr_outer(trIdx,:), Ytr_outer(trIdx), ...
                                    'KernelFunction', 'polynomial', ...
                                    'PolynomialOrder', po, ...
                                    'BoxConstraint', C, ...
                                    'Epsilon', eps, ...
                                    'Standardize', false);
                                Ypred = predict(mdl, Xtr_outer(valIdx,:));
                                sse = sum((Ytr_outer(valIdx) - Ypred).^2);
                                sst = sum((Ytr_outer(valIdx) - mean(Ytr_outer(valIdx))).^2);
                                r2 = 1 - sse/sst;
                                inner_r2s(j) = r2;
                            end
                            mean_r2 = mean(inner_r2s);
                            % fprintf('  C=%.3f, order=%d, eps=%.3f | Inner mean R^2=%.4f\n', C, po, eps, mean_r2);
                            if mean_r2 > best_r2
                                best_r2 = mean_r2;
                                best_param = struct('C', C, 'PolyOrder', po, 'Epsilon', eps);
                            end
                        end
                    end
                end
            case 'linear'
                for C = grid.C
                    for eps = grid.Epsilon
                        inner_r2s = zeros(innerK,1);
                        for j = 1:innerK
                            trIdx = training(innerCV, j);
                            valIdx = test(innerCV, j);
                            mdl = fitrsvm(Xtr_outer(trIdx,:), Ytr_outer(trIdx), ...
                                'KernelFunction', 'linear', ...
                                'BoxConstraint', C, ...
                                'Epsilon', eps, ...
                                'Standardize', false);
                            Ypred = predict(mdl, Xtr_outer(valIdx,:));
                            sse = sum((Ytr_outer(valIdx) - Ypred).^2);
                            sst = sum((Ytr_outer(valIdx) - mean(Ytr_outer(valIdx))).^2);
                            r2 = 1 - sse/sst;
                            inner_r2s(j) = r2;
                        end
                        mean_r2 = mean(inner_r2s);
                        % fprintf('  C=%.3f, eps=%.3f | Inner mean R^2=%.4f\n', C, eps, mean_r2);
                        if mean_r2 > best_r2
                            best_r2 = mean_r2;
                            best_param = struct('C', C, 'Epsilon', eps);
                        end
                    end
                end
        end
        switch kernelType
            case 'rbf'
                mdl_final = fitrsvm(Xtr_outer, Ytr_outer, ...
                    'KernelFunction', 'rbf', ...
                    'BoxConstraint', best_param.C, ...
                    'KernelScale', best_param.KernelScale, ...
                    'Epsilon', best_param.Epsilon, ...
                    'Standardize', false);
            case 'polynomial'
                mdl_final = fitrsvm(Xtr_outer, Ytr_outer, ...
                    'KernelFunction', 'polynomial', ...
                    'PolynomialOrder', best_param.PolyOrder, ...
                    'BoxConstraint', best_param.C, ...
                    'Epsilon', best_param.Epsilon, ...
                    'Standardize', false);
            case 'linear'
                mdl_final = fitrsvm(Xtr_outer, Ytr_outer, ...
                    'KernelFunction', 'linear', ...
                    'BoxConstraint', best_param.C, ...
                    'Epsilon', best_param.Epsilon, ...
                    'Standardize', false);
        end
        Ypred_outer = predict(mdl_final, Xval_outer);
        sse_outer = sum((Yval_outer - Ypred_outer).^2);
        sst_outer = sum((Yval_outer - mean(Yval_outer)).^2);
        outer_r2 = 1 - sse_outer/sst_outer;
        all_outer_r2(i) = outer_r2;
        bestParamList{i} = best_param;

        fprintf('>> [%s] Fold %d best: Outer R^2=%.4f\n', kernelType, i, outer_r2);
    end
    mean_outer_r2 = mean(all_outer_r2);
    bestParams = bestParamList{end}; % For simplicity, use params from last fold
end
