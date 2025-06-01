function results = run_nested_cv_SVM(X_train, Y_train, kernelType, outerK, innerK)
% RUN_NESTED_CV_SVM  Nested CV for SVM regression on pre-normalized data
%
%   results = RUN_NESTED_CV_SVM(X_train, Y_train, kernelType, outerK, innerK)
%
%   Performs k-fold outer and k-fold inner CV to tune SVM hyperparameters
%   by maximizing mean R^2 in the inner folds. Assumes X_train is already
%   normalized. Prints per-fold and overall R^2, RMSE, and MAE, and returns
%   a struct with all metrics and best parameters.

    if nargin < 3 || isempty(kernelType)
        kernelType = 'rbf';
    end

    % Hyperparameter grids
    C_values       = [0.01, 0.1, 1, 10, 50, 100, 1000];
    epsilon_values = [0.05, 0.1, 0.2, 0.5, 1];
    if strcmp(kernelType, 'rbf')
        sigma_values = logspace(-3, 3, 7);
    else
        sigma_values = [];
    end
    if strcmp(kernelType, 'polynomial')
        poly_orders = [2, 3, 4, 5];
    else
        poly_orders = [];
    end

    % CV setting. Number of folds were given as input

    outerCV = cvpartition(size(X_train,1), 'KFold', outerK);

    % Storage
    outerR2   = zeros(outerK,1);
    outerRMSE = zeros(outerK,1);
    outerMAE  = zeros(outerK,1);
    bestParamsList = cell(outerK,1);

    fprintf('\n===== Nested CV SVM (%s kernel) =====\n', kernelType);
    for i = 1:outerK
        fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
        trIdx = training(outerCV, i);
        teIdx = test(outerCV, i);
        % Split raw data for this outer fold
        Xtr_raw = X_train(trIdx, :);
        Ytr     = Y_train(trIdx);
        Xte_raw = X_train(teIdx, :);
        Yte     = Y_train(teIdx);
    
        % Standardize using only the outer‐training set
        [Xtr, mu, sigma] = zscore(Xtr_raw);
    
        % Apply the same mu/sigma to the held‐out fold
        Xte = (Xte_raw - mu) ./ sigma;


        % Inner CV to select hyperparameters
        innerCV = cvpartition(size(Xtr,1), 'KFold', innerK);
        bestInnerR2 = -Inf;
        bestParams = struct();

        for C = C_values
            for eps = epsilon_values
                switch kernelType
                    case 'rbf'
                        for sigma = sigma_values
                            [~, mr2] = svm_inner_cv(Xtr, Ytr, innerCV, 'rbf', C, sigma, [], eps);
                            if mr2 > bestInnerR2
                                bestInnerR2 = mr2;
                                bestParams = struct('C', C, 'sigma', sigma, 'epsilon', eps);
                            end
                        end
                    case 'linear'
                        [~, mr2] = svm_inner_cv(Xtr, Ytr, innerCV, 'linear', C, [], [], eps);
                        if mr2 > bestInnerR2
                            bestInnerR2 = mr2;
                            bestParams = struct('C', C, 'epsilon', eps);
                        end
                    case 'polynomial'
                        for order = poly_orders
                            [~, mr2] = svm_inner_cv(Xtr, Ytr, innerCV, 'polynomial', C, [], order, eps);
                            if mr2 > bestInnerR2
                                bestInnerR2 = mr2;
                                bestParams = struct('C', C, 'PolyOrder', order, 'epsilon', eps);
                            end
                        end
                end
            end
        end

        % Train on full outer train with best inner parameters
        switch kernelType
            case 'rbf'
                model = fitrsvm(Xtr, Ytr, 'KernelFunction','rbf', ...
                    'BoxConstraint',bestParams.C, 'KernelScale',bestParams.sigma, ...
                    'Epsilon',bestParams.epsilon, 'Standardize', false);
            case 'linear'
                model = fitrsvm(Xtr, Ytr, 'KernelFunction','linear', ...
                    'BoxConstraint',bestParams.C, 'Epsilon',bestParams.epsilon, ...
                    'Standardize', false);
            case 'polynomial'
                model = fitrsvm(Xtr, Ytr, 'KernelFunction','polynomial', ...
                    'PolynomialOrder',bestParams.PolyOrder, 'BoxConstraint',bestParams.C, ...
                    'Epsilon',bestParams.epsilon, 'Standardize', false);
        end

        % Evaluate on outer test
        Ypred = predict(model, Xte);
        r2   = 1 - sum((Yte - Ypred).^2) / sum((Yte - mean(Yte)).^2);
        rmse = sqrt(mean((Yte - Ypred).^2));
        mae  = mean(abs(Yte - Ypred));

        outerR2(i)   = r2;
        outerRMSE(i) = rmse;
        outerMAE(i)  = mae;
        bestParamsList{i} = bestParams;

        fprintf('Fold %d: R^2=%.4f, RMSE=%.4f, MAE=%.4f\n', i, r2, rmse, mae);
    end

    % Summary
    meanR2   = mean(outerR2);
    meanRMSE = mean(outerRMSE);
    meanMAE  = mean(outerMAE);
    fprintf('\n===== Summary =====\n');
    fprintf('Mean R^2: %.4f, Mean RMSE: %.4f, Mean MAE: %.4f\n', meanR2, meanRMSE, meanMAE);

    % Most frequent hyperparameters across outer folds
    Cs = cellfun(@(s) s.C, bestParamsList);
    epsilons = cellfun(@(s) s.epsilon, bestParamsList);
    bestMode = struct('C', mode(Cs), 'epsilon', mode(epsilons));
    if strcmp(kernelType,'rbf')
        sigmas = cellfun(@(s) s.sigma, bestParamsList);
        bestMode.sigma = mode(sigmas);
    elseif strcmp(kernelType,'polynomial')
        orders = cellfun(@(s) s.PolyOrder, bestParamsList);
        bestMode.PolyOrder = mode(orders);
    end
    fprintf('Best hyperparameters (mode across folds):\n');
    disp(bestMode);

    % Return results
    results = struct();
    results.outerR2 = outerR2;
    results.outerRMSE = outerRMSE;
    results.outerMAE = outerMAE;
    results.meanR2 = meanR2;
    results.meanRMSE = meanRMSE;
    results.meanMAE = meanMAE;
    results.bestParamsList = bestParamsList;
    results.bestParamsMode = bestMode;
end
