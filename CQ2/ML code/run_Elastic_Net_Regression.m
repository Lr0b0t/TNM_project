function [all_outer_r2, all_outer_rmse, all_outer_mae, bestParamsList, bestAlpha, bestLambda] = run_Elastic_Net_Regression(X_train, Y_train, outerK, innerK)
    % run_Elastic_Net_Regression performs nested cross-validation for Elastic Net regression.
    %
    %
    % Inputs:
    %   X_train - train matrix (n_samples x n_features)
    %   Y_train - response vector (n_samples x 1)
    %   outerK       - (optional) number of outer CV folds (default = 5)
    %   innerK       - (optional) number of inner CV folds (default = 3)
    %
    % Outputs:
    %   all_outer_r2       - R^2 scores for each outer fold (outerK x 1)
    %   all_outer_rmse     - RMSE for each outer fold (outerK x 1)
    %   all_outer_mae      - MAE for each outer fold (outerK x 1)
    %   bestParamsList     - cell array of best hyperparameter structs per fold
    %   bestAlpha          - modal Alpha across folds
    %   bestLambda         - modal Lambda across folds
    
    %  Define hyperparameter grid
    alphas  = [0.1, 0.3, 0.5, 0.7, 0.9, 1];
    lambdas = logspace(-4, 1, 8);
    
    % Nested Cross-validation setup
    if nargin < 4 || isempty(innerK)
        innerK = 3;
    end
    if nargin < 3 || isempty(outerK)
        outerK = 5;
    end
    outerCV = cvpartition(size(X_train,1), 'KFold', outerK);
    
    % Initialize storage
    all_outer_r2   = zeros(outerK,1);
    all_outer_rmse = zeros(outerK,1);
    all_outer_mae  = zeros(outerK,1);
    bestParamsList = cell(outerK,1);
    
    fprintf('\n===== NESTED CROSS-VALIDATION =====\n');
    
    for i = 1:outerK
        %fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
        % Split outer fold
        trainIdx = training(outerCV, i);
        valIdx   = test(outerCV, i);
    
        Xtr_outer = X_train(trainIdx, :);
        Ytr_outer = Y_train(trainIdx);
        Xval_outer = X_train(valIdx, :);
        Yval_outer = Y_train(valIdx);
    
        % Standardize: fit on train, apply to val
        [Xtr_outer_norm, mu_outer, sigma_outer] = zscore(Xtr_outer);
        Xval_outer_norm = (Xval_outer - mu_outer) ./ sigma_outer;
    
        % Inner CV for hyperparameter tuning
        innerCV = cvpartition(size(Xtr_outer_norm,1), 'KFold', innerK);
        best_inner_r2 = -Inf;
        best_inner_alpha = NaN;
        best_inner_lambda = NaN;
    
        % Grid search for both alpha and lambda
        for a = alphas
            for l = lambdas
                inner_r2s = zeros(innerK,1);
                for j = 1:innerK
                    inner_trainIdx = training(innerCV, j);
                    inner_valIdx   = test(innerCV, j);
    
                    Xtr_inner = Xtr_outer_norm(inner_trainIdx, :);
                    Ytr_inner = Ytr_outer(inner_trainIdx);
                    Xval_inner = Xtr_outer_norm(inner_valIdx, :);
                    Yval_inner = Ytr_outer(inner_valIdx);
    
                    % Fit Elastic Net, no standardize, already done
                    [B, FitInfo] = lassoglm(Xtr_inner, Ytr_inner, 'normal', ...
                        'Alpha', a, 'Lambda', l, 'Standardize', false);
    
                    coef     = [FitInfo.Intercept; B];
                    Xval_aug = [ones(size(Xval_inner,1),1), Xval_inner];
                    Ypred_val= Xval_aug * coef;
    
                    % Compute R^2
                    r2 = 1 - sum((Yval_inner - Ypred_val).^2) / sum((Yval_inner - mean(Yval_inner)).^2);
                    inner_r2s(j) = r2;
                end
                mean_r2 = mean(inner_r2s);
                if mean_r2 > best_inner_r2
                    best_inner_r2     = mean_r2;
                    best_inner_alpha  = a;
                    best_inner_lambda = l;
                end
            end
        end
    
        % fprintf('  Best inner params: Alpha=%.2f, Lambda=%.5f (Inner CV R^2=%.4f)\n', ...
         %   best_inner_alpha, best_inner_lambda, best_inner_r2);
    
        % Fit and evaluate on outer validation
        [B, FitInfo] = lassoglm(Xtr_outer_norm, Ytr_outer, 'normal', ...
            'Alpha', best_inner_alpha, 'Lambda', best_inner_lambda, 'Standardize', false);
        coef     = [FitInfo.Intercept; B];
        Xval_aug = [ones(size(Xval_outer_norm,1),1), Xval_outer_norm];
        Ypred_val= Xval_aug * coef;
    
        % Metrics
        r2   = 1 - sum((Yval_outer - Ypred_val).^2) / sum((Yval_outer - mean(Yval_outer)).^2);
        rmse = sqrt(mean((Yval_outer - Ypred_val).^2));
        mae  = mean(abs(Yval_outer - Ypred_val));
    
        all_outer_r2(i)   = r2;
        all_outer_rmse(i) = rmse;
        all_outer_mae(i)  = mae;
        bestParamsList{i} = struct('Alpha', best_inner_alpha, 'Lambda', best_inner_lambda);
    
        %fprintf('>> Fold %d: Outer R^2=%.4f, RMSE=%.4f, MAE=%.4f\n', i, r2, rmse, mae);
    end
    
    % Summary 
    %fprintf('\n===== NESTED CV PERFORMANCE =====\n');
    %fprintf('Mean Outer R^2: %.4f (std %.4f)\n', mean(all_outer_r2), std(all_outer_r2));
    %fprintf('Mean Outer RMSE: %.4f (std %.4f)\n', mean(all_outer_rmse), std(all_outer_rmse));
    %fprintf('Mean Outer MAE: %.4f (std %.4f)\n', mean(all_outer_mae), std(all_outer_mae));
    
    % Final best parameters 
    alphas_cv  = cellfun(@(s) s.Alpha, bestParamsList);
    lambdas_cv = cellfun(@(s) s.Lambda, bestParamsList);
    bestAlpha  = mode(alphas_cv);
    bestLambda = mode(lambdas_cv);
    
    fprintf('\nBest parameters (mode): Alpha=%.2f, Lambda=%.5f\n', bestAlpha, bestLambda);
end
