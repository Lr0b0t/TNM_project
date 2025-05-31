function [all_outer_r2, mean_outer_r2, std_outer_r2, bestParamsList, bestParamsMode] = run_Random_Forest_Regression(X_train_norm, Y_train, outerK, innerK)
    % nestedRF_CV performs nested cross-validation for Random Forest regression.
    % 
    % Syntax:
    %   [all_outer_r2, mean_outer_r2, std_outer_r2, bestParamsList, bestParamsMode] = run_Random_Forest_Regression(X_train_norm, Y_train, outerK, innerK)
    % 
    % Inputs:
    %   X_train_norm - normalized predictor matrix (n_samples x n_features)
    %   Y_train      - response vector (n_samples x 1)
    %   outerK       - (optional) number of outer CV folds (default = 5)
    %   innerK       - (optional) number of inner CV folds (default = 3)
    % 
    % Outputs:
    %   all_outer_r2   - R^2 scores for each outer fold (outerK x 1)
    %   mean_outer_r2  - mean R^2 across outer folds
    %   std_outer_r2   - standard deviation of R^2 across outer folds
    %   bestParamsList - cell array of best hyperparameter structs per fold
    %   bestParamsMode - struct of most frequent hyperparameters across folds
    
    numTrees_grid = [40, 80, 150, 200];      % Number of trees in the forest
    minLeaf_grid = [1, 3, 5, 8, 12];         % Minimum leaf size
    maxNumSplits_grid = [10, 50, 100, 200];  % Maximum number of splits
    
     %  Handle optional arguments
    if nargin < 4 || isempty(innerK)
        innerK = 3;
    end
    if nargin < 3 || isempty(outerK)
        outerK = 5;
    end
    outerCV = cvpartition(size(X_train_norm,1), 'KFold', outerK);
    
    all_outer_r2 = zeros(outerK,1);
    bestParamsList = cell(outerK,1);
    
    fprintf('\n==== Starting Random Forest Nested Cross-Validation ====\n');
    for i = 1:outerK
        fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);
        trainIdx = training(outerCV, i);
        valIdx   = test(outerCV, i);
        Xtr_outer = X_train_norm(trainIdx, :);
        Ytr_outer = Y_train(trainIdx);
        Xval_outer = X_train_norm(valIdx, :);
        Yval_outer = Y_train(valIdx);
    
        % --- Nested: Inner CV for hyperparameter selection ---
        best_inner_r2 = -Inf;
        best_inner_param = struct();
    
        % Grid search over hyperparameters
        for ntree = numTrees_grid
            for minLeaf = minLeaf_grid
                for maxSplit = maxNumSplits_grid
                    % Prepare inner CV
                    innerCV = cvpartition(length(Ytr_outer), 'KFold', innerK);
                    inner_r2s = zeros(innerK,1);
    
                    for j = 1:innerK
                        inner_trainIdx = training(innerCV, j);
                        inner_valIdx   = test(innerCV, j);
    
                        Xtr_inner = Xtr_outer(inner_trainIdx,:);
                        Ytr_inner = Ytr_outer(inner_trainIdx);
                        Xval_inner = Xtr_outer(inner_valIdx,:);
                        Yval_inner = Ytr_outer(inner_valIdx);
    
                        % Train model on inner train
                        model = fitrensemble(Xtr_inner, Ytr_inner, ...
                            'Method', 'Bag', ...
                            'NumLearningCycles', ntree, ...
                            'Learners', templateTree(...
                                'MinLeafSize', minLeaf, ...
                                'MaxNumSplits', maxSplit));
    
                        % Predict on inner val
                        Ypred_inner = predict(model, Xval_inner);
    
                        % Compute R^2 for this inner fold
                        r2_inner = 1 - sum((Yval_inner - Ypred_inner).^2) / sum((Yval_inner - mean(Yval_inner)).^2);
                        inner_r2s(j) = r2_inner;
                    end
    
                    avg_inner_r2 = mean(inner_r2s);
    
                    % Save best hyperparams if this is the best avg R^2 so far
                    if avg_inner_r2 > best_inner_r2
                        best_inner_r2 = avg_inner_r2;
                        best_inner_param = struct('NumTrees', ntree, ...
                                                  'MinLeaf', minLeaf, ...
                                                  'MaxNumSplits', maxSplit);
                    end
                    %fprintf('  (Inner CV) Trees=%d, MinLeaf=%d, MaxSplit=%d | Mean R^2=%.4f\n', ...
                     %       ntree, minLeaf, maxSplit, avg_inner_r2);
                end
            end
        end
    
        % --- Train on full outer train set with best inner params ---
        model_final = fitrensemble(Xtr_outer, Ytr_outer, ...
            'Method', 'Bag', ...
            'NumLearningCycles', best_inner_param.NumTrees, ...
            'Learners', templateTree(...
                'MinLeafSize', best_inner_param.MinLeaf, ...
                'MaxNumSplits', best_inner_param.MaxNumSplits));
    
        % Evaluate on outer validation set
        Ypred_outer = predict(model_final, Xval_outer);
        outer_r2 = 1 - sum((Yval_outer - Ypred_outer).^2) / sum((Yval_outer - mean(Yval_outer)).^2);
    
        all_outer_r2(i) = outer_r2;
        bestParamsList{i} = best_inner_param;
    
        fprintf('>> Fold %d best params: Trees=%d, MinLeaf=%d, MaxSplit=%d | Outer R^2=%.4f\n', ...
            i, best_inner_param.NumTrees, best_inner_param.MinLeaf, best_inner_param.MaxNumSplits, outer_r2);
    end
    mean_outer_r2 = mean(all_outer_r2);
    std_outer_r2 = std(all_outer_r2);
    fprintf('\n==== Nested CV finished ====\n');
    fprintf('Outer fold R^2 scores: '); disp(all_outer_r2');
    fprintf('Mean Outer R^2: %.4f | Std: %.4f\n', mean_outer_r2, std(all_outer_r2));
    
    %% Find the mode (most frequent) best parameters
    numTrees_cv = cellfun(@(s) s.NumTrees, bestParamsList);
    minLeaf_cv = cellfun(@(s) s.MinLeaf, bestParamsList);
    maxSplit_cv = cellfun(@(s) s.MaxNumSplits, bestParamsList);
    
    bestNumTrees = mode(numTrees_cv);
    bestMinLeaf = mode(minLeaf_cv);
    bestMaxSplit = mode(maxSplit_cv);

    bestParamsMode = struct( ...
    'NumTrees',     mode(numTrees_cv), ...
    'MinLeaf',      mode(minLeaf_cv), ...
    'MaxNumSplits', mode(maxSplit_cv) ...
    );
    
    fprintf('\nBest parameters (mode): Trees=%d, MinLeaf=%d, MaxSplit=%d\n', bestNumTrees, bestMinLeaf, bestMaxSplit);


end