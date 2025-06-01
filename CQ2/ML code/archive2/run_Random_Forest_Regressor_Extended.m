function [all_outer_r2, mean_outer_r2, std_outer_r2, bestParamsList, bestParamsMode] = ...
         run_Random_Forest_Regressor_Extended(X_train_norm, Y_train, outerK, innerK)
% run_Random_Forest_Regressor_Extended
%   Performs nested cross-validation for a Random Forest regressor over
%   an expanded grid of hyperparameters. Assumes X_train_norm is already
%   normalized prior to calling this function.
%
%   [all_outer_r2, mean_outer_r2, std_outer_r2, bestParamsList, bestParamsMode] =
%       run_Random_Forest_Regressor_Extended(X_train_norm, Y_train)
%   uses default outerK=5 and innerK=3.
%
%   [all_outer_r2, mean_outer_r2, std_outer_r2, bestParamsList, bestParamsMode] =
%       run_Random_Forest_Regressor_Extended(X_train_norm, Y_train, outerK, innerK)
%   uses the specified outerK (number of outer folds) and innerK (number of inner folds).
%
%   Inputs:
%     X_train_norm - normalized predictor matrix (n_samples x n_features)
%     Y_train      - response vector (n_samples x 1)
%     outerK       - (optional) number of outer CV folds (default = 5)
%     innerK       - (optional) number of inner CV folds (default = 3)
%
%   Outputs:
%     all_outer_r2    - R² scores for each of the outer folds (outerK x 1)
%     mean_outer_r2   - mean R² across all outer folds
%     std_outer_r2    - standard deviation of R² across outer folds
%     bestParamsList  - cell array (length = outerK) of structs containing
%                       the best hyperparameters found on each outer fold
%     bestParamsMode  - struct of the mode (most frequent) hyperparameters
%                       across all outer folds

    % 1) Handle optional arguments
    if nargin < 4 || isempty(innerK)
        innerK = 3;
    end
    if nargin < 3 || isempty(outerK)
        outerK = 5;
    end

    % 2) Define extended hyperparameter grids
    numTrees_grid        = [50, 100, 200, 400];  % # of trees
    minLeaf_grid         = [1, 3, 5, 10];        % minimum leaf size
    maxNumSplits_grid    = [10, 50, 100, 200];   % maximum number of splits
    numVarsToSample_grid = [ ...
        floor(sqrt(size(X_train_norm,2))), ...
        floor(size(X_train_norm,2)/3), ...
        floor(size(X_train_norm,2)/2), ...
        size(X_train_norm,2) ...
    ];  % number of features to sample per split
    useSurrogate_grid    = [false, true];        % surrogate splits on/off

    % 3) Set up nested CV
    outerCV = cvpartition(size(X_train_norm,1), 'KFold', outerK);

    % 4) Pre-allocate storage
    all_outer_r2   = zeros(outerK,1);
    bestParamsList = cell(outerK,1);

    fprintf('\n==== Starting Extended Random Forest Nested CV ====\n');
    for i = 1:outerK
        fprintf('\n--- Outer Fold %d/%d ---\n', i, outerK);

        % 4.1) Split into outer-train / outer-val
        trainIdx  = training(outerCV, i);
        valIdx    = test(outerCV, i);
        Xtr_outer = X_train_norm(trainIdx, :);
        Ytr_outer = Y_train(trainIdx);
        Xval_outer = X_train_norm(valIdx, :);
        Yval_outer = Y_train(valIdx);

        % 5) Inner CV: hyperparameter tuning
        best_inner_r2    = -Inf;
        best_inner_param = struct( ...
            'NumTrees',           [], ...
            'MinLeafSize',        [], ...
            'MaxNumSplits',       [], ...
            'NumVarsToSample',    [], ...
            'UseSurrogateSplits', [] ...
        );

        % 5.1) Loop over the extended grid
        for ntree = numTrees_grid
            for minLeaf = minLeaf_grid
                for maxSplit = maxNumSplits_grid
                    for nvarSample = numVarsToSample_grid
                        for useSur = useSurrogate_grid
                            % Create a fresh inner partition on this outer-train set
                            innerCV = cvpartition(size(Xtr_outer,1), 'KFold', innerK);
                            inner_r2s = zeros(innerK,1);

                            % Evaluate current combination via inner folds
                            for j = 1:innerK
                                inner_trainIdx = training(innerCV, j);
                                inner_valIdx   = test(innerCV, j);

                                Xtr_inner = Xtr_outer(inner_trainIdx,:);
                                Ytr_inner = Ytr_outer(inner_trainIdx);
                                Xval_inner = Xtr_outer(inner_valIdx,:);
                                Yval_inner = Ytr_outer(inner_valIdx);

                                % Build template tree with this set of hyperparameters
                                t = templateTree( ...
                                    'MinLeafSize',         minLeaf, ...
                                    'MaxNumSplits',        maxSplit, ...
                                    'NumVariablesToSample', nvarSample, ...
                                    'Surrogate',           useSur ...
                                );

                                % Train RF on inner-training
                                model_inner = fitrensemble( ...
                                    Xtr_inner, Ytr_inner, ...
                                    'Method', 'Bag', ...
                                    'NumLearningCycles', ntree, ...
                                    'Learners',          t ...
                                );

                                % Predict on inner-validation
                                Ypred_inner = predict(model_inner, Xval_inner);

                                % Compute R² on the inner-validation fold
                                r2_inner = 1 - sum((Yval_inner - Ypred_inner).^2) / ...
                                           sum((Yval_inner - mean(Yval_inner)).^2);
                                inner_r2s(j) = r2_inner;
                            end

                            avg_inner_r2 = mean(inner_r2s);

                            % Update best hyperparameters if this is superior
                            if avg_inner_r2 > best_inner_r2
                                best_inner_r2 = avg_inner_r2;
                                best_inner_param.NumTrees           = ntree;
                                best_inner_param.MinLeafSize        = minLeaf;
                                best_inner_param.MaxNumSplits       = maxSplit;
                                best_inner_param.NumVarsToSample    = nvarSample;
                                best_inner_param.UseSurrogateSplits = useSur;
                            end
                        end
                    end
                end
            end
        end

        % 6) Retrain on full outer-train with chosen hyperparameters
        t_final = templateTree( ...
            'MinLeafSize',         best_inner_param.MinLeafSize, ...
            'MaxNumSplits',        best_inner_param.MaxNumSplits, ...
            'NumVariablesToSample', best_inner_param.NumVarsToSample, ...
            'Surrogate',           best_inner_param.UseSurrogateSplits ...
        );
        model_final = fitrensemble( ...
            Xtr_outer, Ytr_outer, ...
            'Method',            'Bag', ...
            'NumLearningCycles', best_inner_param.NumTrees, ...
            'Learners',          t_final ...
        );

        % 7) Evaluate on outer-validation set
        Ypred_outer = predict(model_final, Xval_outer);
        outer_r2 = 1 - sum((Yval_outer - Ypred_outer).^2) / ...
                   sum((Yval_outer - mean(Yval_outer)).^2);

        all_outer_r2(i)   = outer_r2;
        bestParamsList{i} = best_inner_param;

        % Print per-fold results
        fprintf('Fold %d best params:\n', i);
        fprintf('  NumTrees          = %d\n', best_inner_param.NumTrees);
        fprintf('  MinLeafSize       = %d\n', best_inner_param.MinLeafSize);
        fprintf('  MaxNumSplits      = %d\n', best_inner_param.MaxNumSplits);
        fprintf('  NumVarsToSample   = %d\n', best_inner_param.NumVarsToSample);
        fprintf('  UseSurrogateSplits = %d\n', best_inner_param.UseSurrogateSplits);
        fprintf('  >> Outer R^2       = %.4f\n', outer_r2);
    end

    % 8) Summarize Nested-CV results
    mean_outer_r2 = mean(all_outer_r2);
    std_outer_r2  = std(all_outer_r2);

    fprintf('\n==== Nested CV finished ====\n');
    fprintf('Outer fold R^2 scores: '); disp(all_outer_r2');
    fprintf('Mean Outer R^2: %.4f | Std: %.4f\n', mean_outer_r2, std_outer_r2);

    % Determine the mode of each hyperparameter across folds
    numTrees_cv     = cellfun(@(s) s.NumTrees,           bestParamsList);
    minLeaf_cv      = cellfun(@(s) s.MinLeafSize,        bestParamsList);
    maxSplit_cv     = cellfun(@(s) s.MaxNumSplits,       bestParamsList);
    nvarSample_cv   = cellfun(@(s) s.NumVarsToSample,    bestParamsList);
    useSurrogate_cv = cellfun(@(s) s.UseSurrogateSplits, bestParamsList);

    bestParamsMode = struct( ...
        'NumTrees',           mode(numTrees_cv), ...
        'MinLeafSize',        mode(minLeaf_cv), ...
        'MaxNumSplits',       mode(maxSplit_cv), ...
        'NumVarsToSample',    mode(nvarSample_cv), ...
        'UseSurrogateSplits', mode(useSurrogate_cv) ...
    );

    fprintf('\nBest hyperparameters (mode across folds):\n');
    fprintf('  NumTrees           = %d\n', bestParamsMode.NumTrees);
    fprintf('  MinLeafSize        = %d\n', bestParamsMode.MinLeafSize);
    fprintf('  MaxNumSplits       = %d\n', bestParamsMode.MaxNumSplits);
    fprintf('  NumVarsToSample    = %d\n', bestParamsMode.NumVarsToSample);
    fprintf('  UseSurrogateSplits = %d\n\n', bestParamsMode.UseSurrogateSplits);
end
