function [mean_rmse, mean_r2] = svm_inner_cv(X, Y, cv, kernel, C, sigma, polyorder, eps)
    k = cv.NumTestSets;
    rmse_inner = zeros(k, 1);
    r2_inner = zeros(k, 1);
    for j = 1:k
        trainIdx = training(cv, j);
        valIdx = test(cv, j);
        X_in = X(trainIdx, :);
        Y_in = Y(trainIdx);
        X_val = X(valIdx, :);
        Y_val = Y(valIdx);

        switch kernel
            case 'rbf'
                model = fitrsvm(X_in, Y_in, 'KernelFunction', 'rbf', ...
                    'BoxConstraint', C, 'KernelScale', sigma, 'Epsilon', eps, 'Standardize', false);
            case 'linear'
                model = fitrsvm(X_in, Y_in, 'KernelFunction', 'linear', ...
                    'BoxConstraint', C, 'Epsilon', eps, 'Standardize', false);
            case 'polynomial'
                model = fitrsvm(X_in, Y_in, 'KernelFunction', 'polynomial', ...
                    'PolynomialOrder', polyorder, 'BoxConstraint', C, 'Epsilon', eps, 'Standardize', false);
        end
        Y_pred = predict(model, X_val);
        rmse_inner(j) = sqrt(mean((Y_val - Y_pred).^2));
        r2_inner(j) = 1 - sum((Y_val - Y_pred).^2) / sum((Y_val - mean(Y_val)).^2);
    end
    mean_rmse = mean(rmse_inner);
    mean_r2 = mean(r2_inner);
end



