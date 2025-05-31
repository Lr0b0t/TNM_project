clear; close all; %clc; 

%--------------------------------------------------------------------------
% Purpose:
%   Impute missing data in train and test sets separately using the column mean,
%   and save the results as new CSV files with '_imputed' in the name.
%
% Rounding:
%   - All imputed values: round to nearest integer, except:
%   - For 'CDSOB_baseline' and 'CDSOB_followUp': round to nearest 0.5.
%
% Inputs:
%   - train_features_Q2.csv: Training set (contains NaNs)
%   - test_features_Q2.csv:  Test set (contains NaNs)
%
% Outputs:
%   - train_features_Q2_imputed.csv: Training set with missing values replaced
%   - test_features_Q2_imputed.csv:  Test set with missing values replaced
%
% Notes:
%   - Imputation is performed **separately** on train and test sets
%     (means computed only within each set), so no leakage occurs.
%--------------------------------------------------------------------------

% Define paths
baseDir = fullfile('..', 'final files');
trainFile = fullfile(baseDir, 'train_features_Q2.csv');
testFile  = fullfile(baseDir, 'test_features_Q2.csv');

trainOutFile = fullfile(baseDir, 'train_features_Q2_imputed.csv');
testOutFile  = fullfile(baseDir, 'test_features_Q2_imputed.csv');

specialCols = {'CDSOB_baseline', 'CDSOB_followUp'};

% Read train and test sets
train_tbl = readtable(trainFile);
test_tbl  = readtable(testFile);

% Identify numeric columns (to impute)
isNumTrain = varfun(@isnumeric, train_tbl, 'OutputFormat', 'uniform');
isNumTest  = varfun(@isnumeric, test_tbl,  'OutputFormat', 'uniform');

numVarsTrain = train_tbl.Properties.VariableNames(isNumTrain);
numVarsTest  = test_tbl.Properties.VariableNames(isNumTest);

% Impute train set (with rounding)
for i = 1:numel(numVarsTrain)
    col = numVarsTrain{i};
    x = train_tbl.(col);
    if any(isnan(x))
        m = mean(x(~isnan(x)));
        idx = isnan(x);
        if ismember(col, specialCols)
            % Round to nearest 0.5 for special columns
            imputed_vals = round(m * 2) / 2;
        else
            % Round to nearest integer for all other columns
            imputed_vals = round(m);
        end
        x(idx) = imputed_vals;
        train_tbl.(col) = x;
    end
end

% Impute test set (with rounding)
for i = 1:numel(numVarsTest)
    col = numVarsTest{i};
    x = test_tbl.(col);
    if any(isnan(x))
        m = mean(x(~isnan(x)));
        idx = isnan(x);
        if ismember(col, specialCols)
            imputed_vals = round(m * 2) / 2;
        else
            imputed_vals = round(m);
        end
        x(idx) = imputed_vals;
        test_tbl.(col) = x;
    end
end

% Save imputed tables to new files
writetable(train_tbl, trainOutFile);
writetable(test_tbl, testOutFile);

fprintf('Imputed train set saved to %s (%d rows)\n', trainOutFile, height(train_tbl));
fprintf('Imputed test set saved to %s (%d rows)\n', testOutFile, height(test_tbl));