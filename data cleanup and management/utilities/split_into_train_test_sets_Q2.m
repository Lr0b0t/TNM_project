clc; close all; clear;


% -------------------------------------------------------------------------
%
% Purpose:
% Splits the input dataset (which is well known that contain NaNs) into a 
% training set and a test set.
%
% Key constraint: 
% - The test set must contain **only fully observed rows** 
%   (i.e., rows with no missing values).
% - All rows with at least one NaN are forced into the training set.
%
% This avoids data leakage and ensures clean model evaluation.
% We want to avoid any data leakage as well as ensure that the test set
% contains only observed values.
%
% -------------------------------------------------------------------------
% Input:
%   - CSV file: ../final files/feature_set_master.csv
%     This file should contain:
%       - Predictive features for regression 
%       - A unique identifier column named 'SCRNO'
%
%   - Additionally, one-year follow-up features will be appended by
%     automatically merging with:
%       ../final files/feature_set_followup_master.csv
%     based on matching SCRNO IDs.
%
% Output:
%   - train_features.csv : Training set (rows with or without NaNs)
%   - test_features.csv  : Test set (only rows with complete data)
%
% Usage:
%   - Run in the 'utilities' directory.
%
%--------------------------------------------------------------------------

% Define input paths
baseDir = fullfile('..', 'final files');
featureFile = fullfile(baseDir, 'feature_set_master.csv');
followUpFile = fullfile(baseDir, 'followup_m12_master.csv');

% Load baseline features
baseline = readtable(featureFile);

% Load follow-up data
followUp = readtable(followUpFile);


% Remove SCRNO from follow-up table temporarily to avoid duplication
% followUpVarsOnly = followUp(:, setdiff(followUp.Properties.VariableNames, {'SCRNO'}));

% Merge follow-up data to baseline by SCRNO
data = outerjoin(baseline, followUp, ...
    'Keys', 'SCRNO', ...
    'MergeKeys', true, ...
    'Type', 'left');

% Find rows with any missing values (across all columns now)
hasMissing = any(ismissing(data), 2);

% Extract complete rows (eligible for test set)
completeRows = data(~hasMissing, :);

% Select ~20% for test set
nTotal = height(data);
nTest = round(0.2 * nTotal);

rng(42);  % For reproducibility
perm = randperm(height(completeRows));
testSet = completeRows(perm(1:min(nTest, height(completeRows))), :);

% Identify test SCRNOs
testIdx = ismember(data.SCRNO, testSet.SCRNO);

% Assign remaining to training set
trainSet = data(~testIdx, :);

% Output file paths
trainFile = fullfile(baseDir, 'train_features_Q2.csv');
testFile  = fullfile(baseDir, 'test_features_Q2.csv');

% Save to disk
writetable(trainSet, trainFile);
writetable(testSet, testFile);

fprintf('Train set saved to %s (%d rows)\n', trainFile, height(trainSet));
fprintf('Test set saved to %s (%d rows, no NaNs)\n', testFile, height(testSet));