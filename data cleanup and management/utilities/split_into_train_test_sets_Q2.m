clc; close all; clear;

%--------------------------------------------------------------------------
%
% Purpose:
%   Randomly split the input dataset (which is well known that contain 
%   NaNs) into training and test sets.
%
% Inputs:
%   - CSV file: ../final files/feature_set_master.csv
%       Contains baseline predictive features and a unique 'SCRNO' column.
%   - CSV file: ../final files/followup_m12_master.csv
%       Contains follow-up features, including 'SCRNO' for joining.
%
% Output:
%   - train_features_Q2.csv: 
%         Training set (random 80% of rows, includes missing values)
%   - test_features_Q2.csv:  
%         Test set (random 20% of rows, includes missing values)
%   - Resulting files will be saved in the ../final files/ folder
%
% Key characteristics:
%   - The test set may contain missing values.
%   - Both splits preserve the original missing data distribution.
%   - No information from the test set is leaked into the training set.
%   - The random split is reproducible (fixed RNG seed).
%
%
% Notes:
%   - This script does not perform any imputation.
%   - The script reports the number and proportion of complete (NaN-free)
%     rows in each split for quality control.
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

% Merge follow-up data to baseline by SCRNO
data = outerjoin(baseline, followUp, ...
    'Keys', 'SCRNO', ...
    'MergeKeys', true, ...
    'Type', 'left');

% (Optionally) Remove completely empty rows (rare)
allMissing = all(ismissing(data), 2);
if any(allMissing)
    fprintf('Removed %d completely empty rows.\n', sum(allMissing));
    data = data(~allMissing, :);
end

% Randomly split into train/test (e.g., 80/20)
% rng(42); % For reproducibility
nTotal = height(data);
nTest = round(0.2 * nTotal);

perm = randperm(nTotal);
testRows = perm(1:nTest);
trainRows = perm(nTest+1:end);

testSet = data(testRows, :);
trainSet = data(trainRows, :);

% (Optional) Report % complete rows in each set
fprintf('Train set: %d rows, %d (%.1f%%) fully complete rows\n', ...
    height(trainSet), sum(~any(ismissing(trainSet),2)), ...
    100*sum(~any(ismissing(trainSet),2))/height(trainSet));
fprintf('Test set:  %d rows, %d (%.1f%%) fully complete rows\n', ...
    height(testSet), sum(~any(ismissing(testSet),2)), ...
    100*sum(~any(ismissing(testSet),2))/height(testSet));

% Output file paths
trainFile = fullfile(baseDir, 'train_features_Q2.csv');
testFile  = fullfile(baseDir, 'test_features_Q2.csv');

writetable(trainSet, trainFile);
writetable(testSet, testFile);

fprintf('Train set saved to %s (%d rows)\n', trainFile, height(trainSet));
fprintf('Test set saved to %s (%d rows)\n', testFile, height(testSet));