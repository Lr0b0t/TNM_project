clc; close all; clear;


% Add path if needed
baseDir = fileparts(pwd); % go up from /utilities/
dataDir = fullfile(baseDir, 'final files');

trainFile = fullfile(dataDir, 'train_features_Q2.csv');
testFile  = fullfile(dataDir, 'test_features_Q2.csv');

% Load training data
trainData = readtable(trainFile);

% Remove ID column (assuming it's the first column)
trainData(:,1) = [];

% Again identify numeric variables after removing the ID
numericVars = varfun(@isnumeric, trainData, 'OutputFormat', 'uniform');
numericData = trainData{:, numericVars};

%%




% Normalize the features (z-score)
[normData, mu, sigma] = zscore(numericData);

% Target variables (assuming the last 3 columns are the cognitive decline scores)
Y = normData(:, end-2:end);

% Features (all columns except the last 3)
X = normData(:, 1:end-3);






% Try different imputation methods
imputeMethods = {'mean', 'median', 'knn', 'movmean', 'linear'};

for i = 1:length(imputeMethods)
    method = imputeMethods{i};
    fprintf('Trying method: %s\n', method);
    try
        filledData = impute_data(numericData, method);
        % Check if it worked
        if any(isnan(filledData), 'all')
            fprintf('  --> Still has NaNs!\n');
        else
            fprintf('  --> Success, no NaNs remaining.\n');
        end
    catch ME
        fprintf('  --> Error with method %s: %s\n', method, ME.message);
    end
end



















function filled = impute_data(data, method)
    switch lower(method)
        case 'mean'
            filled = fillmissing(data, 'constant', mean(data, 'omitnan'));
            
        case 'median'
            filled = fillmissing(data, 'constant', median(data, 'omitnan'));
            
        case 'knn'
            % Requires Statistics and Machine Learning Toolbox
            filled = knnimpute(data')';  % transpose for knnimpute format
            
        case 'movmean'
            filled = fillmissing(data, 'movmean', 5);
            
        case 'pchip'
            filled = fillmissing(data, 'pchip', 2);  % Piecewise cubic interpolation
            
        case 'linear'
            filled = fillmissing(data, 'linear');
            
        case 'nearest'
            filled = fillmissing(data, 'nearest');
            
        otherwise
            error('Unknown method: %s', method);
    end
end
