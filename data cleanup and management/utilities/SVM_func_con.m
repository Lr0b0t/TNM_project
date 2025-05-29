close all; clear; clc;


% Paths
baseDir = fileparts(pwd);
dataDir = fullfile(baseDir, 'final files');
connDir = fullfile(dataDir, 'connectivity_n88');

% Load train/test splits (for IDs and MMSE follow-up)
trainFile = fullfile(dataDir, 'train_features_Q2.csv');
testFile  = fullfile(dataDir, 'test_features_Q2.csv');
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

%%
for k = 1:length(train_matrices)
    figure(1); clf;
    imagesc(train_matrices{k});
    title(['Train subject ', num2str(train_ids(k)), ' (', num2str(k), '/', num2str(length(train_matrices)), ')']);
    colorbar;
    % Optionally set same color limits for all
    % caxis([-1 1]);
    xlabel('Node');
    ylabel('Node');
    % Pause for you to inspect before showing next matrix
    disp('Press any key to continue to the next matrix...');
    pause;
end

for k = 1:length(test_matrices)
    figure(2); clf;
    imagesc(test_matrices{k});
    title(['Test subject ', num2str(test_ids(k)), ' (', num2str(k), '/', num2str(length(test_matrices)), ')']);
    colorbar;
    % Optionally set same color limits for all
    % caxis([-1 1]);
    xlabel('Node');
    ylabel('Node');
    disp('Press any key to continue to the next matrix...');
    pause;
end

%%
% Assume all matrices are square, size [N x N]
N = size(train_matrices{1}, 1);
triu_mask = triu(true(N), 1); % upper triangle, no diagonal

% Vectorize train matrices
X_train = zeros(length(train_matrices), sum(triu_mask(:)));
for i = 1:length(train_matrices)
    mat = train_matrices{i};
    X_train(i, :) = mat(triu_mask);
end

% Vectorize test matrices
X_test = zeros(length(test_matrices), sum(triu_mask(:)));
for i = 1:length(test_matrices)
    mat = test_matrices{i};
    X_test(i, :) = mat(triu_mask);
end

fprintf('Train matrix shape: %d x %d\n', size(X_train, 1), size(X_train, 2));
fprintf('Test matrix shape: %d x %d\n', size(X_test, 1), size(X_test, 2));
