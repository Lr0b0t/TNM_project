%--------------------------------------------------------------------------
% Script: elasticnet_fc_regression.m
%
% Purpose:
%   - Predict MMSE using elastic net regression, using FC matrices as input.
%   - Dimensionality reduction: vectorized, PCA, or graph-theory features.
%   - Nested CV (R^2 selection, NO leakage). Final model tested on external test set.
%
% Inputs:
%   - train_features_Q2_imputed.csv, test_features_Q2_imputed.csv
%   - connectivity_n88/<ID>/func_connectivity.mat (with 'fc_mat')
%
% Usage:
%   Set 'dim_reduc' to 'vectorized', 'pca', or 'graph' below.
%--------------------------------------------------------------------------

clc; clear; close all;

%% ---- USER OPTION: choose dim reduction type ----
dim_reduc = 'graph';  
% 'vectorized', 'pca', or 'graph'

%% ---- PATH HANDLING ----
% Find baseDir (where this script lives)
baseDir = pwd;

% Path to features CSVs (relative to here)
featuresDir = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(featuresDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(featuresDir, 'test_features_Q2_imputed.csv');

% Path to connectivity
connDir = fullfile(baseDir, 'connectivity_n88');
if ~exist(connDir, 'dir')
    connDir = fullfile(baseDir, '..','..', 'connectivity_n88');
end
if ~exist(connDir, 'dir')
    error('Could not find connectivity_n88 folder.');
end

%% ---- LOAD DATA ----
trainData = readtable(trainFile);
testData  = readtable(testFile);

id_col = 1;
mmse_col = find(strcmp(trainData.Properties.VariableNames, 'MMSCORE_followUp'));
train_ids = trainData{:, id_col};
test_ids  = testData{:, id_col};
Y_train = trainData{:, mmse_col};
Y_test  = testData{:, mmse_col};
pad_id = @(id) sprintf('%07d', id);

fprintf('\nLoading TRAIN FC matrices...\n');
fc_train = load_fc_matrices(train_ids, connDir, pad_id);
fprintf('Loading TEST FC matrices...\n');
fc_test  = load_fc_matrices(test_ids,  connDir, pad_id);


function matrices = load_fc_matrices(ids, connDir, pad_id)
    matrices = cell(length(ids),1);
    for i = 1:length(ids)
        folderName = pad_id(ids(i));
        matFile = fullfile(connDir, folderName, 'rdcm_connectivity.mat');
        if exist(matFile, 'file')
            data = load(matFile);
            if isfield(data, 'output_m_all')
                matrices{i} = data.output_m_all;
            else
                error('output_m_all variable not found in %s', matFile);
            end
        else
            error('File not found: %s', matFile);
        end
    end
end

function [X, feature_labels, idx_mask] = vectorize_rdcm_train(matrices)
    % Only use train matrices to create the mask!
    N = size(matrices{1}, 1);
    mask = ~eye(N);  % Off-diagonal elements only
    % Find any entry that's nonzero across the TRAINING SET
    nonzero_mask = false(N,N);
    for i = 1:length(matrices)
        nonzero_mask = nonzero_mask | (matrices{i} ~= 0);
    end
    mask = mask & nonzero_mask; % Only keep nonzero off-diag elements
    idx_mask = find(mask);
    X = zeros(length(matrices), length(idx_mask));
    for i = 1:length(matrices)
        X(i,:) = matrices{i}(idx_mask);
    end
    [row, col] = ind2sub([N,N], idx_mask);
    feature_labels = arrayfun(@(i,j) sprintf('Conn_%d_%d',i,j), row, col, 'UniformOutput', false);
end

function X = vectorize_rdcm_test(matrices, idx_mask)
    % Apply the mask from the TRAINING SET!
    X = zeros(length(matrices), length(idx_mask));
    for i = 1:length(matrices)
        X(i,:) = matrices{i}(idx_mask);
    end
end

function labels = vectorized_labels(N)
    mask = triu(true(N), 1);
    [row, col] = find(mask);
    labels = arrayfun(@(i,j) sprintf('Conn_%d_%d',i,j), row, col, 'UniformOutput', false);
end

function Xg = extract_graph_features(matrices)
    N = size(matrices{1}, 1);
    Xg = zeros(length(matrices), N*4 + 2); % strength, clustering, local eff, degree, + 2 global
    for i = 1:length(matrices)
        A = matrices{i};
        A(A < 0) = 0;
        if exist('strengths_und', 'file')
            node_strength = strengths_und(A); % 1 x N
        else
            node_strength = sum(A,2)';
        end
        if exist('clustering_coef_wu', 'file')
            clustering = clustering_coef_wu(A);
        else
            clustering = zeros(1,N);
        end
        if exist('efficiency_wei', 'file')
            local_eff = efficiency_wei(A, 1);
            global_eff = efficiency_wei(A);
        else
            local_eff = zeros(1,N);
            global_eff = 0;
        end
        degree = sum(A > 0, 2)';
        mean_strength = mean(node_strength);
        Xg(i,:) = [node_strength, clustering, local_eff, degree, global_eff, mean_strength];
    end
end

function labels = graph_feature_labels(N)
    labels = [arrayfun(@(i) sprintf('Strength_%d', i), 1:N, 'UniformOutput',false), ...
              arrayfun(@(i) sprintf('Clustering_%d', i), 1:N, 'UniformOutput',false), ...
              arrayfun(@(i) sprintf('LocalEff_%d', i), 1:N, 'UniformOutput',false), ...
              arrayfun(@(i) sprintf('Degree_%d', i), 1:N, 'UniformOutput',false), ...
              {'GlobalEff', 'MeanStrength'}];
end
