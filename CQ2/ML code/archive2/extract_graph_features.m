%--------------------------------------------------------------------------
% Script: randomforest_fc_regression.m
%
% Purpose:
%   - Predict MMSE using Random Forest regression, using FC matrices as input.
%   - Dimensionality reduction: vectorized, PCA, or graph-theory features.
%   - Nested CV (R^2 selection, NO leakage). Final model tested on external test set.
%
% Inputs:
%   - train_features_Q2_imputed.csv, test_features_Q3_imputed.csv
%   - connectivity_n88/<ID>/rdcm_connectivity.mat (with 'output_m_all')
%
% Usage:
%   Set 'dim_reduc' to 'vectorized', 'pca', or 'graph' below.
%--------------------------------------------------------------------------

clc; clear; close all;

baseDir = pwd;
featuresDir = fullfile('..','..', 'data cleanup and management', 'final files');
trainFile = fullfile(featuresDir, 'train_features_Q2_imputed.csv');
testFile  = fullfile(featuresDir, 'test_features_Q2_imputed.csv');

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


X_train = extract_ec_graph_features(fc_train);
X_test  = extract_ec_graph_features(fc_test);
feature_labels = graph_feature_labels(size(fc_train{1},1));

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


function X_fc = extract_fc_graph_features(fc_matrices)
% EXTRACT_FC_GRAPH_FEATURES  Extracts key undirected graph metrics from FC
%
% Inputs:
%   fc_matrices : cell array of [N×N] symmetric FC matrices (correlation or weights)
%
% Outputs:
%   X_fc : [nSubjects × nFeatures] matrix with features in this order:
%     - Node strength (N)
%     - Clustering coefficient (N)
%     - Local efficiency (N)
%     - Degree (N)
%     - Betweenness centrality (N)
%     - Characteristic path length (1)
%     - Global efficiency (1)
%     - Modularity Q (1)
%     - Rich-club coefficient (1)
%
% Requires Brain Connectivity Toolbox (BCT) for weighted measures.

    nSubj = numel(fc_matrices);
    N     = size(fc_matrices{1},1);
    % total features = 5*N + 4 (five node-level vectors, four scalars)
    X_fc = zeros(nSubj, 5*N + 4);

    for i = 1:nSubj
        W = fc_matrices{i};
        W(W<0) = 0;  % ensure non-negative weights

        % 1) Strength (weighted degree)
        if exist('strengths_und','file')
            str = strengths_und(W);     % 1×N
        else
            str = sum(W,2)';            % fallback
        end

        % 2) Clustering coefficient (weighted)
        if exist('clustering_coef_wu','file')
            C = clustering_coef_wu(W);
        else
            C = zeros(1,N);
        end

        % 3) Local efficiency
        if exist('efficiency_wei','file')
            Eloc = efficiency_wei(W,1);
        else
            Eloc = zeros(1,N);
        end

        % 4) Degree (binary)
        deg = sum(W>0,2)';

        % 5) Betweenness centrality (weighted)
        if exist('betweenness_wei','file')
            BTW = betweenness_wei(1./W);
        else
            BTW = zeros(1,N);
        end

        % 6) Characteristic path length
        if exist('distance_wei','file') && exist('charpath','file')
            D   = distance_wei(1./W);
            CPL = charpath(D,0,0);
        else
            CPL = NaN;
        end

        % 7) Global efficiency
        if exist('efficiency_wei','file')
            Eglob = efficiency_wei(W);
        else
            Eglob = NaN;
        end

        % 8) Modularity Q (Louvain)
        if exist('modularity_louvain_und','file')
            [~, Q] = modularity_louvain_und(W);
        else
            Q = NaN;
        end

        % 9) Rich-club coefficient (highest level)
        if exist('rich_club_wu','file')
            R = rich_club_wu(W);
            rc = R(end);
        else
            rc = NaN;
        end

        % assemble
        X_fc(i,:) = [str, C, Eloc, deg, BTW, CPL, Eglob, Q, rc];
    end
end



function X_ec = extract_ec_graph_features(ec_matrices)
% EXTRACT_EC_GRAPH_FEATURES  Extracts key directed graph metrics from EC
%
% Inputs:
%   ec_matrices : cell array of [N×N] directed EC matrices (e.g., rDCM)
%
% Outputs:
%   X_ec : [nSubjects × nFeatures] matrix with features in this order:
%     - In-strength (N)
%     - Out-strength (N)
%     - In-degree (N)
%     - Out-degree (N)
%     - Asymmetry (N) = out_strength – in_strength
%     - Directed clustering coeff. (N)
%     - Directed betweenness (N)
%     - PageRank centrality (N)
%     - Global directed efficiency (1)
%
% Requires BCT functions for directed metrics.

    nSubj = numel(ec_matrices);
    N     = size(ec_matrices{1},1);
    % total features = 5*N + 1 (five node-level vectors, one global scalar)
    X_ec = zeros(nSubj, 5*N + 1);

    for i = 1:nSubj
        W = ec_matrices{i};
        W(logical(eye(N))) = 0;  % remove self-connections

        % 1) In-strength
        instr = sum(W,1);   % 1×N
        % 2) Out-strength
        outstr = sum(W,2)'; % 1×N

        % 3) In-degree
        indeg = sum(W>0,1);
        % 4) Out-degree
        outdeg = sum(W>0,2)';

        % 5) Asymmetry (sender vs receiver)
        asym = outstr - instr;

        % 6) Directed clustering coefficient
        if exist('clustering_coef_wd','file')
            Cdir = clustering_coef_wd(W);
        else
            Cdir = zeros(1,N);
        end

        % 7) Directed betweenness
        if exist('betweenness_bin','file')
            BTWdir = betweenness_bin(W>0);
        else
            BTWdir = zeros(1,N);
        end

        % 8) PageRank centrality
        if exist('pagerank_centrality','file')
            PR = pagerank_centrality(W,0.85);
        else
            PR = zeros(1,N);
        end

        % 9) Global directed efficiency
        if exist('efficiency_bin','file')
            Eglob_dir = efficiency_bin(W>0);
        else
            Eglob_dir = NaN;
        end

        % assemble
        X_ec(i,:) = [instr, outstr, indeg, outdeg, asym, Cdir, BTWdir, PR, Eglob_dir];
    end
end
