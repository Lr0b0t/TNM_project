clc; close all; clear;

% Purpose:
%   - Load subject IDs from two CSV files (train and test), each containing
%     an “ID” column and multiple feature columns.
%   - Load each subject’s Functional Connectivity (FC) matrix (func_connectivity.mat)
%     and Effective Connectivity (EC) matrix (rdcm_connectivity.mat) from an
%     “connectivity_n88/” folder structure.
%   - Extract basic graph‐theoretic features from each matrix:
%       • FC (undirected): node strength, node degree, mean strength, mean degree.
%       • EC (directed): in‐strength, out‐strength, in‐degree, out‐degree, asymmetry,
%         mean in‐strength, mean out‐strength, mean asymmetry.
%   - Merge the train and test sets into two combined tables—one for FC features
%     and one for EC features—so that each row is a subject, beginning with their ID.
%   - Save both combined tables to:
%       1) a MATLAB .mat file (as a table with variable names = column headers), and
%       2) a CSV file (with the first row = feature names, including “ID”).
%
% Inputs:
%   • train_features_Q2_imputed.csv
%   • test_features_Q2_imputed.csv
%       - These must reside in: 
%         ../CQ2/ML code/../../data cleanup and management/final files/
%       - The first column is subject ID, any other columns are ignored here.
%
%   • Connectivity folder: connectivity_n88/<ID>/
%       - Each subject‐ID folder contains:
%           • func_connectivity.mat   (variable "fc_mat")
%           • rdcm_connectivity.mat   (variable "output_m_all")
%       - This folder must be on the MATLAB path, for example by running:
%           >> addpath(genpath('/home/matl/Downloads/BCT'));
%
% Outputs (all placed in a subfolder “combined_graph_features” beneath the script):
%   • combined_FC_table.mat    — MATLAB table with columns: ID + FC feature names
%   • combined_FC.csv          — CSV with header row: ID,Strength_1,Strength_2,…,MeanDegree
%   • combined_EC_table.mat    — MATLAB table with columns: ID + EC feature names
%   • combined_EC.csv          — CSV with header row: ID,InStr_1,InStr_2,…,MeanAsym
%
% Dependencies:
%   • Brain Connectivity Toolbox (BCT) functions must be on the MATLAB path
%     (e.g., clustering_coef_wu, strengths_und, etc.) if you later reroute to
%     more advanced metrics. The basic features here use only sum() and logical
%     indexing, but having BCT on the path allows seamless extension.
%
% Folder structure after running (the script creates a new folder with the results):
%   <script_folder>/
%     combine_and_save_graph_features.m
%       … (other files) …
%     combined_graph_features/ 
%       combined_FC_table.mat
%       combined_FC.csv
%       combined_EC_table.mat
%       combined_EC.csv
%
% Usage:
%   1) Ensure MATLAB’s path includes:
%        - The parent folder of “connectivity_n88/”
%        - The Brain Connectivity Toolbox folder (if you plan to extend metrics)
%   2) Place this script under “CQ2/ML code/”.
%--------------------------------------------------------------------------



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

id_col   = 1;
mmse_col = find(strcmp(trainData.Properties.VariableNames, 'MMSCORE_followUp'));

% Gather IDs and targets from both sets
train_ids = trainData{:, id_col};
test_ids  = testData{:,  id_col};
all_ids   = [train_ids; test_ids];  % Combined ID vector

Y_train = trainData{:, mmse_col};    % (Not used further here, but available)
Y_test  = testData{:,  mmse_col};
all_Y   = [Y_train; Y_test];

pad_id = @(id) sprintf('%07d', id);

fprintf('\nLoading TRAIN EC from rDCM matrices...\n');
ec_rDCM_train = load_rdcm_matrices(train_ids, connDir, pad_id);
fprintf('Loading TEST EC from rDCM matrices...\n');
ec_rDCM_test  = load_rdcm_matrices(test_ids,  connDir, pad_id);
ec_all        = [ec_rDCM_train; ec_rDCM_test];

fprintf('\nLoading TRAIN FC matrices...\n');
fc_train = load_fc_matrices(train_ids, connDir, pad_id);
fprintf('Loading TEST FC matrices...\n');
fc_test  = load_fc_matrices(test_ids,  connDir, pad_id);
fc_all   = [fc_train; fc_test];

%% ---- EXTRACT GRAPH-THEORETIC FEATURES ----
% 1) Functional Connectivity features (undirected)
fprintf('\nExtracting functional (undirected) graph features for FC matrices...\n');
[X_train_FC, FC_labels] = extract_fc_graph_features(fc_train);
[X_test_FC,  ~]         = extract_fc_graph_features(fc_test);
X_all_FC = [X_train_FC; X_test_FC];  % Combine train+test

% 2) Effective Connectivity features (directed)
fprintf('\nExtracting effective (directed) graph features for EC from rDCM matrices...\n');
[X_train_EC, EC_labels] = extract_ec_graph_features(ec_rDCM_train);
[X_test_EC,  ~]         = extract_ec_graph_features(ec_rDCM_test);
X_all_EC = [X_train_EC; X_test_EC];

%% ---- SAVE COMBINED FEATURES AS TABLES WITH HEADER ROWS ----
% Create an output directory under baseDir
outputDir = fullfile(baseDir, 'combined_graph_features');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
    fprintf('Created output directory: %s\n', outputDir);
end

%%% --- Functional Connectivity (FC) ---%  
% Prepare table: first column "ID", then FC_labels
FC_varNames = ['ID', FC_labels];            % 1×(2N+3) cell array of strings
T_FC = array2table([all_ids, X_all_FC], 'VariableNames', FC_varNames);

% Save to .mat
save(fullfile(outputDir, 'combined_FC_table.mat'), 'T_FC');

% Also write to CSV (this will put variable names as the first row)
writetable(T_FC, fullfile(outputDir, 'combined_FC.csv'));

fprintf('\nSaved combined FC features to:\n');
fprintf('  %s\n', fullfile(outputDir, 'combined_FC_table.mat'));
fprintf('  %s\n', fullfile(outputDir, 'combined_FC.csv'));

%%% --- Effective Connectivity (EC) ---%  
% Prepare table: first column "ID", then EC_labels
EC_varNames = ['ID', EC_labels];            % 1×(5N+4) cell array of strings
T_EC = array2table([all_ids, X_all_EC], 'VariableNames', EC_varNames);

% Save to .mat
save(fullfile(outputDir, 'combined_EC_table.mat'), 'T_EC');

% Also write to CSV
writetable(T_EC, fullfile(outputDir, 'combined_EC.csv'));

fprintf('\nSaved combined EC features to:\n');
fprintf('  %s\n', fullfile(outputDir, 'combined_EC_table.mat'));
fprintf('  %s\n', fullfile(outputDir, 'combined_EC.csv'));

fprintf('\nAll combined feature tables are ready in %s\n', outputDir);

%% --- HELPER FUNCTIONS ---

function matrices = load_rdcm_matrices(ids, connDir, pad_id)
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

function matrices = load_fc_matrices(ids, connDir, pad_id)
    matrices = cell(length(ids),1);
    for i = 1:length(ids)
        folderName = pad_id(ids(i));
        matFile = fullfile(connDir, folderName, 'func_connectivity.mat');
        if exist(matFile, 'file')
            data = load(matFile);
            if isfield(data, 'fc_mat')
                matrices{i} = data.fc_mat;
            else
                error('fc_mat variable not found in %s', matFile);
            end
        else
            error('File not found: %s', matFile);
        end
    end
end

% function [X_fc, feature_labels] = extract_fc_graph_features(fc_matrices)
% % EXTRACT_FC_GRAPH_FEATURES  Extracts undirected graph metrics from FC
% %
% % Inputs:
% %   fc_matrices : cell array of [N×N] symmetric FC matrices (weights ≥ 0)
% %
% % Outputs:
% %   X_fc           : [nSubjects × nFeatures] numeric matrix
% %   feature_labels : 1×nFeatures cell array of labels
% %
% % Feature order per subject:
% %   1–N    Node strength      (sum of weights per node)
% %   N+1–2N Node degree        (count of nonzero weights per node)
% %   2N+1   Global mean strength (scalar)
% %   2N+2   Global mean degree   (scalar)
% 
%     nSubj = numel(fc_matrices);
%     N     = size(fc_matrices{1}, 1);
% 
%     nFeat = 2*N + 2;
%     X_fc  = zeros(nSubj, nFeat);
% 
%     % Create labels
%     feature_labels = cell(1, nFeat);
%     idx = 1;
%     for v = 1:N
%         feature_labels{idx} = sprintf('Strength_%d', v);
%         idx = idx + 1;
%     end
%     for v = 1:N
%         feature_labels{idx} = sprintf('Degree_%d', v);
%         idx = idx + 1;
%     end
%     feature_labels{idx}   = 'MeanStrength'; idx = idx + 1;
%     feature_labels{idx}   = 'MeanDegree';
% 
%     % Compute features
%     for i = 1:nSubj
%         W = fc_matrices{i};
%         W(W < 0) = 0;  % ensure non-negative
% 
%         % Node strength
%         str = sum(W, 2);    % N×1
% 
%         % Node degree (binary)
%         deg = sum(W > 0, 2); % N×1
% 
%         % Global summaries
%         meanStr = mean(str);
%         meanDeg = mean(deg);
% 
%         X_fc(i, :) = [str', deg', meanStr, meanDeg];
%     end
% end

function [X_fc, feature_labels] = extract_fc_graph_features(fc_matrices)
% EXTRACT_FC_GRAPH_FEATURES  Extracts undirected graph metrics from FC
%   Inputs:
%     fc_matrices : cell array of [N×N] symmetric FC matrices (weights ≥0)
%   Outputs:
%     X_fc           : [nSubjects × nFeatures] feature matrix
%     feature_labels : 1×nFeatures cell array of labels
%
%   Feature order:
%     1) Strength (N)
%     2) Clustering coefficient (N)
%     3) Local efficiency (N)
%     4) Degree (N)
%     5) Betweenness centrality (N)
%     6) Characteristic path length (1)
%     7) Global efficiency (1)
%     8) Modularity Q (1)
%     9) Rich-club coefficient (1)

    nSubj = numel(fc_matrices);
    N     = size(fc_matrices{1},1);
    X_fc  = zeros(nSubj, 5*N + 4);

    % Pre-generate feature labels
    feature_labels = cell(1, 5*N + 4);
    idx = 1;
    for name = ["Strength_", "ClustCoeff_", "LocalEff_", "Degree_", "BetwCent_"]
        for v = 1:N
            feature_labels{idx} = char(name + v);
            idx = idx + 1;
        end
    end
    feature_labels{idx}   = 'CharPathLength'; idx = idx + 1;
    feature_labels{idx}   = 'GlobalEff';      idx = idx + 1;
    feature_labels{idx}   = 'ModularityQ';     idx = idx + 1;
    feature_labels{idx}   = 'RichClubCoeff';

    % Check availability of BCT functions
    has_strengths    = exist('strengths_und','file') == 2;
    if ~has_strengths,    fprintf('  Warning: strengths_und not found; using sum(W,2).\n'); end
    has_clust         = exist('clustering_coef_wu','file') == 2;
    if ~has_clust,        fprintf('  Warning: clustering_coef_wu not found; zeros used.\n'); end
    has_effi          = exist('efficiency_wei','file') == 2;
    if ~has_effi,         fprintf('  Warning: efficiency_wei not found; zeros/NaN used.\n'); end
    has_btw           = exist('betweenness_wei','file') == 2;
    if ~has_btw,          fprintf('  Warning: betweenness_wei not found; zeros used.\n'); end
    has_dist          = exist('distance_wei','file') == 2 && exist('charpath','file') == 2;
    if ~has_dist,         fprintf('  Warning: distance_wei/charpath not found; CPL=NaN.\n'); end
    has_modlouvain    = exist('modularity_louvain_und','file') == 2;
    if ~has_modlouvain,   fprintf('  Warning: modularity_louvain_und not found; Q=NaN.\n'); end
    has_richclub      = exist('rich_club_wu','file') == 2;
    if ~has_richclub,     fprintf('  Warning: rich_club_wu not found; rc=NaN.\n'); end

    for i = 1:nSubj
        W = fc_matrices{i};
        W(W<0) = 0;

        % 1) Strength
        if has_strengths
            str = strengths_und(W);
        else
            str = sum(W,2)';
        end

        % 2) Clustering coefficient
        if has_clust
            C = clustering_coef_wu(W);
        else
            C = zeros(1,N);
        end

        % 3) Local efficiency
        if has_effi
            Eloc = efficiency_wei(W,1);
        else
            Eloc = zeros(1,N);
        end

        % 4) Degree (binary)
        deg = sum(W>0,2)';

        % 5) Betweenness centrality
        if has_btw
            BTW = betweenness_wei(1./W);
        else
            BTW = zeros(1,N);
        end
        % 6) Characteristic path length
        if has_dist
            D   = distance_wei(1./W);
            CPL = charpath(D,0,0);
        else
            CPL = NaN;
        end
        % 7) Global efficiency
        if has_effi
            Eglob = efficiency_wei(W);
        else
            Eglob = NaN;
        end

                disp('heyyyyyyyyyyyyyyyyyy')

        % 9) Rich-club coefficient
        if has_richclub
            R  = rich_club_wu(W);
            rc = R(end);
        else
            rc = NaN;
        end
                disp('heyyyyyyyyyyyyyyyyyy2222')

        X_fc(i,:) = [str, C, Eloc, deg, BTW, CPL, Eglob, rc];
    end
end
function [X_ec, feature_labels] = extract_ec_graph_features(ec_matrices)
% EXTRACT_EC_GRAPH_FEATURES  Basic directed graph metrics from EC matrices
%
% Inputs:
%   ec_matrices : cell array of [N×N] directed EC matrices
%
% Outputs:
%   X_ec           : [nSubjects × nFeatures] numeric matrix
%   feature_labels : 1×nFeatures cell array of labels
%
% Feature order per subject:
%   1–N    In-strength    (sum of incoming weights per node)
%   N+1–2N Out-strength   (sum of outgoing weights per node)
%   2N+1–3N In-degree      (count of incoming edges per node)
%   3N+1–4N Out-degree     (count of outgoing edges per node)
%   4N+1–5N Asymmetry      (out_strength – in_strength per node)
%   5N+1   Global mean in-strength   (scalar)
%   5N+2   Global mean out-strength  (scalar)
%   5N+3   Global mean asymmetry     (scalar)

    nSubj = numel(ec_matrices);
    N     = size(ec_matrices{1}, 1);
    
    nFeat = 5*N + 3;
    X_ec  = zeros(nSubj, nFeat);
    
    % Create labels
    feature_labels = cell(1, nFeat);
    idx = 1;
    for v = 1:N
        feature_labels{idx} = sprintf('InStr_%d', v);
        idx = idx + 1;
    end
    for v = 1:N
        feature_labels{idx} = sprintf('OutStr_%d', v);
        idx = idx + 1;
    end
    for v = 1:N
        feature_labels{idx} = sprintf('InDeg_%d', v);
        idx = idx + 1;
    end
    for v = 1:N
        feature_labels{idx} = sprintf('OutDeg_%d', v);
        idx = idx + 1;
    end
    for v = 1:N
        feature_labels{idx} = sprintf('Asym_%d', v);
        idx = idx + 1;
    end
    feature_labels{idx}   = 'MeanInStr'; idx = idx + 1;
    feature_labels{idx}   = 'MeanOutStr'; idx = idx + 1;
    feature_labels{idx}   = 'MeanAsym';
    
    % Compute features
    for i = 1:nSubj
        W = ec_matrices{i};
        W(logical(eye(N))) = 0;  % remove self-connections
        
        % In-strength
        instr = sum(W, 1);   % 1×N
        
        % Out-strength
        outstr = sum(W, 2)'; % 1×N
        
        % In-degree
        indeg = sum(W > 0, 1); % 1×N
        
        % Out-degree
        outdeg = sum(W > 0, 2)'; % 1×N
        
        % Asymmetry
        asym = outstr - instr; % 1×N
        
        % Global means
        meanInStr  = mean(instr);
        meanOutStr = mean(outstr);
        meanAsym   = mean(asym);
        
        X_ec(i, :) = [instr, outstr, indeg, outdeg, asym, meanInStr, meanOutStr, meanAsym];
    end
end
