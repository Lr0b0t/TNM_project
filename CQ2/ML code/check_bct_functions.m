%==========================================================================
% CHECK_BCT_FUNCTIONS.M
%   Verify presence of Brain Connectivity Toolbox routines needed by 
%   extract_fc_graph_features.m and extract_ec_graph_features.m.
%
%   Run this in your MATLAB session. It will print which functions exist
%   on the path and which are missing.
%==========================================================================

% List of BCT functions used in extract_fc_graph_features:
fc_funcs = { ...
    'strengths_und', ...
    'clustering_coef_wu', ...
    'efficiency_wei', ...
    'distance_wei', ...
    'charpath', ...
    'betweenness_wei', ...
    'modularity_louvain_und', ...
    'rich_club_wu' ...
};

% List of BCT functions used in extract_ec_graph_features:
ec_funcs = { ...
    'clustering_coef_wd', ...
    'betweenness_bin', ...
    'pagerank_centrality', ...
    'efficiency_bin' ...
};

% Combine and remove duplicates
all_funcs = unique([fc_funcs, ec_funcs]);

fprintf('\n=== Checking Brain Connectivity Toolbox functions on MATLAB path ===\n\n');

missing = {};
for i = 1:numel(all_funcs)
    fname = all_funcs{i};
    if exist(fname, 'file') == 2
        fprintf('  [OK]   %s\n', fname);
    else
        fprintf('  [MISSING] %s\n', fname);
        missing{end+1} = fname; 
    end
end

if isempty(missing)
    fprintf('\nAll required BCT functions are present.\n');
else
    fprintf('\nThe following BCT functions were NOT FOUND on your path:\n');
    for i = 1:numel(missing)
        fprintf('   - %s\n', missing{i});
    end
    fprintf('\nYou need to download and add the Brain Connectivity Toolbox (BCT) folder\n');
    fprintf('to your MATLAB path. For example:\n');
    fprintf('   >> addpath(genpath(fullfile(<path_to_BCT_folder>)));\n\n');
end
