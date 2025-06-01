% Assume your matrix is called A (246x246)
load('pca_matrix_rdcm.mat');  % or however you have A defined
%%
close all 
figure('Position', [100, 100, 1400, 600]);  % Bigger figure window

% Prepare subplot 1 (PC1)
ax1 = subplot(1, 2, 1);
plot_effective_connectivity(ax1, binary_otsu_pc1, group_labels);
title('Effective Connectivity PC1', 'FontSize', 14);

% Prepare subplot 2 (PC2)
ax2 = subplot(1, 2, 2);
plot_effective_connectivity(ax2, binary_fc_otsu_pc1, group_labels);
title('Functional Connectivity PC1', 'FontSize', 14);

% --- Function to generate the effective connectivity plot ---
function plot_effective_connectivity(ax, binary_otsu, group_labels)
    % Get unique groups in stable order
    [unique_groups, ~, group_idx] = unique(group_labels, 'stable');
    num_groups = length(unique_groups);

    % Initialize normalized matrix
    normalized_matrix = zeros(num_groups);

    % Normalize each group-to-group connection
    for i = 1:num_groups
        for j = 1:num_groups
            source_idx = find(group_idx == i);  % outgoing
            target_idx = find(group_idx == j);  % incoming

            total_weight = sum(sum(binary_otsu(source_idx, target_idx)));
            possible_connections = length(source_idx) * length(target_idx);

            if possible_connections > 0
                normalized_matrix(i, j) = total_weight / possible_connections;
            end
        end
    end

    % Flip the order of outgoing labels and corresponding matrix rows
    flipped_order = num_groups:-1:1;
    normalized_matrix = normalized_matrix(flipped_order, :);

    % Update labels
    labels_out = strcat(unique_groups(flipped_order), ' (Outgoing)');
    labels_in  = strcat(unique_groups,              ' (Incoming)');
    all_labels = [labels_out; labels_in];

    % Create extended matrix
    extended_matrix = zeros(2 * num_groups);
    extended_matrix(1:num_groups, num_groups+1:end) = normalized_matrix;

    % Set the current axes before calling circularGraph
    axes(ax);  % Ensure drawing happens in the right subplot
    circularGraph(extended_matrix, 'Label', all_labels);
end
