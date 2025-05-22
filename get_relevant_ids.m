%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCRIPT NAME: filter_and_sort_by_id.m
%
% PURPOSE:
%   This script filters multiple CSV data files located in various study-related
%   directories by keeping only the rows where the `SCRNO` column matches a set
%   of unique subject IDs provided in `unique_ids.csv`. It also sorts the
%   filtered rows in descending order by `SCRNO`, and saves the results with a
%   `_F.csv` suffix to distinguish them.
%
% USAGE:
%   - Place this script in the root directory of your project.
%   - Ensure `unique_ids.csv` (without a header, single column of SCRNO values)
%     is in the same directory as the script.
%   - Make sure all relevant folders (e.g., `Diagnosis`, `Enrollment`, 
%     subfolders like `medical history/Drugs`) exist and contain `.csv` files.
%   - Run the script. It will:
%       - Log output messages to a timestamped `.txt` file.
%       - Delete previously generated `_F.csv` files.
%       - Filter and sort new files based on `SCRNO`.
%       - Report missing IDs.
%
% CONTEXT / USE CASES:
%   - Data cleaning and curation in longitudinal or clinical studies.
%   - Preparing subject-specific data extracts for analysis or export.
%   - Verifying which subjects are present across different datasets.
%
% OUTPUT:
%   - A filtered, sorted version of each `.csv` file, saved as `filename_F.csv`.
%   - A log file named like `log_filtering_YYYYMMDD_HHMMSS.txt` capturing all actions.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;

% Start log
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
log_filename = ['log_filtering_', timestamp, '.txt'];
diary(log_filename);
diary on;

%% Load unique IDs from CSV file with no header
try
    unique_ids = readmatrix('unique_ids.csv', 'OutputType', 'string');
catch
    error('Could not read unique_ids.csv. Make sure the file exists and is properly formatted.');
end

% Trim whitespace
unique_ids = strtrim(unique_ids);

%% Define folders
main_folders = {
    'Biospecimen_Results', 'Neuropsychological', 'Diagnosis', ...
    'study_info_Data___Database', 'Enrollment', 'Subject_Characteristics', ...
    'Telephone_Pre-Screen_Assessments'
};

% Add subfolders from 'medical history'
medical_subfolders = {'Adverse_Events', 'Drugs', 'Medical_History', 'Physical_Neurological_Exams'};
medical_root = 'medical history';
for i = 1:numel(medical_subfolders)
    main_folders{end+1} = fullfile(medical_root, medical_subfolders{i});
end

%% Process each folder
for i = 1:length(main_folders)
    folder_path = main_folders{i};

    % Delete old filtered files ending in _F.csv
    old_filtered = dir(fullfile(folder_path, '*_F.csv'));
    for k = 1:length(old_filtered)
        delete(fullfile(folder_path, old_filtered(k).name));
        fprintf('Deleted old filtered file: %s\n', fullfile(folder_path, old_filtered(k).name));
    end

    % Get all CSV files
    files = dir(fullfile(folder_path, '*.csv'));
    for j = 1:length(files)
        % Skip already filtered files
        if endsWith(files(j).name, '_F.csv')
            continue;
        end

        file_path = fullfile(folder_path, files(j).name);

        % Read CSV into table
        try
            T = readtable(file_path);
        catch
            fprintf('Could not read file: %s\n', file_path);
            continue;
        end

        % Check for SCRNO column
        if ~ismember('SCRNO', T.Properties.VariableNames)
            fprintf('"SCRNO" column not found in file: %s\n', file_path);
            continue;
        end

        % Convert SCRNO to string and trim
        file_ids = strtrim(string(T.SCRNO));
        ref_ids = strtrim(unique_ids);

        % Apply filtering
        match_mask = ismember(file_ids, ref_ids);
        filtered_T = T(match_mask, :);

        % Sort by SCRNO in descending order
        try
            filtered_T.SCRNO = string(filtered_T.SCRNO);
            filtered_T = sortrows(filtered_T, 'SCRNO', 'descend');
        catch
            fprintf('Failed to sort by SCRNO in file: %s\n', file_path);
        end

        % Report filtering stats
        if height(filtered_T) == height(T)
            fprintf('Warning: No filtering occurred (all rows retained) in file: %s\n', file_path);
        elseif height(filtered_T) == 0
            fprintf('Warning: No matching IDs found in file: %s\n', file_path);
        else
            fprintf('%d of %d rows retained for file: %s\n', height(filtered_T), height(T), file_path);
        end

        % Check for missing IDs
        found_ids = unique(filtered_T.SCRNO);
        missing_ids = setdiff(unique_ids, found_ids);
        if ~isempty(missing_ids)
            fprintf('%d ID(s) missing in file: %s\n', length(missing_ids), file_path);
            fprintf('Missing IDs: %s\n', strjoin(missing_ids, ', '));
        else
            fprintf('All IDs are present in file: %s\n', file_path);
        end

        % Save filtered + sorted table with _F suffix
        [~, name, ext] = fileparts(files(j).name);
        output_file = fullfile(folder_path, [name '_F' ext]);
        try
            writetable(filtered_T, output_file);
        catch
            fprintf('Failed to write filtered file: %s\n', output_file);
        end
        fprintf('======================= NEXT FILE ========================== \n');
    end

    
end

fprintf('Processing complete.\n');

%%
diary off
