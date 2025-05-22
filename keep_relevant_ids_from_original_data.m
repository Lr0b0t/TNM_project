clc; clear; close all;

% Load unique IDs from CSV file 
try
    unique_ids = readmatrix('unique_ids.csv', 'OutputType', 'string');
    unique_ids = strtrim(unique_ids);
catch
    error('Could not read unique_ids.csv. Ensure it exists and is correctly formatted.');
end

% Input file name
input_filename = 'screening_and_1_year_follow_up_all_collumns.csv';

% Read the main data table
try
    T = readtable(input_filename);
catch
    error('Could not read %s. Ensure the file exists and is a valid CSV.', input_filename);
end

% Check if 'Subject ID' column exists
if ~ismember('SubjectID', T.Properties.VariableNames)
    error('"Subject ID" column not found in file: %s', input_filename);
end

% Filter rows based on unique IDs
filtered_T = T(ismember(string(T.("SubjectID")), unique_ids), :);

% Write the filtered table to a new file with prefix 'C_'
[~, name, ext] = fileparts(input_filename);
output_filename = ['C_' name ext];

try
    writetable(filtered_T, output_filename);
    fprintf('Filtered file written to: %s\n', output_filename);
catch
    error('Failed to write the filtered file: %s', output_filename);
end



%% Filter out Screening MRI rows with all NaNs in key score columns

% File name of previous output
input_file = 'C_screening_and_1_year_follow_up_all_collumns.csv';

% Load the filtered file
try
    T = readtable(input_file);
catch
    error('Could not read file: %s', input_file);
end

% Define columns of interest
key_columns = ["MMSETotalScore", "GDSCALETotalScore", "GlobalCDR", "FAQTotalScore"];

% Check if all required columns are present
missing_cols = setdiff(key_columns, T.Properties.VariableNames);
if ~isempty(missing_cols)
    error('Missing required columns: %s', strjoin(missing_cols, ', '));
end

% Logical index for "Screening MRI" rows
is_screening_mri = strcmpi(strtrim(string(T.Visit)), 'Screening MRI');

% Logical index for rows where all key columns are NaN
all_nans = all(ismissing(T{:, key_columns}), 2);

% Rows to delete: Screening MRI with all NaNs in key columns
rows_to_delete = is_screening_mri & all_nans;

% Rows to check for inconsistency: Screening MRI with at least one non-NaN
rows_to_warn = is_screening_mri & ~all_nans;

% Print warning if applicable
if any(rows_to_warn)
    fprintf('Warning: %d "Screening MRI" rows have non-NaN values in score columns:\n', sum(rows_to_warn));
    disp(T(rows_to_warn, ["Subject ID", "Visit", key_columns]));
end

% Remove unwanted rows
T_cleaned = T(~rows_to_delete, :);

% Overwrite original file (or save with a new name if preferred)
writetable(T_cleaned, input_file);
fprintf('Filtered file saved: %s\n', input_file);


%% Remove consecutive duplicate rows with same ID and same score values

% Reload the cleaned file
input_file = 'C_screening_and_1_year_follow_up_all_collumns.csv';
try
    T = readtable(input_file);
catch
    error('Could not read file: %s', input_file);
end

% Define ID and score columns
id_col = "SubjectID";
score_cols = ["MMSETotalScore", "GDSCALETotalScore", "GlobalCDR", "FAQTotalScore"];

% Initialize logical mask to keep rows
keep_row = true(height(T), 1);

% Iterate through table rows
for i = 2:height(T)
    same_id = isequal(T{i, id_col}, T{i-1, id_col});
    same_scores = isequaln(T{i, score_cols}, T{i-1, score_cols});  % use isequaln for NaN-safe comparison

    if same_id && same_scores
        keep_row(i) = false; % mark as duplicate
    end
end

% Apply filter
T_no_duplicates = T(keep_row, :);

% Save cleaned result (overwrite or rename if needed)
writetable(T_no_duplicates, input_file);
fprintf('Removed consecutive duplicate entries with same ID and score values.\n');
