clc; close all; clear;

% -------------------------------------------------------------------------
% 
%   This script prepares neuropsychological assessment data for regression
%   workflows. It splits each cleaned file into two files based 
%   on the VISCODE value — e.g., 'sc', 'bl' (baseline) vs 'm12' (follow-up).
%
%   After splitting, the original cleaned files are archived, and the
%   resulting split files are left in the working directory to be used
%   as ML input features and targets.
%
% Inputs:
%   - CDR_cleaned.csv
%   - FAQ_cleaned.csv
%   - MMSE_cleaned.csv
%   - GDSCALE_cleaned.csv
%
% Output:
%   - Split versions of the input files by VISCODE (e.g., *_sc.csv, *_m12.csv)
%   - Original cleaned files moved to 'archived_cleaned_files' subfolder
%
% Expected Folder Structure After Execution:
%
% Neuropsychological Useful data/
% ├── CDR_cleaned_sc.csv
% ├── CDR_cleaned_m12.csv
% ├── FAQ_cleaned_bl.csv
% ├── FAQ_cleaned_m12.csv
% ├── MMSE_cleaned_sc.csv
% ├── MMSE_cleaned_m12.csv
% ├── GDSCALE_cleaned_sc.csv
% ├── GDSCALE_cleaned_m12.csv
% │
% └── archived_cleaned_files/
%     ├── CDR_cleaned.csv
%     ├── FAQ_cleaned.csv
%     ├── MMSE_cleaned.csv
%     ├── GDSCALE_cleaned.csv
% 
% -------------------------------------------------------------------------

% Define base paths
baseFolder        = fileparts(pwd);  % Assumes script lives in 'utilities'
dataFolder        = fullfile(baseFolder, 'Neuropsychological Useful data');
archiveFolder     = fullfile(dataFolder, 'archived_cleaned_files');

% Create archive folder if it doesn't exist
if ~exist(archiveFolder, 'dir')
    mkdir(archiveFolder);
end

% File list and their corresponding VISCODE splits
fileList = {
    'CDR_cleaned.csv',     {'sc', 'm12'};
    'FAQ_cleaned.csv',     {'bl', 'm12'};
    'MMSE_cleaned.csv',    {'sc', 'm12'};
    'GDSCALE_cleaned.csv', {'sc', 'm12'};
};

% Loop through files and process
for i = 1:size(fileList, 1)
    fileName = fileList{i, 1};
    viscodes = fileList{i, 2};
    inputFile = fullfile(dataFolder, fileName);

    % Load the file
    if exist(inputFile, 'file')
        T = readtable(inputFile);

        % Write split files directly in dataFolder
        for j = 1:length(viscodes)
            vc = viscodes{j};
            splitData = T(strcmpi(T.VISCODE, vc), :);
            outputName = fullfile(dataFolder, replace(fileName, '.csv', ['_' vc '.csv']));
            writetable(splitData, outputName);
        end

        % Move original cleaned file to archive
        movefile(inputFile, fullfile(archiveFolder, fileName));
    else
        warning('File not found: %s', fileName);
    end
end