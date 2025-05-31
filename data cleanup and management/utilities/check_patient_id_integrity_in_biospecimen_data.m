clear; close all; %clc; 

% -------------------------------------------------------------------------
%
% Purpose:
%   This script performs validation checks on all cleaned biospecimen data files to check if:
%     - Each patient ID (SCRNO) appears exactly once per file
%     - No patient IDs from the reference list (unique_patient_ids.csv)
%       are missing in any of the files
%     - All duplicate rows are printed if any exist
%
% Important:
%   - The unique_patient_ids.csv must contain a single column with no header
%   - The script only processes files in the biospecimen useful data folder
%   - All data files must contain a 'SCRNO' column
%
% Usage:
%   To be run from the 'utilities' folder.
% 
% Comments on the printed outcomes:
% Based on the outcome we get when we run this script, it is clear that 
% more than half of the IDs (all=88) are missing in three of the four
% files. Hence, only the data from the APOE results file can be used for
% addressing CQ3.

% -------------------------------------------------------------------------

% Define paths
baseFolder     = fileparts(pwd);  % assuming script is in 'utilities'
dataFolder     = fullfile(baseFolder, 'Biospecimen Useful data');
uniqueIDFile   = fullfile(baseFolder, 'utilities', 'unique_patient_ids.csv');

% Load unique patient IDs
uniqueIDs = readtable(uniqueIDFile, 'ReadVariableNames', false);
uniqueIDs = uniqueIDs.(1);  % Extract as array
expectedIDCount = numel(uniqueIDs);

% List all .csv files in biospecimen folder
csvFiles = dir(fullfile(dataFolder, '*.csv'));

% Process each file
for k = 1:length(csvFiles)
    fileName = csvFiles(k).name;
    filePath = fullfile(dataFolder, fileName);

    fprintf('Checking file: %s\n', fileName);

    % Read table
    T = readtable(filePath);

    if ~ismember('SCRNO', T.Properties.VariableNames)
        warning('File "%s" does not contain a SCRNO column.', fileName);
        continue;
    end

    % Extract IDs
    fileIDs = T.SCRNO;

    % Check for duplicate IDs
    [uniqueInFile, ~, ic] = unique(fileIDs);
    counts = accumarray(ic, 1);
    hasDuplicates = any(counts > 1);

    if hasDuplicates
        dupIDs = uniqueInFile(counts > 1);
        fprintf('  Warning: Duplicate SCRNOs found in %s: %d duplicates\n', ...
                fileName, numel(dupIDs));

        for i = 1:numel(dupIDs)
            dupRows = T(fileIDs == dupIDs(i), :);
            fprintf('    Duplicate rows for SCRNO %s:\n', string(dupIDs(i)));
            disp(dupRows);
        end
    end

    % Check for missing IDs
    missingIDs = setdiff(uniqueIDs, fileIDs);
    if ~isempty(missingIDs)
        fprintf('  Warning: Missing SCRNOs in %s: %d IDs missing\n', ...
                fileName, numel(missingIDs));
        fprintf('    Missing SCRNOs:\n');
        disp(missingIDs);
    end

    % Check for unexpected extra IDs
    extraIDs = setdiff(fileIDs, uniqueIDs);
    if ~isempty(extraIDs)
        fprintf('  Note: File %s contains %d unexpected SCRNOs not in unique ID list\n', ...
                fileName, numel(extraIDs));
    end
end

fprintf('Biospecimen ID integrity checks completed.\n');
