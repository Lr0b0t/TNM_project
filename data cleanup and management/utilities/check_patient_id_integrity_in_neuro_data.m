clear; close all; %clc; 

% -------------------------------------------------------------------------
%
% Purpose:
%   This script performs validation and automatic correction of all cleaned
%   neuropsychological data files by ensuring that:
%     - Each patient ID (SCRNO) appears exactly once per file
%     - No patient IDs from the reference list (unique_patient_ids.csv)
%       are missing in any of the files
%     - All duplicate rows are printed before being fixed
%
% Fixes applied:
%   - For duplicate IDs: keeps only the last occurrence
%   - For missing IDs: adds rows with SCRNO set and NaN in all other fields
%
% Important:
%   - The unique_patient_ids.csv must contain a single column with no header
%   - The script processes only files in the main data folder,
%     excluding any files inside the 'archived_cleaned_files' folder
%   - All files must contain a 'SCRNO' column
%
% -------------------------------------------------------------------------

% Define paths
baseFolder     = fileparts(pwd);  % assuming script is in 'utilities'
dataFolder     = fullfile(baseFolder, 'Neuropsychological Useful data');
uniqueIDFile   = fullfile(baseFolder, 'utilities', 'unique_patient_ids.csv');

% Load unique patient IDs
uniqueIDs = readtable(uniqueIDFile, 'ReadVariableNames', false);
uniqueIDs = uniqueIDs.(1);  % Extract as array

% List all .csv files (excluding archived subfolder)
csvFiles = dir(fullfile(dataFolder, '*.csv'));
csvFiles = csvFiles(~contains({csvFiles.name}, 'archived_cleaned_files'));

% Process each file
for k = 1:length(csvFiles)
    fileName = csvFiles(k).name;
    filePath = fullfile(dataFolder, fileName);

    fprintf('\nChecking file: %s\n', fileName);
    
    % Read table
    T = readtable(filePath);
    
    if ~ismember('SCRNO', T.Properties.VariableNames)
        warning('  File "%s" does not contain a SCRNO column. Skipping...\n', fileName);
        continue;
    end

    % ------------------ FIX DUPLICATES ------------------
    [~, lastIdx] = unique(T.SCRNO, 'last');
    duplicateRows = setdiff(1:height(T), lastIdx);

    if ~isempty(duplicateRows)
        fprintf('  Warning: Found %d duplicate SCRNO entries. Keeping last occurrence.\n', ...
                numel(duplicateRows));

        dupIDs = T.SCRNO(duplicateRows);
        for i = 1:numel(dupIDs)
            fprintf('    Duplicate SCRNO: %s\n', string(dupIDs(i)));
            disp(T(T.SCRNO == dupIDs(i), :));
        end

        T = T(lastIdx, :);  % Keep only last occurrence of each SCRNO
    end

    % ------------------ FIX MISSING IDs ------------------
    missingIDs = setdiff(uniqueIDs, T.SCRNO);

    if ~isempty(missingIDs)
        fprintf('  Warning: %d missing SCRNOs. Adding rows with the missing IDs...\n', numel(missingIDs));
        fprintf('    Missing SCRNOs:\n');
        disp(missingIDs);

        % Add each missing ID with empty fields
        for i = 1:numel(missingIDs)
            missingID = missingIDs(i);
            newRow = table();
            newRow.SCRNO = missingID;
            
            % Initialize other variables appropriately
            for varName = T.Properties.VariableNames
                if strcmp(varName{1}, 'SCRNO'), continue; end
                varClass = class(T.(varName{1}));
                switch varClass
                    case 'double'
                        newRow.(varName{1}) = NaN;
                    case 'cell'
                        newRow.(varName{1}) = {''};
                    case 'datetime'
                        newRow.(varName{1}) = NaT;
                    otherwise
                        newRow.(varName{1}) = missing;
                end
            end
            
            T = [T; newRow]; % Append new row
        end

        % % Create table with NaNs for missing IDs
        % missingTable = cell2table(cell(numel(missingIDs), width(T)), ...
        %                           'VariableNames', T.Properties.VariableNames);
        % missingTable.SCRNO = missingIDs;
        % 
        % T = [T; missingTable];
    end
    
    if (~isempty(missingIDs) || ~isempty(duplicateRows))
        % ------------------ RECHECK AFTER FIXES ------------------
        fprintf('  Rechecking file after corrections.\n');
    
        % Check for duplicates again
        [~, ic] = unique(T.SCRNO, 'stable');
        if length(ic) ~= height(T)
            error(' Watch out! Duplicates still present after cleanup!');
        end
    
        % Check if any of the expected IDs are still missing
        stillMissing = setdiff(uniqueIDs, T.SCRNO);
        if ~isempty(stillMissing)
            error('  Watch out! Some IDs are still missing after insertion!');
        end
    
        % Save fixed file back
        writetable(T, filePath);
        fprintf('  File "%s" has been fixed and saved.\n', fileName);
    end

end

fprintf('\nAll files checked and corrected.\n');
























% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% clc; close all; clear;
% 
% % -------------------------------------------------------------------------
% %
% % Purpose:
% %   This script performs validation checks on all cleaned and split
% %   neuropsychological data files to ensure that:
% %     - Each patient ID (SCRNO) appears exactly once per file
% %     - No patient IDs from the reference list (unique_patient_ids.csv)
% %       are missing in any of the files
% %     - All duplicate rows are printed if any exist
% %
% % Important:
% %   - The unique_patient_ids.csv must contain a single column with no header
% %   - The script only processes files in the main data folder, and ignores
% %     any files inside the 'archived_cleaned_files' folder
% %   - All data files must contain a 'SCRNO' column
% %
% % Usage:
% %   To be run from the 'utilities' folder.
% % -------------------------------------------------------------------------
% 
% % Define paths
% baseFolder     = fileparts(pwd);  % assuming script is in 'utilities'
% dataFolder     = fullfile(baseFolder, 'Neuropsychological Useful data');
% uniqueIDFile   = fullfile(baseFolder, 'utilities', 'unique_patient_ids.csv');
% 
% % Load unique patient IDs
% uniqueIDs = readtable(uniqueIDFile, 'ReadVariableNames', false);
% uniqueIDs = uniqueIDs.(1);  % Extract as array
% expectedIDCount = numel(uniqueIDs);
% 
% % List all .csv files (excluding archived subfolder)
% csvFiles = dir(fullfile(dataFolder, '*.csv'));
% csvFiles = csvFiles(~contains({csvFiles.name}, 'archived_cleaned_files'));
% 
% % Process each file
% for k = 1:length(csvFiles)
%     fileName = csvFiles(k).name;
%     filePath = fullfile(dataFolder, fileName);
% 
%     fprintf('Checking file: %s\n', fileName);
% 
%     % Read table
%     T = readtable(filePath);
% 
%     if ~ismember('SCRNO', T.Properties.VariableNames)
%         warning('File "%s" does not contain a SCRNO column.', fileName);
%         continue;
%     end
% 
%     % Extract IDs
%     fileIDs = T.SCRNO;
% 
%     % Check for duplicate IDs
%     [uniqueInFile, ~, ic] = unique(fileIDs);
%     counts = accumarray(ic, 1);
%     hasDuplicates = any(counts > 1);
% 
%     if hasDuplicates
%         dupIDs = uniqueInFile(counts > 1);
%         fprintf('  Warning: Duplicate SCRNOs found in %s: %d duplicates\n', ...
%                 fileName, numel(dupIDs));
% 
%         for i = 1:numel(dupIDs)
%             dupRows = T(fileIDs == dupIDs(i), :);
%             fprintf('    Duplicate rows for SCRNO %s:\n', string(dupIDs(i)));
%             disp(dupRows);
%         end
%     end
% 
%     % Check for missing IDs
%     missingIDs = setdiff(uniqueIDs, fileIDs);
%     if ~isempty(missingIDs)
%         fprintf('  Warning: Missing SCRNOs in %s: %d IDs missing\n', ...
%                 fileName, numel(missingIDs));
%         fprintf('    Missing SCRNOs:\n');
%         disp(missingIDs);
%     end
% 
%     % Check for unexpected extra IDs
%     extraIDs = setdiff(fileIDs, uniqueIDs);
%     if ~isempty(extraIDs)
%         fprintf('  Note: File %s contains %d unexpected SCRNOs not in unique ID list\n', ...
%                 fileName, numel(extraIDs));
%     end
% end
% 
% fprintf('ID integrity checks completed.\n');
