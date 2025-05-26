clc; close all; clear;

% -------------------------------------------------------------------------
% Purpose:
%   Compare two ID files in the 'utilities' folder:
%     - unique_patient_ids_old.csv (reference)
%     - eligible_patient_ids.csv   (newer)
%
%   This script identifies:
%     - New IDs present only in the new file
%     - Missing IDs that were in the old file but are no longer present
%
% Assumptions:
%   - Both CSV files are in the same format: one column, no header
%   - Script is located inside the 'utilities' folder
% -------------------------------------------------------------------------

% Define file paths
currentFolder = pwd;  % should be 'utilities'
oldFile       = fullfile(currentFolder, 'unique_patient_ids_old.csv');
newFile       = fullfile(currentFolder, 'eligible_patient_ids.csv');

% Load data from both files
if ~isfile(oldFile) || ~isfile(newFile)
    error('One or both ID files are missing in the utilities folder.');
end

oldIDs = readmatrix(oldFile, 'FileType', 'text');
newIDs = readmatrix(newFile, 'FileType', 'text');

% Ensure IDs are treated as strings (in case of leading zeros)
oldIDs = string(oldIDs);
newIDs = string(newIDs);

% Identify differences
addedIDs   = setdiff(newIDs, oldIDs);
removedIDs = setdiff(oldIDs, newIDs);

% Report differences
fprintf('Comparing patient ID files:\n');
fprintf(' - Old file: %s\n', oldFile);
fprintf(' - New file: %s\n\n', newFile);

fprintf('Total IDs in old file: %d\n', numel(oldIDs));
fprintf('Total IDs in new file: %d\n', numel(newIDs));
fprintf('New IDs added: %d\n', numel(addedIDs));
fprintf('IDs removed (missing in new): %d\n\n', numel(removedIDs));

% Print added IDs if any
if ~isempty(addedIDs)
    fprintf('IDs added:\n');
    disp(addedIDs);
end

% Print removed IDs if any
if ~isempty(removedIDs)
    fprintf('WARNING: Some IDs from the old file are missing in the new list:\n');
    disp(removedIDs);
else
    fprintf('No IDs were lost. All previous IDs are retained in the new file.\n');
end

