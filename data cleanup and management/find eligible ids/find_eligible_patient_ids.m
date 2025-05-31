clc; close all; clear;

% -------------------------------------------------------------------------
%
% Purpose:
%   This script identifies eligible subject IDs that:
%   - Exist in 'basic_imaging_information_for_every_subject.csv'
%   - Have both 'sc' and 'm12' entries in the VISCODE column
%     of ALL of the following files:
%       - GDSCALE.csv
%       - MMSE.csv
%       - CDR.csv
%   - Excludes specific SCRNOs due to poor imaging quality:
%       * 168947, 196043, 228333
%
% Requirements:
%   - All files should be placed according to the described structure.
%   - Rows with VISCODE 'tau' or 'tau2' are ignored.
%   - Output: Number of eligible subjects and their IDs.
%   - Requires that the file
%     'basic_imaging_information_for_every_subject.csv' is already manually
%     placed inside this folder, as it has to be downoladed from the
%     database. This file was the list given by the database's website, and
%     has the relevant data that accompany the imaging data that we
%     downloaded.
%
% Directory Assumptions:
%   - Script is inside "find eligible ids" folder.
%   - Neuropsychological CSVs are in ../study data as downloaded/Neuropsychological/
%
% Output:
%   - Save the list of our study IDs to a CSV file (no headers) in the utilities
%    folder under the name 'unique_patient_ids.csv'
% -------------------------------------------------------------------------

% Define folders
currentFolder = pwd;
baseFolder = fileparts(currentFolder);
neuroFolder = fullfile(baseFolder, 'study data as downloaded', 'Neuropsychological');
mainInputFile = fullfile(currentFolder, 'basic_imaging_information_for_every_subject.csv');

% Load main list of unique IDs
fprintf('Loading main input file...\n');
mainData = readtable(mainInputFile);
if any(strcmp('SubjectID', mainData.Properties.VariableNames))
    allIDs = unique(mainData.SubjectID);
else
    error('The file must contain a column named "SubjectID".');
end

% Function to get eligible IDs from a neuropsychological file
getEligibleIDs = @(fileName) getIDsWithSCandM12(fullfile(neuroFolder, fileName));

% Process each file
fprintf('Processing GDSCALE.csv...\n');
idsGDS = getEligibleIDs('GDSCALE.csv');

fprintf('Processing MMSE.csv...\n');
idsMMSE = getEligibleIDs('MMSE.csv');

fprintf('Processing CDR.csv...\n');
idsCDR = getEligibleIDs('CDR.csv');

% Intersect across all sets and the initial ID list
eligibleIDs = intersect(intersect(idsGDS, idsMMSE), idsCDR);
finalEligibleIDs = intersect(eligibleIDs, allIDs);

% Exclude bad-quality subjects
excludedIDs = [168947; 196043; 228333];
finalEligibleIDs = setdiff(finalEligibleIDs, excludedIDs);

% Display results
fprintf('\nTotal eligible IDs found (after exclusion): %d\n', numel(finalEligibleIDs));
disp('List of eligible SCRNOs:');
disp(finalEligibleIDs);

% Save eligible IDs to a CSV file (no headers) in the utilities folder
utilitiesFolder = fullfile(baseFolder, 'utilities');
if ~exist(utilitiesFolder, 'dir')
    error('Utilities folder does not exist: %s', utilitiesFolder);
end

outputFile = fullfile(utilitiesFolder, 'unique_patient_ids.csv');
writematrix(finalEligibleIDs, outputFile, 'FileType', 'text');

fprintf('Eligible IDs saved to: %s\n', outputFile);


function ids = getIDsWithSCandM12(filePath)
    T = readtable(filePath);

    % Ensure SCRNO and VISCODE columns exist
    if ~all(ismember({'SCRNO', 'VISCODE'}, T.Properties.VariableNames))
        error('File %s must contain SCRNO and VISCODE columns.', filePath);
    end

    % Remove rows with VISCODE 'tau' or 'tau2'
    viscode = lower(strtrim(T.VISCODE));
    isValid = ~(strcmp(viscode, 'tau') | strcmp(viscode, 'tau2'));
    T = T(isValid, :);

    % Group by SCRNO
    ids = unique(T.SCRNO);
    idsWithSC  = unique(T.SCRNO(strcmpi(T.VISCODE, 'sc')));
    idsWithM12 = unique(T.SCRNO(strcmpi(T.VISCODE, 'm12')));

    ids = intersect(idsWithSC, idsWithM12);
end
