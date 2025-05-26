clc; close all; clear;

% -------------------------------------------------------------------------
%
% Purpose:
%   This script constructs a master table for baseline (sc) features,
%   collecting selected variables from multiple *_cleaned_sc.csv files.
%   Each file contains exactly one value per SCRNO (patient ID).
%
% Important:
%   - Files used and corresponding extracted columns:
%       * CDR_cleaned_sc.csv        -> CDSOB
%       * GDSCALE_cleaned_sc.csv    -> GDTOTAL
%       * MMSE_cleaned_sc.csv       -> MMSCORE
%       * CAPSLIFE_cleaned.csv      -> CAPSSCORE (renamed CAPSSCORELIFE)
%       * CAPSCURR_cleaned.csv      -> CAPSSCORE (renamed CAPSSCORECURR)
%       * NEUROBAT_cleaned.csv      -> ANARTERR
%       * CES_cleaned.csv           -> CES1A, CES2, CES3A, CES4, CES5, CES6, CES7
%
%   - All files are expected in the 'Neuropsychological Useful data' folder.
%   - The reference list of unique patient IDs comes from unique_patient_ids.csv.
%   - FAQ_cleaned_bl.csv was not used due to excessive missing data.
%
% Output:
%   - A master CSV file containing:
%     SCRNO | CDSOB | GDTOTAL | MMSCORE | CAPSSCORELIFE | CAPSSCORECURR | ANARTERR | CES1A ... CES7
% -------------------------------------------------------------------------

% Define paths
baseFolder      = fileparts(pwd);  % assuming script is in 'utilities'
dataFolder      = fullfile(baseFolder, 'Neuropsychological Useful data');
uniqueIDFile    = fullfile(baseFolder, 'utilities', 'unique_patient_ids.csv');

% Output path
finalFolder     = fullfile(baseFolder, 'final files');
if ~exist(finalFolder, 'dir')
    mkdir(finalFolder);
end
outputFile      = fullfile(finalFolder, 'feature_set_master.csv');

% Load list of unique patient IDs
uniqueIDs = readtable(uniqueIDFile, 'ReadVariableNames', false);
uniqueIDs = uniqueIDs.(1);  % Extract as array

% Initialize master table with SCRNO
master = table(uniqueIDs, 'VariableNames', {'SCRNO'});

% ------------------- CDSOB from CDR -------------------------
fprintf('Loading CDSOB from CDR...\n');
T = readtable(fullfile(dataFolder, 'CDR_cleaned_sc.csv'));
T = T(:, {'SCRNO', 'CDSOB'});
master = outerjoin(master, T, 'Keys', 'SCRNO', 'MergeKeys', true);

% ------------------- GDTOTAL from GDSCALE -------------------
fprintf('Loading GDTOTAL from GDSCALE...\n');
T = readtable(fullfile(dataFolder, 'GDSCALE_cleaned_sc.csv'));
T = T(:, {'SCRNO', 'GDTOTAL'});
master = outerjoin(master, T, 'Keys', 'SCRNO', 'MergeKeys', true);

% ------------------- MMSCORE from MMSE ----------------------
fprintf('Loading MMSCORE from MMSE...\n');
T = readtable(fullfile(dataFolder, 'MMSE_cleaned_sc.csv'));
T = T(:, {'SCRNO', 'MMSCORE'});
master = outerjoin(master, T, 'Keys', 'SCRNO', 'MergeKeys', true);

% ------------------- CAPSSCORELIFE from CAPSLIFE ------------
fprintf('Loading CAPSSCORELIFE from CAPSLIFE...\n');
T = readtable(fullfile(dataFolder, 'CAPSLIFE_cleaned.csv'));
T = T(:, {'SCRNO', 'CAPSSCORE'});
T.Properties.VariableNames{'CAPSSCORE'} = 'CAPSSCORELIFE';
master = outerjoin(master, T, 'Keys', 'SCRNO', 'MergeKeys', true);

% ------------------- CAPSSCORECURR from CAPSCURR ------------
fprintf('Loading CAPSSCORECURR from CAPSCURR...\n');
T = readtable(fullfile(dataFolder, 'CAPSCURR_cleaned.csv'));
T = T(:, {'SCRNO', 'CAPSSCORE'});
T.Properties.VariableNames{'CAPSSCORE'} = 'CAPSSCORECURR';
master = outerjoin(master, T, 'Keys', 'SCRNO', 'MergeKeys', true);

% ------------------- ANARTERR from NEUROBAT -----------------
fprintf('Loading ANARTERR from NEUROBAT...\n');
T = readtable(fullfile(dataFolder, 'NEUROBAT_cleaned.csv'));
T = T(:, {'SCRNO', 'ANARTERR'});
master = outerjoin(master, T, 'Keys', 'SCRNO', 'MergeKeys', true);

% ------------------- CES Scores -----------------------------
fprintf('Loading CES scores from CES...\n');
T = readtable(fullfile(dataFolder, 'CES_cleaned.csv'));
cesCols = {'SCRNO', 'CES1A', 'CES2', 'CES3A', 'CES4', 'CES5', 'CES6', 'CES7'};
T = T(:, cesCols);
master = outerjoin(master, T, 'Keys', 'SCRNO', 'MergeKeys', true);

% ------------------- Save Final Feature File ----------------
writetable(master, outputFile);
fprintf('Feature set master file saved to:\n  %s\n', outputFile);
