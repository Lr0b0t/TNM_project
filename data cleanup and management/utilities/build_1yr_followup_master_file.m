clc; close all; clear;

% -------------------------------------------------------------------------
%
% Purpose:
%   This script constructs a master table for 1-year follow-up (m12)
%   neuropsychological data. It collects selected outcome values (the ones
%   that we want to do regression for) from cleaned *_cleaned_m12.csv files
%   and aligns them by patient ID (SCRNO).
%
% Important:
%   - Files used:
%       * GDSCALE_cleaned_m12.csv  -> GDTOTAL
%       * MMSE_cleaned_m12.csv     -> MMSCORE
%       * CDR_cleaned_m12.csv      -> CDSOB
%   - The FAQ_cleaned_m12.csv file is no longer used due to excessive NaNs.
%     Originaly, our intention was to also us it as an outcome value.
%   - All files must contain one row per SCRNO.
%   - The unique_patient_ids.csv file must contain one column with no header.
%   - All files are expected in the main 'Neuropsychological Useful data' folder.
%
% Output:
%   - A master CSV file containing:
%       SCRNO | GDTOTAL | MMSCORE | CDSOB
% -------------------------------------------------------------------------

% Define paths
baseFolder        = fileparts(pwd);  % assuming script is inside 'utilities'
dataFolder        = fullfile(baseFolder, 'Neuropsychological Useful data');
uniqueIDFile      = fullfile(baseFolder, 'utilities', 'unique_patient_ids.csv');

% Define output folder and ensure it exists
finalFolder       = fullfile(baseFolder, 'final files');
if ~exist(finalFolder, 'dir')
    mkdir(finalFolder);
end
outputFile = fullfile(finalFolder, 'followup_m12_master.csv');

% Load list of unique patient IDs
uniqueIDs = readtable(uniqueIDFile, 'ReadVariableNames', false);
uniqueIDs = uniqueIDs.(1);  % Extract as array

% Initialize master table with SCRNO
master = table(uniqueIDs, 'VariableNames', {'SCRNO'});

% ------------------- Load GDTOTAL from GDSCALE -------------------
fprintf('Loading GDTOTAL from GDSCALE...\n');
T = readtable(fullfile(dataFolder, 'GDSCALE_cleaned_m12.csv'));
T = T(:, {'SCRNO', 'GDTOTAL'});
master = outerjoin(master, T, 'Keys', 'SCRNO', 'MergeKeys', true);

% ------------------- Load MMSCORE from MMSE ----------------------
fprintf('Loading MMSCORE from MMSE...\n');
T = readtable(fullfile(dataFolder, 'MMSE_cleaned_m12.csv'));
T = T(:, {'SCRNO', 'MMSCORE'});
master = outerjoin(master, T, 'Keys', 'SCRNO', 'MergeKeys', true);

% ------------------- Load CDSOB from CDR -------------------------
fprintf('Loading CDSOB from CDR...\n');
T = readtable(fullfile(dataFolder, 'CDR_cleaned_m12.csv'));
T = T(:, {'SCRNO', 'CDSOB'});

% Replace -1 with NaN in CDSOB, because -1 indicates a missing value
T.CDSOB(T.CDSOB == -1) = NaN;

master = outerjoin(master, T, 'Keys', 'SCRNO', 'MergeKeys', true);

% ------------------- Save Master File ----------------------------
writetable(master, outputFile);
fprintf('Follow-up master file saved to:\n  %s\n', outputFile);
