clear; close all; %clc; 

% -------------------------------------------------------------------------
%
% Purpose:
%   This script constructs a master table for baseline (sc) features,
%   collecting selected variables from multiple *_cleaned_sc.csv files.
%   Each file contains exactly one value per SCRNO (patient ID).
%
% Important:
%   - Files used and corresponding extracted columns:
%       * CDR_cleaned_sc.csv        -> CDSOB (note: -1 values replaced with NaN)
%       * GDSCALE_cleaned_sc.csv    -> GDTOTAL
%       * MMSE_cleaned_sc.csv       -> MMSCORE
%       * CAPSLIFE_cleaned.csv      -> CAPSSCORE (renamed CAPSSCORELIFE)
%       * CAPSCURR_cleaned.csv      -> CAPSSCORE (renamed CAPSSCORECURR)
%       * NEUROBAT_cleaned.csv      -> ANARTERR
%       * CES_cleaned.csv           -> CES1A, CES2, CES3A, CES4, CES5, CES6, CES7
%       * sub_diagnosis_n88.csv     -> group_1_4__4_NC__3_PTSD_TBI__2_PTSD__1_TBI_
%                                    (renamed PATIENTGROUP, matched to SCRNO via 'sub' column)
%       * basic_imaging_information_for_every_subject.csv -> Age, Weight
%                                    (matched to SCRNO via valid SubjectIDs)
%
%   - All files are expected in the 'Neuropsychological Useful data' folder,
%     except sub_diagnosis_n88.csv, which must be manually placed in the 'utilities' folder,
%     and basic_imaging_information_for_every_subject.csv located in 'find eligible ids' folder.
%   - The reference list of unique patient IDs comes from unique_patient_ids.csv.
%   - FAQ_cleaned_bl.csv was not used due to excessive missing data.
%
% Output:
%   - A master CSV file containing:
%     SCRNO | CDSOB | GDTOTAL | MMSCORE | CAPSSCORELIFE | CAPSSCORECURR | ANARTERR | CES1A ... CES7 | PATIENTGROUP | Age | Weight
% -------------------------------------------------------------------------

% Define paths
baseFolder      = fileparts(pwd);  % assuming script is in 'utilities'
dataFolder      = fullfile(baseFolder, 'Neuropsychological Useful data');
uniqueIDFile    = fullfile(baseFolder, 'utilities', 'unique_patient_ids.csv');
diagnosisFile = fullfile(baseFolder, 'utilities', 'sub_diagnosis_n88.csv');
basicImagingFile = fullfile(baseFolder, 'find eligible ids', 'basic_imaging_information_for_every_subject.csv');

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

% Replace -1 with NaN in CDSOB, because -1 indicates a missing value
T.CDSOB(T.CDSOB == -1) = NaN;

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

% ------------------- PATIENTGROUP from sub_diagnosis_n88.csv ----------------
fprintf('Loading PATIENTGROUP from sub_diagnosis_n88.csv...\n');
T = readtable(diagnosisFile);


% Select only relevant columns
T = T(:, {'sub', 'group_1_4_4_NC_3_PTSD_TBI_2_PTSD_1_TBI_'});

% Rename for merging
 T.Properties.VariableNames = {'SCRNO', 'PATIENTGROUP'};

% Merge with master table
master = outerjoin(master, T, 'Keys', 'SCRNO', 'MergeKeys', true);

% ------------------- Add Age and Weight from basic_imaging_information_for_every_subject.csv -------------
fprintf('Loading Age and Weight from basic imaging info...\n');
basicData = readtable(basicImagingFile);

% Extract only SubjectID, Age, Weight
basicData = basicData(:, {'SubjectID', 'Age', 'Weight'});

% Filter basicData to only those SubjectID present in master.SCRNO
isInMaster = ismember(basicData.SubjectID, master.SCRNO);
basicData = basicData(isInMaster, :);

% Remove duplicate SubjectID entries, keep the first occurrence only
[~, uniqueIdx] = unique(basicData.SubjectID, 'stable'); % 'stable' keeps original order
basicData = basicData(uniqueIdx, :);

% Rename SubjectID to SCRNO for merging
basicData.Properties.VariableNames{'SubjectID'} = 'SCRNO';

% Merge filtered Age and Weight into master table by SCRNO
master = outerjoin(master, basicData, 'Keys', 'SCRNO', 'MergeKeys', true);



% ------------------- Save Final Feature File ----------------
writetable(master, outputFile);
fprintf('Feature set master file saved to:\n  %s\n', outputFile);
