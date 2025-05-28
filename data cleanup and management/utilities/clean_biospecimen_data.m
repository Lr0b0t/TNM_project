clc; close all; clear;

% folder paths
baseFolder = fileparts(pwd);  % assuming script is in 'utilities'
inputFolder = fullfile(baseFolder, 'study data as downloaded', 'Biospecimen_Results');
outputFolder = fullfile(baseFolder, 'Biospecimen Useful data');
idFile = fullfile(pwd, 'unique_patient_ids.csv');

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Load unique patient IDs (no header in file)
uniqueIDs = readtable(idFile, 'ReadVariableNames', false);
uniqueIDs = uniqueIDs{:,1};  % extract as array

%% APOERES.csv (full name: ApoE Genotyping - Results) - keep SCRNO, APGEN1, APGEN2
file1 = fullfile(inputFolder, 'APOERES.csv');
T1 = readtable(file1);

requiredCols = {'SCRNO', 'APGEN1', 'APGEN2'};
T1 = T1(:, requiredCols);

% Filter by unique IDs
T1 = T1(ismember(T1.SCRNO, uniqueIDs), :);

% Save cleaned file
writetable(T1, fullfile(outputFolder, 'APOERES_cleaned.csv'));


%% 2) UPENNBIOMK_DOD_2017.csv (full name: UPENN ADNI-DOD CSF Elecsys results) 
%  keep SCRNO, ABETA, AB40, TAU, PTAU, AB4240 NOTE
file2 = fullfile(inputFolder, 'UPENNBIOMK_DOD_2017.csv');
T2 = readtable(file2);

cols2 = {'SCRNO', 'ABETA', 'AB40', 'TAU', 'PTAU', 'AB4240', 'NOTE'};
T2 = T2(:, cols2);

% Filter by unique IDs
T2 = T2(ismember(T2.SCRNO, uniqueIDs), :);

% Save cleaned file
writetable(T2, fullfile(outputFolder, 'UPENNBIOMK_DOD_2017_cleaned.csv'));


%% 3) UPENN_MSMS_ABETA.csv (full name: UPENN CSF Abeta42, Abeta40 and Abeta38 by LC-MS/MS)
% keep SCRNO, ABETA42, ABETA40, ABETA38
file3 = fullfile(inputFolder, 'UPENN_MSMS_ABETA.csv');
T3 = readtable(file3);

cols3 = {'SCRNO', 'ABETA42', 'ABETA40', 'ABETA38'};
T3 = T3(:, cols3);

% Filter by unique IDs
T3 = T3(ismember(T3.SCRNO, uniqueIDs), :);

% Save cleaned file
writetable(T3, fullfile(outputFolder, 'UPENN_MSMS_ABETA_cleaned.csv'));


%% 4) UPENNBIOMK.csv (full name: UPENN CSF Biomarkers)
%  keep SCRNO, TAU, ABETA, PTAU, TAU_RAW, ABETA_RAW, PTAU_RAW
file4 = fullfile(inputFolder, 'UPENNBIOMK.csv');
T4 = readtable(file4);

cols4 = {'SCRNO', 'TAU', 'ABETA', 'PTAU', 'TAU_RAW', 'ABETA_RAW', 'PTAU_RAW'};
T4 = T4(:, cols4);

% Filter by unique IDs
T4 = T4(ismember(T4.SCRNO, uniqueIDs), :);

% Save cleaned file
writetable(T4, fullfile(outputFolder, 'UPENNBIOMK_cleaned.csv'));
