clear; close all; %clc; 

% -------------------------------------------------------------------------
%
% Purpose:
%   For downstream modeling, this script merges APOE genotype information 
%   (APGEN1 and APGEN2 from APOERES_cleaned.csv) into the imputed feature 
%   files for both training and testing sets.
%
%   The APOE columns are inserted immediately after 'Weight' and before 
%   'GDTOTAL_followUp' in each feature file.
%
%   The input feature files are:
%     - train_features_Q2_imputed.csv
%     - test_features_Q2_imputed.csv
%   The output files are named accordingly with Q3:
%     - train_features_Q3_imputed.csv
%     - test_features_Q3_imputed.csv
%
% Key constraints:
%   - Only rows where SCRNO matches between feature files and APOERES_cleaned.csv
%     will receive APGEN1 and APGEN2; others are set to NaN.
%   - All rows with a NaN value in APGEN1 or APGEN2 are removed from output.
%   - The script does not change the order or number of other columns.
%
% Inputs:
%   - 'APOERES_cleaned.csv' from 'Biospecimen Useful data'
%   - 'train_features_Q2_imputed.csv' and 'test_features_Q2_imputed.csv'
%     from 'final files'
%
% Outputs:
%   - 'train_features_Q3_imputed.csv' and 'test_features_Q3_imputed.csv'
%     with APGEN1 and APGEN2 columns appended and no missing APOE data.
%
% Usage:
%   - Run this script in the 'utilities' directory.
%
% -------------------------------------------------------------------------

% --- Paths ---
baseFolder = fileparts(pwd);  % assumes script is in 'utilities'
finalFolder = fullfile(baseFolder, 'final files');
biospecimenFolder = fullfile(baseFolder, 'Biospecimen Useful data');
apoeFile = fullfile(biospecimenFolder, 'APOERES_cleaned.csv');

% Load APOERES_cleaned.csv
apoe = readtable(apoeFile);

% Only keep SCRNO, APGEN1, APGEN2 columns
apoe = apoe(:, {'SCRNO', 'APGEN1', 'APGEN2'});

% Helper function to append APOE data and save new file
function appendAPOE(inputFile, outputFile, apoe)
    data = readtable(inputFile);

    % Match APGEN1/APGEN2 by SCRNO
    [found, idxAPOE] = ismember(data.SCRNO, apoe.SCRNO);
    apgen1 = nan(height(data), 1);
    apgen2 = nan(height(data), 1);

    apgen1(found) = apoe.APGEN1(idxAPOE(found));
    apgen2(found) = apoe.APGEN2(idxAPOE(found));

    % Insert APGEN1 and APGEN2 after Weight, before GDTOTAL_followUp
    colNames = data.Properties.VariableNames;
    idxWeight = find(strcmp(colNames, 'Weight'));
    idxGDTOTAL = find(strcmp(colNames, 'GDTOTAL_followUp'));

    % Defensive: if for some reason GDTOTAL_followUp is not found, add at end
    if isempty(idxGDTOTAL)
        idxGDTOTAL = length(colNames) + 1;
    end

    % Build new table with correct column order
    T_left = data(:, 1:idxWeight);
    T_apoe = table(apgen1, apgen2, 'VariableNames', {'APGEN1', 'APGEN2'});
    T_mid = data(:, (idxWeight+1):(idxGDTOTAL-1));
    T_right = data(:, idxGDTOTAL:end);

    newData = [T_left T_apoe T_mid T_right];

    % --- Remove any rows with NaN in either APGEN1 or APGEN2
    nanRows = isnan(newData.APGEN1) | isnan(newData.APGEN2);
    if any(nanRows)
        fprintf('  Removing %d rows with missing APOE genotype from %s\n', sum(nanRows), inputFile);
    end
    newData(nanRows, :) = [];

    % Save to new file
    writetable(newData, outputFile);
    fprintf('Saved updated file: %s (%d rows, no NaNs in APOE columns)\n', outputFile, height(newData));
end

% Process train file
trainIn = fullfile(finalFolder, 'train_features_Q2_imputed.csv');
trainOut = fullfile(finalFolder, 'train_features_Q3_imputed.csv');
appendAPOE(trainIn, trainOut, apoe);

% Process test file
testIn = fullfile(finalFolder, 'test_features_Q2_imputed.csv');
testOut = fullfile(finalFolder, 'test_features_Q3_imputed.csv');
appendAPOE(testIn, testOut, apoe);
