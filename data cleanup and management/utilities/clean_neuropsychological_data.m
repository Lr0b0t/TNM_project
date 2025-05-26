clc; close all; clear;

% folder paths
baseFolder = fileparts(pwd);  % We are at the 'utilities' folder, go one level up
inputFolder = fullfile(baseFolder, 'study data as downloaded', 'Neuropsychological');
outputFolder = fullfile(baseFolder, 'Neuropsychological Useful data');
idFile = fullfile(pwd, 'unique_patient_ids.csv');

% create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Load unique patient IDs (no header in file)
uniqueIDs = readtable(idFile, 'ReadVariableNames', false);
uniqueIDs = uniqueIDs{:,1};  % extract as array

%% CDR file (full name: Clinical Dementia Rating Scale (CDR))
% Load CDR.csv file
cdrFile = fullfile(inputFolder, 'CDR.csv');
cdrData = readtable(cdrFile);

% Keep only necessary columns: SCRNO, VISCODE, USERDATE, CDSOB
requiredCols = {'SCRNO', 'VISCODE', 'USERDATE', 'CDSOB'};
cdrData = cdrData(:, requiredCols);

% Keep only rows with VISCODE equal to 'sc' or 'm12'
validViscodes = ismember(lower(string(cdrData.VISCODE)), ["sc", "m12"]);
cdrData = cdrData(validViscodes, :);

% Keep only rows where SCRNO is in the list of unique IDs
isValidID = ismember(cdrData.SCRNO, uniqueIDs);
cdrData = cdrData(isValidID, :);

% Save the cleaned data to the new folder
outputFile = fullfile(outputFolder, 'CDR_cleaned.csv');
writetable(cdrData, outputFile);

%% CAPSCURR file (full name: Clinician Administered PTSD Scale (CAPS) - Current)

% Load CAPSCURR.csv file
capsFile = fullfile(inputFolder, 'CAPSCURR.csv');
capsData = readtable(capsFile);

% Keep only necessary columns: CAPSSCORE, EXAMDATE, VISCODE, SCRNO
requiredCols = {'SCRNO', 'VISCODE', 'EXAMDATE', 'CAPSSCORE'};
capsData = capsData(:, requiredCols);

% Keep only rows where SCRNO is in the list of unique IDs
isValidID = ismember(capsData.SCRNO, uniqueIDs);
capsData = capsData(isValidID, :);

% Save the cleaned data to the new folder
outputFile = fullfile(outputFolder, 'CAPSCURR_cleaned.csv');
writetable(capsData, outputFile);


%% CAPSLIFE file (full name: Clinician Administered PTSD Scale (CAPS) - Lifetime)

% Load CAPSLIFE.csv file
caplifeFile = fullfile(inputFolder, 'CAPSLIFE.csv');
caplifeData = readtable(caplifeFile);

% Keep only necessary columns in specified order
requiredCols = {'SCRNO', 'VISCODE', 'EXAMDATE', 'CAPSSCORE'};
caplifeData = caplifeData(:, requiredCols);

% Keep only rows where SCRNO is in the list of unique IDs
isValidID = ismember(caplifeData.SCRNO, uniqueIDs);
caplifeData = caplifeData(isValidID, :);

% Save the cleaned data to the new folder
outputFile = fullfile(outputFolder, 'CAPSLIFE_cleaned.csv');
writetable(caplifeData, outputFile);


%% CES file (full name: Combat Exposure Scale (CES))
% Load CES.csv file
cesFile = fullfile(inputFolder, 'CES.csv');
cesData = readtable(cesFile);

% Keep only necessary columns in specified order
requiredCols = {'SCRNO', 'VISCODE', 'EXAMDATE', ...
                'CES1A', 'CES2', 'CES3A', 'CES4', 'CES5', 'CES6', 'CES7'};
cesData = cesData(:, requiredCols);

% Keep only rows with VISCODE equal to 'sc3va'
validViscodes = strcmpi(string(cesData.VISCODE), 'sc3va');
cesData = cesData(validViscodes, :);

% Keep only rows where SCRNO is in the list of unique IDs
isValidID = ismember(cesData.SCRNO, uniqueIDs);
cesData = cesData(isValidID, :);

% Save the cleaned data to the new folder
outputFile = fullfile(outputFolder, 'CES_cleaned.csv');
writetable(cesData, outputFile);

%% FAQ file (full name: Functional Activities Questionnaire (FAQ))

% Load FAQ.csv file
faqFile = fullfile(inputFolder, 'FAQ.csv');
faqData = readtable(faqFile);

% Keep only necessary columns in specified order
requiredCols = {'SCRNO', 'VISCODE', 'USERDATE', 'FAQTOTAL'};
faqData = faqData(:, requiredCols);

% Keep only rows with VISCODE equal to 'bl' or 'm12'
validViscodes = ismember(lower(string(faqData.VISCODE)), ["bl", "m12"]);
faqData = faqData(validViscodes, :);

% Keep only rows where SCRNO is in the list of unique IDs
isValidID = ismember(faqData.SCRNO, uniqueIDs);
faqData = faqData(isValidID, :);

% Save the cleaned data to the new folder
outputFile = fullfile(outputFolder, 'FAQ_cleaned.csv');
writetable(faqData, outputFile);

%% GDSCALE file (full name: Geriatric Depression Scale (GDS))

% Load GDSCALE.csv file
gdsFile = fullfile(inputFolder, 'GDSCALE.csv');
gdsData = readtable(gdsFile);

% Keep only necessary columns in specified order
requiredCols = {'SCRNO', 'VISCODE', 'USERDATE', 'GDTOTAL'};
gdsData = gdsData(:, requiredCols);

% Keep only rows with VISCODE equal to 'sc' or 'm12'
validViscodes = ismember(lower(string(gdsData.VISCODE)), ["sc", "m12"]);
gdsData = gdsData(validViscodes, :);

% Keep only rows where SCRNO is in the list of unique IDs
isValidID = ismember(gdsData.SCRNO, uniqueIDs);
gdsData = gdsData(isValidID, :);

% Save the cleaned data to the new folder
outputFile = fullfile(outputFolder, 'GDSCALE_cleaned.csv');
writetable(gdsData, outputFile);

%% MMSE file (full name: Mini-Mental State Examination (MMSE))

% Load MMSE.csv file
mmseFile = fullfile(inputFolder, 'MMSE.csv');
mmseData = readtable(mmseFile);

% Keep only necessary columns in specified order
requiredCols = {'SCRNO', 'VISCODE', 'USERDATE', 'MMSCORE'};
mmseData = mmseData(:, requiredCols);

% Keep only rows with VISCODE equal to 'sc' or 'm12'
validViscodes = ismember(lower(string(mmseData.VISCODE)), ["sc", "m12"]);
mmseData = mmseData(validViscodes, :);

% Keep only rows where SCRNO is in the list of unique IDs
isValidID = ismember(mmseData.SCRNO, uniqueIDs);
mmseData = mmseData(isValidID, :);

% Save the cleaned data to the new folder
outputFile = fullfile(outputFolder, 'MMSE_cleaned.csv');
writetable(mmseData, outputFile);

%% NEUROBAT file (full name: Neuropsychological Battery)
%note: here only the baseline visit has a non NaN value at total score

% Load NEUROBAT.csv file
neurobatFile = fullfile(inputFolder, 'NEUROBAT.csv');
neurobatData = readtable(neurobatFile);

% Keep only necessary columns in specified order
requiredCols = {'SCRNO', 'VISCODE', 'USERDATE', 'ANARTERR'};
neurobatData = neurobatData(:, requiredCols);

% Keep only rows with VISCODE equal to 'sc', 'm12', or 'bl'
validViscodes = ismember(lower(string(neurobatData.VISCODE)), "bl");
neurobatData = neurobatData(validViscodes, :);

% Keep only rows where SCRNO is in the list of unique IDs
isValidID = ismember(neurobatData.SCRNO, uniqueIDs);
neurobatData = neurobatData(isValidID, :);

% Save the cleaned data to the new folder
outputFile = fullfile(outputFolder, 'NEUROBAT_cleaned.csv');
writetable(neurobatData, outputFile);


%% SHQ file (full name: Smoking History Questionnaire)
% NOTE: probably dont do this because of data inconsistencies
% % Load SHQ.csv file
% shqFile = fullfile(inputFolder, 'SHQ.csv');
% shqData = readtable(shqFile);
% 
% % Keep only necessary columns in specified order
% % SHQFTOTYR column question: How many total years did you smoke over your lifetime?
% requiredCols = {'SCRNO', 'VISCODE', 'USERDATE', 'SHQFTOTYR'};
% shqData = shqData(:, requiredCols);
% 
% % Keep only rows where SCRNO is in the list of unique IDs
% isValidID = ismember(shqData.SCRNO, uniqueIDs);
% shqData = shqData(isValidID, :);
% 
% % Save the cleaned data to the new folder
% outputFile = fullfile(outputFolder, 'SHQ_cleaned.csv');
% writetable(shqData, outputFile);