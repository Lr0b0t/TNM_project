clc; close all; %clear;

% -------------------------------------------------------------------------
% Master pipeline to run all data processing scripts in order, from the
% top-level ("General Data Cleanup and Management") folder. The step can
% also be sun separately, manually.
% -------------------------------------------------------------------------

fprintf('--- Starting full data preparation pipeline ---\n');

% === 1: Find Eligible IDs (run from inside 'find eligible ids' folder) ===
fprintf('\n Finding eligible patient IDs...\n');
findEligibleFolder = fullfile(pwd, 'find eligible ids');

cd(findEligibleFolder);
run('find_eligible_patient_ids.m');

cd(".."); % Return to original folder

%%
% === 2: Remaining steps in 'utilities' ===
utilitiesFolder = fullfile(pwd, 'utilities'); % this varisble will be used
% only once, then utilitiesFolder gets deleted from the workspace
% and I did not want to make it a global variable, so for the rest of the
% code I use direclty the command to generate the correct path.

fprintf('\n Cleaning neuropsychological data...\n');
run(fullfile(utilitiesFolder, 'clean_neuropsychological_data.m'));
pwd
fprintf('\n Cleaning biospecimen data...\n');
run(fullfile(fullfile(pwd, 'utilities'), 'clean_biospecimen_data.m'));

fprintf('\n Splitting neuropsychological data by VISCODE...\n');
run(fullfile(fullfile(pwd, 'utilities'), 'split_cleaned_files_by_visitcode.m'));

fprintf('\n Checking patient ID integrity in neuro data...\n');
run(fullfile(fullfile(pwd, 'utilities'), 'check_patient_id_integrity_in_neuro_data.m'));

fprintf('\n Checking patient ID integrity in biospecimen data...\n');
run(fullfile(fullfile(pwd, 'utilities'), 'check_patient_id_integrity_in_biospecimen_data.m'));

fprintf('\n Building one-year follow-up master file...\n');
run(fullfile(fullfile(pwd, 'utilities'), 'build_1yr_followup_master_file.m'));

fprintf('\n Building baseline master feature file...\n');
run(fullfile(fullfile(pwd, 'utilities'), 'build_baseline_master_feature_file.m'));

fprintf('\n Splitting into train/test sets for Q2...\n');
run(fullfile(fullfile(pwd, 'utilities'), 'split_into_train_test_sets_Q2.m'));

fprintf('\n Imputing missing values in train/test sets...\n');
run(fullfile(fullfile(pwd, 'utilities'), 'mean_impute_train_test.m'));

fprintf('\n Splitting into train/test sets for Q3...\n');
run(fullfile(fullfile(pwd, 'utilities'), 'build_train_test_sets_Q3.m'));

fprintf('\n Building class labels...\n');
run(fullfile(fullfile(pwd, 'utilities'), 'build_classes.m'));


fprintf('\n--- Data preparation pipeline completed! :D ---\n');
