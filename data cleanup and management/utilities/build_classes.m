clear; close all; %clc; 

% -------------------------------------------------------------------------
%
% Purpose:
%   Analyze MMSE changes between baseline and follow-up for all subjects in
%   the train and test imputed feature files. Detect and print cases where
%   the MMSE group classification worsens, stays the same, or improves.
%   Also, create new CSV files with an added 'decline' column.
%
% Usage:
%   - Run in the 'utilities' folder.
%
% -------------------------------------------------------------------------

% File paths
baseDir = fullfile('..', 'final files');
fileList = {'train_features_Q2_imputed.csv', 'test_features_Q2_imputed.csv'};
outputList = {'train_features_Q2_imputed_and_categorized.csv', 'test_features_Q2_impute_and_categorized_.csv'};

for fileIdx = 1:numel(fileList)
    fileName = fileList{fileIdx};
    filePath = fullfile(baseDir, fileName);
    outputFile = fullfile(baseDir, outputList{fileIdx});

    fprintf('\n---- Processing file: %s ----\n', fileName);
    T = readtable(filePath);

    % Column names for MMSE
    col_base = 'MMSCORE_baseline';
    col_follow = 'MMSCORE_followUp';

    if ~all(ismember({col_base, col_follow}, T.Properties.VariableNames))
        error('File %s does not contain the required MMSE columns!', fileName);
    end

    mmse_baseline = T.(col_base);
    mmse_followup = T.(col_follow);

    % Classify function
    group_baseline = arrayfun(@(x) classify_mmse(x), mmse_baseline, 'UniformOutput', false);
    group_followup = arrayfun(@(x) classify_mmse(x), mmse_followup, 'UniformOutput', false);

    % Decline label (initialize)
    decline = zeros(height(T), 1);

    % Counters
    nDrop = 0; nSame = 0; nImprove = 0;

    for i = 1:height(T)
        baseG = group_baseline{i};
        follG = group_followup{i};
        if isempty(baseG) || isempty(follG)
            continue
        end
        baseIdx = mmse_group_idx(baseG);
        follIdx = mmse_group_idx(follG);

        if follIdx > baseIdx
            nDrop = nDrop + 1;
            decline(i) = 1;
            fprintf('[Drop]    SCRNO: %s | MMSE: %g → %g | Group: %s → %s | Δ=%g\n', ...
                string(T.SCRNO(i)), mmse_baseline(i), mmse_followup(i), ...
                baseG, follG, mmse_followup(i) - mmse_baseline(i));
        elseif follIdx == baseIdx
            nSame = nSame + 1;
            fprintf('[Stable]  SCRNO: %s | MMSE: %g → %g | Group: %s → %s | Δ=%g\n', ...
                string(T.SCRNO(i)), mmse_baseline(i), mmse_followup(i), ...
                baseG, follG, mmse_followup(i) - mmse_baseline(i));
        elseif follIdx < baseIdx
            nImprove = nImprove + 1;
            fprintf('[Improve] SCRNO: %s | MMSE: %g → %g | Group: %s → %s | Δ=%g\n', ...
                string(T.SCRNO(i)), mmse_baseline(i), mmse_followup(i), ...
                baseG, follG, mmse_followup(i) - mmse_baseline(i));
        end
    end
    fprintf('\nSummary for %s:\n', fileName);
    fprintf('  Dropped to worse MMSE group:   %d\n', nDrop);
    fprintf('  Stayed in the same MMSE group: %d\n', nSame);
    fprintf('  Improved to better MMSE group: %d\n', nImprove);

    % Add 'decline' column and save new file
    T.decline = decline;
    writetable(T, outputFile);
    fprintf('Saved %s with decline column.\n', outputFile);
end

% ----- Helper functions -----
function grp = classify_mmse(score)
    if isnan(score)
        grp = '';
    elseif score >= 28
        grp = 'normal';
    elseif score >= 25
        grp = 'mci';
    elseif score >= 18
        grp = 'mild dementia';
    elseif score >= 10
        grp = 'moderate dementia';
    else
        grp = 'severe dementia';
    end
end

function idx = mmse_group_idx(group)
    % Returns group index: 1=normal, 2=mci, 3=mild, 4=moderate, 5=severe
    switch char(group)
        case 'normal', idx = 1;
        case 'mci', idx = 2;
        case 'mild dementia', idx = 3;
        case 'moderate dementia', idx = 4;
        case 'severe dementia', idx = 5;
        otherwise, idx = NaN;
    end
end
