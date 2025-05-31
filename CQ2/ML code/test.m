clc; clear; close all;


% ------------------------------------------------------------------------
% Script: load_vae_results.m
% Purpose: From inside the “MLCode” folder, move two levels up, navigate
%          to “latentresults/vae-results”, and load the specified .mat files.
%
% Usage:
%   1. Place this script in the “MLCode” directory (i.e. cq2.mlcode/MLCode).
%   2. Run it; it will automatically locate and load “fc-dim10.mat” and
%      “rdcm-dim10.mat” from the latentresults/vae-results folder.
% ------------------------------------------------------------------------

% 1) Determine the folder where this script lives (MLCode folder)
scriptDir = fileparts( mfilename('fullpath') );

% 2) Go two levels up from MLCode:
%      MLCode → cq2.mlcode → (parent of cq2.mlcode)
parentLevel1 = fileparts(scriptDir);       % one level up: cq2.mlcode
parentLevel2 = fileparts(parentLevel1);    % two levels up

% 3) Construct path to “latentresults/vae-results”
vaeResultsDir = fullfile(parentLevel2, 'latent_results', 'vae_results');

% 4) Verify that the folder exists
if ~isfolder(vaeResultsDir)
    error('Directory not found: %s\nMake sure “latentresults/vae-results” exists two levels above MLCode.', ...
           vaeResultsDir);
end


% connDir = fullfile(baseDir, 'latentresults\v');
% if ~exist(connDir, 'dir')
%     connDir = fullfile(baseDir, '..','..', 'latentresults');
% end

% 5) Define full paths to the .mat files
file1 = fullfile(vaeResultsDir, 'FC_dim10.mat');
file2 = fullfile(vaeResultsDir, 'RDCM_dim10.mat');

% 6) Check for existence before loading
if ~isfile(file1)
    error('File not found: %s', file1);
end
if ~isfile(file2)
    error('File not found: %s', file2);
end

% 7) Load the .mat files into distinct variables (structures)
fcData   = load(file1);    % loads all variables from fc-dim10.mat into struct fcData
rdcmData = load(file2);    % loads all variables from rdcm-dim10.mat into struct rdcmData

% 8) Optional: Display confirmation and list loaded variables
fprintf('Loaded ''%s'' with variables:\n', file1);
disp(fieldnames(fcData));

fprintf('Loaded ''%s'' with variables:\n', file2);
disp(fieldnames(rdcmData));

% 9) (Now you can access the contents of each .mat file via fcData.variableName,
%       rdcmData.variableName, etc.)

% Example usage:
%   If fc-dim10.mat contained a variable called “fc_matrix”, you can reference it as:
%       myFC = fcData.fc_matrix;
%
%   Similarly, if rdcm-dim10.mat contained “rdcm_results”, you can do:
%       myRDCM = rdcmData.rdcm_results;
% ------------------------------------------------------------------------
