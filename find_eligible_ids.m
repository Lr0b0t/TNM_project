clc; close all; clear;

% filename = 'screening_and_1_year_follow_up_all_collumns.csv';

filename = 'screening_and_1yr_follow-up_AND.csv';

data = readtable(filename); 


ids = data.SubjectID;
visits = data{:,5};  

valid_ids = false(size(ids));


unique_ids = unique(ids);

num_unique_ids = numel(unique_ids) % num of unique ids before the clening

% by cleaning, is meant that only the ID's that have at least one Screening
% MRI and at least one Year follow-up clinic visit, are considered

for i = 1:numel(unique_ids)
    current_id = unique_ids(i);
    
    % logical mask for current ID
    rows = (ids == current_id);
    
    % visit types for this ID
    visit_types = visits(rows);
    
    has_screening = any(strcmp(visit_types, 'Screening MRI'));
    has_year1     = any(strcmp(visit_types, 'Year 1 Clinic Visit'));
    
    % if both are present, mark all rows of that ID as valid
    if has_screening && has_year1
        valid_ids(rows) = true;
    end
end

final_data = data(valid_ids, :);

fprintf('Number of rows in final_data (including some duplicates in some cases): %d\n', height(final_data));


%%
ids_final = final_data.SubjectID % displays the ID's that meet teh above criteria


unique_ids_final = unique(ids_final);

writematrix(unique_ids_final, 'unique_ids_final_2.csv');
num_unique_ids_final = numel(unique_ids_final) % counts the number of the final subjects