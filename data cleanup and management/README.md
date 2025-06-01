 

## Purpose

This repository/folder provides scripts and step-by-step procedures for cleaning, organizing, and preparing study data as downloaded from the database website. The ultimate goal is to enable downstream generative embedding and machine learning analysis by constructing well-structured and reliable feature sets from the raw data.
---
## One-Click Full Pipeline

A master script—**`run_full_data_management_pipeline.m`**—is provided in this repository. 
**This script executes the *entire* data cleaning and feature construction pipeline from start to finish.** 
By running this single script, all steps described below (from raw data to ready-for-ML feature sets) will be performed automatically, eliminating the need to execute individual scripts manually. 

---

## Folder Structure

- **study data as downloaded/** 
  Raw data files and folders, exactly as downloaded from the database website.

- **find eligible ids/** 
  Contains scripts for determining the list of eligible subjects (IDs) based on imaging quality and the availability of key assessments.

- **utilities/** 
  Central location for scripts that perform data cleaning, integrity checks, and data processing steps. Also stores the latest list of eligible IDs as `unique_patient_ids.csv`.

- **Neuropsychological Useful data/** 
  Cleaned and filtered neuropsychological data files, ready for analysis.

- **Biospecimen Useful data/** 
  Cleaned and filtered biospecimen data files, ready for analysis.

- **final files/** 
  Master feature sets and train/test splits for modeling and machine learning. This folder also contains imputed datasets.

---

## Workflow Overview

1. **Start with the raw data:** 
   Ensure that all downloaded study data is unzipped and available in the `study data as downloaded` folder.

2. **Find Eligible IDs:** 
   - Navigate to the `find eligible ids` folder.
   - Run the script provided to generate the list of eligible subject IDs (those with good baseline imaging and key assessments available).
   - The resulting list will be saved as a CSV file in the `utilities` folder.

3. **Data Cleaning – Neuropsychological Data:** 
   - From the `utilities` folder, run the relevant cleaning scripts for neuropsychological data.
   - This will create the `Neuropsychological Useful data` folder, containing cleaned files with only eligible subjects and selected columns.

4. **Data Cleaning – Biospecimen Data:** 
   - Similarly, run the cleaning scripts for biospecimen data to create the `Biospecimen Useful data` folder.

5. **Split Neuropsychological Data by Visit Code:** 
   - Use scripts to separate neuropsychological data into baseline (screening) and one-year follow-up variables by the visit code (`VISCODE`).
   - This step is crucial for constructing feature sets for prediction and regression analyses.

6. **Integrity Checks:** 
   - Run provided integrity check scripts (in `utilities`) for both neuropsychological and biospecimen data to ensure all IDs match and that there are no duplicates or missing IDs.

7. **Construct Master Feature Sets:** 
   - Build comprehensive feature sets for both baseline and one-year follow-up data using the cleaned files.
   - These are saved in the `final files` folder.

8. **Create Train and Test Splits:** 
   - Using scripts in `utilities`, split the data into training and test sets for clinical questions 2 and 3.
   - The split ensures that the test set contains only subjects with fully observed data (no missing values).

9. **Impute Missing Data:** 
   - For columns with missing values in the training and test sets, perform mean imputation using provided scripts.
   - Imputed files are saved in the `final files` folder.

---

## Notes

- All scripts are intended to be run from within the `utilities` folder unless otherwise specified.
- All outputs and cleaned files are stored in their respective folders for easy navigation and reproducibility.
- Detailed comments exist in every script.

---

For further details on each data file and the features extracted, see Table 1 in the report.


