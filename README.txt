Functional Connectivity and rDCM Pipeline
========================================

This pipeline extracts regional time series from resting-state fMRI (rsfMRI) data,
computes functional connectivity (FC), and estimates rigid regression dynamic causal
models (rDCMs) for multiple subjects.


Directory Structure
-------------------

Your working directory should be organized as follows:

data/
├── FunImgARCFW_n88/
│   ├── 0006315/
│   │   └── wFiltered_4DVolume.nii
│   ├── 000XXXX/
│   │   └── wFiltered_4DVolume.nii
│   └── ...
├── BN_Atlas_246_3mm.nii
├── BNA_matrix_binary_246x246.csv
└── connectivity_n88/
    ├── 0006315/
    │   ├── time_series.mat
    │   ├── func_connectivity.mat
    │   └── rdcm_connectivity.mat
    ├── 000XXXX/
    │   └── ...
    └── ...

------------
- FunImgARCFW_n88/: Folder containing rsfMRI NIfTI files, one subfolder per subject.
- BN_Atlas_246_3mm.nii: Atlas used for extracting regional time series.
- BNA_matrix_binary_246x246.csv: Structural adjacency matrix for rDCM estimation.
- connectivity_n88/: Output folder. Each subject gets their own folder with results:
    - time_series.mat: Extracted BOLD signals.
    - func_connectivity.mat: Functional connectivity matrices.
    - rdcm_connectivity.mat: Estimated rDCM parameters.


=== Run Timeseries Extraction and Connectivity Estimation Pipeline ===
All the steps for extracting time series and estimating functional connectivity 
are implemented in the notebook:

    extract_timeseries_and_connectivity.ipynb

This notebook covers:

1) Extract Time Series
----------------------

Call the function:

    extract_timeseries_for_all_subjects(
        "data/FunImgARCFW_n88", 
        "data/connectivity_n88", 
        "wFiltered_4DVolume.nii", 
        "data", 
        "BN_Atlas_246_3mm.nii"
    )

This function:
- Loads preprocessed fMRI data from each subject.
- Uses the atlas to extract average BOLD signals from 246 regions.
- Saves time_series.mat in each subject's output folder.


2) Compute Functional Connectivity
----------------------------------

Call the function:

    compute_func_connectivity_for_all_subjects("data/connectivity_n88")

This function computes for each subject:
- fc_mat: Pearson correlation matrix.
- fc_z: Fisher Z-transformed correlation matrix.
- cov_mat: Covariance matrix.
Results are saved in func_connectivity.mat.


3) Estimate rDCM Models
-----------------------

Call the function:

    estimate_rdcm_for_all_subjects(
        "data/connectivity_n88", 
        "data/BN_Atlas_246_3mm.nii", 
        "data/BNA_matrix_binary_246x246.csv"
    )

This function:
- Loads time series and uses structural matrix to fit an rDCM.
- Saves output_m_all to rdcm_connectivity.mat for each subject.


End of README
-------------
