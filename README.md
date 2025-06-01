

# Neuroimaging Pipeline for Clinical Questions in TBI, PTSD, and Dementia

This repository contains a modular neuroimaging analysis pipeline, structured into separate stages corresponding to distinct clinical questions and tasks. The code supports analysis of resting-state fMRI (rsfMRI) data and machine learning-based outcome prediction across several neurological and psychiatric conditions.

Before running julia code, you can execute setup.jl located in the root of this repository. This script sets up necessary Julia environments, installs required packages, and ensures consistent reproducibility across the pipeline.
## Repository Structure

### 1. `timeseries_and_connectivity/`

This folder contains code for extracting time series from preprocessed rsfMRI data and computing both functional and effective connectivity matrices. This serves as the foundational data processing stage for subsequent clinical analyses.

### 2. `CQ1/`

This folder contains scripts for **Clinical Question 1**:
Exploring **dimensionality reduction** and **clustering** techniques to subgroup subjects based on connectivity profiles. The goal is to identify subtypes within the TBI (Traumatic Brain Injury), PTSD (Post-Traumatic Stress Disorder), and control groups.

### 3. `CQ2_3/`

This folder includes machine learning code for **Clinical Questions 2 and 3**:

* **CQ2**: Predicting 12-month cognitive and mental health outcomes using connectivity data.
* **CQ3**: Assessing feature importance and model generalizability across diagnostic categories.

Targets include:

* **Mini-Mental State Examination (MMSE)**
* **Clinical Dementia Rating – Sum of Boxes (CDR-SOB)**
* **Geriatric Depression Scale (GDS-15)**

## Data Access

Due to the large size of neuroimaging data, it is not hosted directly in this repository.

🔗 **Access the data here**: [Polybox Storage Link](https://polybox.ethz.ch/index.php/s/NQjJR4CXEkykmGp?path=%2F)

A separate `README.md` file within the Polybox directory provides descriptions of the available data files and folder organization.

## Getting Started

Each folder contains its own `README.md` file with instructions specific to that analysis stage, including:

* Environment setup
* Dependencies
* Data formats
* Run instructions
* Output description

We recommend starting with the `timeseries_and_connectivity` folder before proceeding to `CQ1` and then to `CQ2_3`.
 
