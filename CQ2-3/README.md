# ML Code: Machine Learning for Cognitive Decline Prediction

This folder contains MATLAB scripts and functions for predicting cognitive decline, for addressing clinical questions 2 and 3. The workflow applies and compares Elastic Net regression, Random Forest regression, and Support Vector Machines (SVMs) with nested cross-validation, and implements a “mixture of experts” integration strategy.

## Folder Structure & Main Scripts

- **run_Elastic_Net_Regression.m** (function)

- **run_Random_Forest_Regression.m** (function)

- **run_nested_cv_SVM.m** (function)

- **svm_inner_cv.m** (function)
  Helper function for inner cross-validation loops for SVM models.

- **run_ML_models_for_questionnaire_features.m** (script)
  Applies all three regression models (Elastic Net, Random Forest, SVM) to the clinical/psychometric questionnaire features.

- **run_ML_models_for_FC.m** (script)
  Applies all three models to the functional connectivity (FC) features.

- **run_ML_models_for_rDCM_EC.m** (script)
  Applies all three models to the effective connectivity (EC) features as derived from rDCM.

- **run_ML_models_for_APOE.m** (script)
  Applies all models to features related to APOE genotype (third clinical question).

- **final_experts.m** (script)
  Combines predictions from the best-performing models in each feature domain (mixture of experts approach). Produces the final evaluation and integration of results.

## Workflow

1. **Feature Domain Evaluation:** 
   - For each data type (questionnaire, FC, EC, APOE), all three models are trained, validated, and compared using the relevant `run_ML_models_for_*.m` script.
   - Hyperparameters are selected using nested cross-validation.

2. **Expert Selection & Ensemble:** 
   - The `final_experts.m` script implements the mixture of experts strategy, integrating the best expert model from each domain for final prediction and interpretation.


## Dependencies

- **MATLAB Statistics and Machine Learning Toolbox**
- **Input Data:** 
  1. The scripts expect preprocessed CSV files (train/test sets) in 
  `../data cleanup and management/final files/`
  2.  The scripts expects connectivity files (.mat) in 
  `../latent_results/vae_results/`

## Data Organization

- **Train and Test CSVs:** 
  - Should have subject ID in the first column, followed by features and outcome scores (`MMSCORE_followUp`, `CDSOB_followUp`, `GDTOTAL_followUp`).
- **Connectivity Matrices:** 
  - FC and EC matrices are assumed to be available for all subjects at once as `.mat` files in a designated connectivity folder.

## Documentation

- Each script contains detailed comments and documentation within the file.

---

