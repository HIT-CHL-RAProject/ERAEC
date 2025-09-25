# Environmental Risk Assessment of Emerging Contaminants (ERAEC)

Basic machine learning workflow and configuration files for chemical-biological data integration and pattern discovery purposes
This project explores how machine learning (ML) can identify and prioritize emerging contaminants (ECs) that disrupt microbial functions in various environments. It focuses on pattern discovery rather than predictive modeling, demonstrating how ML-based workflows highlight chemicals most likely driving community shifts or pathway disturbances.

Key Objectives

-Data Integration: Combine EC concentration data (often scaled by PNEC) with metagenomic or omics-derived endpoints (e.g., ARG abundance, metabolic pathways).

-Modeling Framework: Apply gradient boosting or random forest models with feature selection, cross-validation, and interpretability tools (SHAP, PDP).

Methodology Highlights:

-Preprocessing:  

  -Log-transform and normalize concentration/PNEC ratios or concentrations, depending on YAML configuration files.
  
  -Normalize biological endpoints, depending on YAML configuration files.
  
  -Filter out low-detection features.

-Cross-Validation:

  -5-fold, 10-fold, or leave-one-out CV.
  
  -CV R² is reported as the proportion of endpoint variance explained by chemical features within the dataset.


-Interpretability:

  -XGBoost feature importance.
  
  -SHAP summary plots.
  
  -Partial dependence curves to show concentration–response thresholds.

Case Studies:

-Landfill Leachate: Highly contaminated environment analyzed for carbon/nitrogen metabolic pathways.

-Yangtze River Sediment: Broader antibiotic and heavy metal contamination profiles linked to ARG dissemination.

# Repository Contents

Main_workflow/: Python scripts for data preprocessing, model training, and interpretability plots (with YAML configuration files).

data/: All relevant datasets for both cases in Main_workflow/.

Biodata_preprocessing: Describe the packages we used to process the biological data, which are irrelevant for the Main_workflow.



# Requirements

OS & Environment: Windows 11, Python 3.12.7 (tested on VS Code).

Dependencies (install via pip install -r requirements.txt or individually):

pandas
numpy
scikit-learn
tensorflow
keras-tuner
matplotlib
seaborn
xgboost
pyyaml
networkx
econml

# Instruction

1. Download or prepare data for the case study.

2. Edit file paths in configuration_path.yaml and configuration_arg.yaml.

3. Run:

  -case1_path_main.py for pathway case study.
  
  -case2_arg_main.py for ARG case study.
Results (figures, feature rankings, SHAP plots) will be saved automatically.

# Contact & Contributing

Contributors: Hanlin Cui, Bin Liang, Aijie Wang
OrganizationL: State Key Laboratory of Urban Water Resource and Environment, School of Civil & Environmental Engineering, Harbin Institute of Technology.
Contributions: Feel free to open issues or pull requests for improved data integration, advanced ML models, or additional visualization features.
For questions or collaboration, contact 21B929086@stu.hit.edu.cn; liangbin1214@hit.edu.cn
