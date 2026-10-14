"""
CampusPlacementIQ · Central Configuration & Path Registry
Defines unified paths, feature schemas, model output directories, and random seeds.
"""

import os

# Project Root Directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data Paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "placement_raw.csv")
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "placement_clean.csv")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "splits")

# Output Directories per Curriculum Session
INGESTION_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_01_foundations", "01_ingestion")
BASELINE_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_01_foundations", "02_baseline")
EDA_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_01_foundations", "03_eda")
CLEANING_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_01_foundations", "04_cleaning")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_02_linear_optimization", "05_split")
REGRESSION_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_02_linear_optimization", "06_linear_regression")
SCALING_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_02_linear_optimization", "07_scaling")
ENCODING_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_02_linear_optimization", "08_encoding")
LOGISTIC_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_02_linear_optimization", "09_logistic_regression")
REGULARIZATION_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_02_linear_optimization", "10_regularization")
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_02_linear_optimization", "11_pipeline")
COEFFICIENTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_02_linear_optimization", "12_coefficients")
DECISION_TREE_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_03_tree_ensembles", "13_decision_tree")
SPLITTING_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_03_tree_ensembles", "14_splitting_criteria")
BIAS_VARIANCE_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_03_tree_ensembles", "15_bias_variance")
RANDOM_FOREST_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_03_tree_ensembles", "16_random_forest")
BOOSTING_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_03_tree_ensembles", "17_boosting")
MODERN_BOOSTING_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_03_tree_ensembles", "18_modern_boosting")
SHAP_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_03_tree_ensembles", "19_shap")
MODEL_COMPARISON_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_03_tree_ensembles", "20_model_comparison")
KMEANS_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_04_unsupervised_learning", "21_kmeans")
KMEANS_PP_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_04_unsupervised_learning", "22_kmeans_plusplus")
PCA_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_04_unsupervised_learning", "23_pca")
HIERARCHICAL_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_04_unsupervised_learning", "24_hierarchical")
DBSCAN_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_04_unsupervised_learning", "25_dbscan")
TSNE_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_04_unsupervised_learning", "26_tsne")
UMAP_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_04_unsupervised_learning", "27_umap")
ANOMALY_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_04_unsupervised_learning", "28_anomaly")
CLUSTER_FEATURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_04_unsupervised_learning", "29_cluster_features")

# Module 5: Model Evaluation & Optimization
EXAMPLE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ExampleDataSetPP.csv")
MODULE_05_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_05_model_evaluation")
SPLIT_LEAKAGE_DIR = os.path.join(MODULE_05_DIR, "30_split_leakage")
CV_SCHEMES_DIR = os.path.join(MODULE_05_DIR, "31_cross_validation")
PIPELINE_EVAL_DIR = os.path.join(MODULE_05_DIR, "32_pipeline_eval")
NESTED_CV_DIR = os.path.join(MODULE_05_DIR, "33_nested_cv")
ROC_PR_DIR = os.path.join(MODULE_05_DIR, "34_roc_pr_curves")
REG_LOSS_DIR = os.path.join(MODULE_05_DIR, "35_regression_loss")
SEARCH_SCHEMES_DIR = os.path.join(MODULE_05_DIR, "36_search_schemes")
LEARNING_CURVES_DIR = os.path.join(MODULE_05_DIR, "37_learning_curves")
OPTUNA_TUNING_DIR = os.path.join(MODULE_05_DIR, "38_optuna_tuning")
CALIBRATION_DIR = os.path.join(MODULE_05_DIR, "39_calibration")

# Module 6: ML Engineering & Production Serving
MODULE_06_DIR = os.path.join(PROJECT_ROOT, "outputs", "module_06_ml_engineering")
PROD_PACKAGING_DIR = os.path.join(MODULE_06_DIR, "40_prod_packaging")
SKEW_DETECTION_DIR = os.path.join(MODULE_06_DIR, "41_skew_detection")
SERIALIZATION_DIR = os.path.join(MODULE_06_DIR, "42_serialization")
MICROSERVICE_DIR = os.path.join(MODULE_06_DIR, "43_microservice_serving")
AUTOMATED_TESTING_DIR = os.path.join(MODULE_06_DIR, "44_automated_testing")
DRIFT_MONITORING_DIR = os.path.join(MODULE_06_DIR, "45_drift_monitoring")
EXPERIMENT_TRACKING_DIR = os.path.join(MODULE_06_DIR, "46_experiment_tracking")
DOCKER_CICD_DIR = os.path.join(MODULE_06_DIR, "47_docker_cicd")
CAPSTONE_SUMMARY_DIR = os.path.join(MODULE_06_DIR, "48_capstone_summary")

# Module 5 Case Study Feature Names
M5_FEATURES = ["CGPA", "Aptitude", "Coding", "Communication", "Internship", "Projects"]
M5_TARGET = "Placed"

# Feature Column Definitions
TARGET_COLUMN = "PlacementStatus"
ID_COLUMN = "StudentID"

NUMERIC_COLUMNS = [
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
    "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8",
    "CGPA", "AttendancePercent", "Internships", "Projects",
    "Workshops", "Certifications", "Publications",
    "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore",
    "MockInterviewScore", "ExtraCurricular",
]

CATEGORY_COLUMNS = ["Gender", "City", "Stream", "Specialisation"]
YES_NO_COLUMNS = ["Hostel", "HistoryOfBacklogs"]
TEXT_COLUMNS = CATEGORY_COLUMNS + YES_NO_COLUMNS + ["CollegeTier", "CGPA_Tier"]

RANDOM_STATE = 42