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
INGESTION_DIR = os.path.join(PROJECT_ROOT, "outputs", "ingestion")
BASELINE_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline")
EDA_DIR = os.path.join(PROJECT_ROOT, "outputs", "eda")
CLEANING_DIR = os.path.join(PROJECT_ROOT, "outputs", "cleaning")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "outputs", "split")
REGRESSION_DIR = os.path.join(PROJECT_ROOT, "outputs", "linear_regression")
SCALING_DIR = os.path.join(PROJECT_ROOT, "outputs", "scaling")
ENCODING_DIR = os.path.join(PROJECT_ROOT, "outputs", "encoding")
LOGISTIC_DIR = os.path.join(PROJECT_ROOT, "outputs", "logistic_regression")
REGULARIZATION_DIR = os.path.join(PROJECT_ROOT, "outputs", "regularization")
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "outputs", "pipeline")
COEFFICIENTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "coefficients")
DECISION_TREE_DIR = os.path.join(PROJECT_ROOT, "outputs", "decision_tree")
SPLITTING_DIR = os.path.join(PROJECT_ROOT, "outputs", "splitting_criteria")
BIAS_VARIANCE_DIR = os.path.join(PROJECT_ROOT, "outputs", "bias_variance")
RANDOM_FOREST_DIR = os.path.join(PROJECT_ROOT, "outputs", "random_forest")
BOOSTING_DIR = os.path.join(PROJECT_ROOT, "outputs", "boosting")
MODERN_BOOSTING_DIR = os.path.join(PROJECT_ROOT, "outputs", "modern_boosting")
SHAP_DIR = os.path.join(PROJECT_ROOT, "outputs", "shap")
MODEL_COMPARISON_DIR = os.path.join(PROJECT_ROOT, "outputs", "model_comparison")
KMEANS_DIR = os.path.join(PROJECT_ROOT, "outputs", "kmeans")
HIERARCHICAL_DIR = os.path.join(PROJECT_ROOT, "outputs", "hierarchical")

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