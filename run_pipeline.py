"""
CampusPlacementIQ · Master Pipeline Orchestrator
Executes all 48 curriculum sessions sequentially across Foundations, Linear Optimization,
Tree Ensembles, Unsupervised Clustering, and Model Evaluation & Infrastructure,
followed by visual gallery generation.
"""

import os
import sys
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXEC = sys.executable

CURRICULUM_PIPELINE = [
    # Module 01: Foundations
    ("Session 01: Cohort Ingestion", "src/module_01_foundations/01_cohort_ingestion.py"),
    ("Session 02: Heuristic Baseline", "src/module_01_foundations/02_heuristic_baseline.py"),
    ("Session 03: Exploratory Profiling", "src/module_01_foundations/03_exploratory_profiling.py"),
    ("Session 04: Data Sanitization", "src/module_01_foundations/04_data_sanitization.py"),

    # Module 02: Linear Optimization
    ("Session 05: Data Partitioning", "src/module_02_linear_optimization/05_data_partitioning.py"),
    ("Session 06: OLS Linear Regression", "src/module_02_linear_optimization/06_ols_linear_regression.py"),
    ("Session 07: Feature Standardization", "src/module_02_linear_optimization/07_feature_standardization.py"),
    ("Session 08: Categorical Imputation", "src/module_02_linear_optimization/08_categorical_imputation.py"),
    ("Session 09: Logistic Classification", "src/module_02_linear_optimization/09_logistic_classification.py"),
    ("Session 10: Regularization Shrinkage", "src/module_02_linear_optimization/10_penalty_regularization.py"),
    ("Session 11: End-to-End Pipeline", "src/module_02_linear_optimization/11_pipeline_orchestration.py"),
    ("Session 12: Coefficient Attribution", "src/module_02_linear_optimization/12_coefficient_attribution.py"),

    # Module 03: Tree Ensembles
    ("Session 13: Decision Trees", "src/module_03_tree_ensembles/13_cart_decision_tree.py"),
    ("Session 14: Pruning & Splitting", "src/module_03_tree_ensembles/14_tree_pruning_splitting.py"),
    ("Session 15: Bias-Variance Analysis", "src/module_03_tree_ensembles/15_bias_variance_depth.py"),
    ("Session 16: Random Forests & Bagging", "src/module_03_tree_ensembles/16_random_forest_bagging.py"),
    ("Session 17: Sequential Boosting", "src/module_03_tree_ensembles/17_sequential_boosting.py"),
    ("Session 18: Modern Boosted Trees", "src/module_03_tree_ensembles/18_modern_boosted_trees.py"),
    ("Session 19: SHAP Attribution", "src/module_03_tree_ensembles/19_shap_explainability.py"),
    ("Session 20: Cross-Model Tournament", "src/module_03_tree_ensembles/20_cross_model_tournament.py"),

    # Module 04: Unsupervised Learning
    ("Session 21: K-Means Clustering", "src/module_04_unsupervised_learning/21_kmeans_clustering.py"),
    ("Session 22: K-Means++ & Scalability", "src/module_04_unsupervised_learning/22_kmeans_plusplus_scalability.py"),
    ("Session 23: Principal Component Analysis", "src/module_04_unsupervised_learning/23_pca_decomposition.py"),
    ("Session 24: Hierarchical Clustering", "src/module_04_unsupervised_learning/24_hierarchical_clustering.py"),
    ("Session 25: Density-Based Clustering", "src/module_04_unsupervised_learning/25_dbscan_density.py"),
    ("Session 26: t-SNE Manifold Visualization", "src/module_04_unsupervised_learning/26_tsne_manifold.py"),
    ("Session 27: UMAP Manifold Projection", "src/module_04_unsupervised_learning/27_umap_manifold.py"),
    ("Session 28: Anomaly & Outlier Detection", "src/module_04_unsupervised_learning/28_anomaly_detection.py"),
    ("Session 29: Cluster-Augmented Features", "src/module_04_unsupervised_learning/29_cluster_augmented_features.py"),

    # Module 05: Evaluation, Validation & Optimization
    ("Session 30: Split Discipline & Data Leakage", "src/module_05_model_evaluation/30_split_discipline_leakage.py"),
    ("Session 31: K-Fold & Stratified Cross-Validation", "src/module_05_model_evaluation/31_cross_validation_schemes.py"),
    ("Session 32: Pipeline & Held-Out Test Evaluation", "src/module_05_model_evaluation/32_pipeline_test_evaluation.py"),
    ("Session 33: Nested Stratified Cross-Validation", "src/module_05_model_evaluation/33_nested_stratified_cv.py"),
    ("Session 34: ROC & PR Curve Diagnostics", "src/module_05_model_evaluation/34_roc_pr_curve_diagnostics.py"),
    ("Session 35: Regression Loss Robustness", "src/module_05_model_evaluation/35_regression_loss_robustness.py"),
    ("Session 36: Hyperparameter Search Schemes", "src/module_05_model_evaluation/36_hyperparameter_search_schemes.py"),
    ("Session 37: Learning Curves & Complexity", "src/module_05_model_evaluation/37_learning_curves_complexity.py"),
    ("Session 38: Bayesian TPE Optimization", "src/module_05_model_evaluation/38_bayesian_tpe_optimization.py"),
    ("Session 39: Probability Calibration & Brier", "src/module_05_model_evaluation/39_probability_calibration_brier.py"),

    # Module 06: ML Engineering & Production Serving
    ("Session 40: Production Pipeline Packaging", "src/module_06_ml_engineering/40_production_pipeline_packaging.py"),
    ("Session 41: Training-Serving Skew Detection", "src/module_06_ml_engineering/41_training_serving_skew.py"),
    ("Session 42: Model Serialization & Portability", "src/module_06_ml_engineering/42_model_serialization_portability.py"),
    ("Session 43: Microservice Inference Serving", "src/module_06_ml_engineering/43_microservice_inference_serving.py"),
    ("Session 44: Automated ML Testing Suite", "src/module_06_ml_engineering/44_automated_ml_testing.py"),
    ("Session 45: Data & Concept Drift Monitoring", "src/module_06_ml_engineering/45_data_concept_drift_monitoring.py"),
    ("Session 46: Experiment Tracking & Registry", "src/module_06_ml_engineering/46_experiment_tracking_registry.py"),
    ("Session 47: Docker & CI/CD Deployment Audit", "src/module_06_ml_engineering/47_containerization_cicd_pipeline.py"),
    ("Session 48: Executive Capstone Summary", "src/module_06_ml_engineering/48_executive_capstone_summary.py"),

    # Visual Analytics Showcase
    ("Showcase Visual Gallery", "src/module_05_visual_analytics/showcase_figure_generator.py"),
]


def execute_master_pipeline():
    print("=" * 70)
    print("CAMPUSPLACEMENTIQ · MASTER CURRICULUM PIPELINE RUNNER")
    print("=" * 70)
    start_total = time.time()

    success_count = 0
    total_steps = len(CURRICULUM_PIPELINE)

    for idx, (label, rel_path) in enumerate(CURRICULUM_PIPELINE, 1):
        script_full_path = os.path.join(PROJECT_ROOT, rel_path)
        print(f"\n[{idx:02d}/{total_steps:02d}] Executing: {label}...")
        print(f"      Path: {rel_path}")

        t0 = time.time()
        result = subprocess.run([PYTHON_EXEC, script_full_path], cwd=PROJECT_ROOT)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"❌ Error encountered in {rel_path} (exit code: {result.returncode}). Pipeline aborted.")
            sys.exit(result.returncode)

        print(f"      ✓ Finished in {elapsed:.2f}s")
        success_count += 1

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"🎉 ALL {success_count}/{total_steps} SESSIONS EXECUTED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("All diagnostic reports, visual charts, and model artifacts are up to date.")
    print("=" * 70)


if __name__ == "__main__":
    execute_master_pipeline()
