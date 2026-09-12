"""
CampusPlacementIQ · Master Pipeline Orchestrator
Executes all 22 curriculum sessions sequentially across Foundations, Linear Optimization,
Tree Ensembles, and Unsupervised Clustering, followed by visual gallery generation.
"""

import os
import sys
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXEC = sys.executable

CURRICULUM_PIPELINE = [
    # Module 1: Foundations
    ("Session 01: Cohort Ingestion", "src/foundations_pipeline/s01_cohort_ingestion.py"),
    ("Session 02: Heuristic Baseline", "src/foundations_pipeline/s02_heuristic_baseline.py"),
    ("Session 03: Exploratory Profiling", "src/foundations_pipeline/s03_exploratory_profiling.py"),
    ("Session 04: Data Sanitization", "src/foundations_pipeline/s04_data_sanitization.py"),

    # Module 2: Linear Optimization
    ("Session 05: Data Partitioning", "src/linear_optimization/s05_data_partitioning_splits.py"),
    ("Session 06: OLS Linear Regression", "src/linear_optimization/s06_ols_linear_regression.py"),
    ("Session 07: Feature Standardization", "src/linear_optimization/s07_feature_standardization.py"),
    ("Session 08: Categorical Imputation", "src/linear_optimization/s08_categorical_imputation.py"),
    ("Session 09: Logistic Classification", "src/linear_optimization/s09_logistic_classifier_boundary.py"),
    ("Session 10: Regularization Shrinkage", "src/linear_optimization/s10_penalty_regularization.py"),
    ("Session 11: End-to-End Pipeline", "src/linear_optimization/s11_end_to_end_orchestration.py"),
    ("Session 12: Coefficient Attribution", "src/linear_optimization/s12_interpretability_coefficients.py"),

    # Module 3: Tree Ensembles
    ("Session 13: Decision Trees", "src/tree_ensemble_analytics/s13_cart_decision_tree.py"),
    ("Session 14: Pruning & Splitting", "src/tree_ensemble_analytics/s14_tree_pruning_split_criteria.py"),
    ("Session 15: Bias-Variance Analysis", "src/tree_ensemble_analytics/s15_bias_variance_depth_tradeoff.py"),
    ("Session 16: Random Forests & Bagging", "src/tree_ensemble_analytics/s16_bagging_random_forest.py"),
    ("Session 17: Sequential Boosting", "src/tree_ensemble_analytics/s17_gradient_boosting_core.py"),
    ("Session 18: Modern Boosted Trees", "src/tree_ensemble_analytics/s18_modern_boosted_ensembles.py"),
    ("Session 19: SHAP Attribution", "src/tree_ensemble_analytics/s19_shap_feature_explanations.py"),
    ("Session 20: Cross-Model Tournament", "src/tree_ensemble_analytics/s20_cross_model_tournament.py"),

    # Module 4: Unsupervised Segmentation
    ("Session 21: K-Means Clustering", "src/unsupervised_segmentation/s21_kmeans_student_clustering.py"),
    ("Session 24: Hierarchical Clustering", "src/unsupervised_segmentation/s24_hierarchical_dendrogram_clustering.py"),

    # Showcase Figures
    ("Showcase Visual Gallery", "src/showcase_figure_generator.py"),
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
