# CampusPlacementIQ · Modular ML Intelligence & Predictive Platform

An end-to-end Machine Learning intelligence platform analyzing **50,000 real student records** across 48 structured curriculum sessions grouped into 6 core engineering modules. Features production-grade pipeline orchestration, rigorous diagnostics, model explainability with SHAP, unsupervised clustering, and an interactive Flask intelligence web dashboard.

---

## 🏗️ Project Architecture

```text
PythonProject2/
├── config.py                                 # Unified configuration for paths, feature schemas & seeds
├── requirements.txt                          # Core ML & visualization dependencies
├── README.md                                 # Comprehensive documentation & execution manual
├── run_pipeline.py                           # Master curriculum pipeline runner (Sessions 01–25)
├── app.py                                    # Root web application runner (port 5001)
├── main.py                                   # Alternative entrypoint runner
├── data/
│   ├── raw/
│   │   └── placement_raw.csv                 # 50,000-record raw student cohort dataset
│   └── processed/
│       ├── placement_clean.csv               # Sanitized & IQR-audited dataset (Session 04)
│       └── splits/                           # Partitioned datasets across pipeline stages
│           ├── train.csv, val.csv, test.csv
│           ├── train_scaled.csv, val_scaled.csv, test_scaled.csv
│           └── train_ready.csv, val_ready.csv, test_ready.csv
├── src/
│   ├── module_01_foundations/                # Module 01: Foundations (Sessions 01–04)
│   │   ├── 01_cohort_ingestion.py            # Ingestion, schema validation & missing value audit
│   │   ├── 02_heuristic_baseline.py          # Zero-rule majority class & domain heuristics
│   │   ├── 03_exploratory_profiling.py       # Univariate, bivariate, multivariate & correlation EDA
│   │   └── 04_data_sanitization.py           # Outlier detection (IQR method) & range clipping
│   ├── module_02_linear_optimization/        # Module 02: Linear Models & Optimization (Sessions 05–12)
│   │   ├── 05_data_partitioning.py           # Stratified train (70%) / val (10%) / test (20%) split
│   │   ├── 06_ols_linear_regression.py       # Ordinary Least Squares regression & residual plots
│   │   ├── 07_feature_standardization.py     # Zero-leakage Z-score feature standardization
│   │   ├── 08_categorical_imputation.py      # Median imputation & one-hot categorical encoding
│   │   ├── 09_logistic_classification.py     # Binary/multiclass logistic classification & sigmoid
│   │   ├── 10_penalty_regularization.py      # Ridge (L2), Lasso (L1), and Elastic Net penalties
│   │   ├── 11_pipeline_orchestration.py      # Scikit-learn ColumnTransformer & Pipeline assembly
│   │   └── 12_coefficient_attribution.py     # Coefficient dynamics, odds ratios & collinearity
│   ├── module_03_tree_ensembles/             # Module 03: Tree Models & Ensembles (Sessions 13–20)
│   │   ├── 13_cart_decision_tree.py          # CART decision trees & topology visualization
│   │   ├── 14_tree_pruning_splitting.py      # Gini vs Entropy, Information Gain & Cost-Complexity
│   │   ├── 15_bias_variance_depth.py         # Tree depth progression & bias-variance frontier
│   │   ├── 16_random_forest_bagging.py       # Bootstrap aggregating, random feature subsets & OOB
│   │   ├── 17_sequential_boosting.py         # AdaBoost stumps vs Gradient Boosting residuals
│   │   ├── 18_modern_boosted_trees.py        # XGBoost, LightGBM, and CatBoost with early stopping
│   │   ├── 19_shap_explainability.py         # SHAP TreeExplainer, Beeswarm & Waterfall attribution
│   │   └── 20_cross_model_tournament.py      # 10-model head-to-head tournament & Pareto frontier
│   ├── module_04_unsupervised_learning/      # Module 04: Unsupervised Learning & Clustering (Sessions 21–29)
│   │   ├── 21_kmeans_clustering.py           # K-Means clustering, Inertia Elbow & Silhouette curves
│   │   ├── 23_pca_decomposition.py           # PCA variance curves, loadings heatmap & 2D projections
│   │   ├── 24_hierarchical_clustering.py     # Agglomerative clustering & Linkage dendrograms
│   │   ├── 25_dbscan_density.py              # Density-based DBSCAN clustering & noise isolation
│   │   ├── 26_tsne_manifold.py               # t-SNE non-linear projection & perplexity dynamics
│   │   ├── 27_umap_manifold.py               # UMAP Riemannian manifold approximation & benchmarking
│   │   ├── 28_anomaly_detection.py           # Isolation Forest & LOF unsupervised trajectory auditing
│   │   └── 29_cluster_augmented_features.py  # Cluster-augmented semi-supervised feature engineering
│   ├── module_05_model_evaluation/           # Module 05: Model Evaluation, Validation & Optimization (Sessions 30–33)
│   │   ├── 30_split_discipline_leakage.py    # 80/20 train/test split discipline & data leakage vectors
│   │   ├── 31_cross_validation_schemes.py    # K-Fold & Stratified CV with standard error metrics
│   │   ├── 32_pipeline_test_evaluation.py    # Leakage-free Sklearn Pipeline & 10-student case study
│   │   └── 33_nested_stratified_cv.py        # Dual-loop nested CV & hyperparameter selection bias
│   └── module_05_visual_analytics/           # Visual Gallery Figures
│       └── showcase_figure_generator.py      # High-resolution showcase charts generator
├── outputs/                                  # Structured per-module execution logs & visual charts
│   ├── module_01_foundations/                # Module 01 outputs (01_ingestion ... 04_cleaning)
│   ├── module_02_linear_optimization/        # Module 02 outputs (05_split ... 12_coefficients)
│   ├── module_03_tree_ensembles/             # Module 03 outputs (13_decision_tree ... 20_model_comparison)
│   ├── module_04_unsupervised_learning/      # Module 04 outputs (21_kmeans ... 29_cluster_features)
│   └── module_05_model_evaluation/           # Module 05 outputs (30_split ... 33_nested_cv)

└── dashboard/                                # Modern Web Intelligence Platform
    ├── app.py                                # Flask server, live inference API & Module 5 routes
    ├── sessions_data.py                      # 48-session metadata registry & curriculum roadmap
    ├── static/css/style.css                  # Slate & Emerald luxury UI styling (Glassmorphism)
    └── templates/
        ├── base.html                         # Responsive layout, sidebar navigation & search
        ├── home.html                         # Hero dashboard, interactive simulator & visual gallery
        ├── session.html                      # Terminal audit console & chart viewer with lightbox
        ├── dataset.html                      # 50,000-student cohort explorer & paginator
        └── module5.html                      # Interactive Module 5 Evaluation Workbench & Simulator
```

---

## ⚡ Execution Instructions

### 1. Run the Full Master Pipeline
To execute all 26 active curriculum sessions sequentially and generate all diagnostic reports and high-resolution figures:

```bash
python run_pipeline.py
```

### 2. Execute Individual Modules or Sessions
You can run any individual milestone independently:

```bash
# Module 01: Foundations
python src/module_01_foundations/01_cohort_ingestion.py
python src/module_01_foundations/02_heuristic_baseline.py
python src/module_01_foundations/03_exploratory_profiling.py
python src/module_01_foundations/04_data_sanitization.py

# Module 02: Linear Optimization
python src/module_02_linear_optimization/05_data_partitioning.py
python src/module_02_linear_optimization/06_ols_linear_regression.py
python src/module_02_linear_optimization/07_feature_standardization.py
python src/module_02_linear_optimization/08_categorical_imputation.py
python src/module_02_linear_optimization/09_logistic_classification.py
python src/module_02_linear_optimization/10_penalty_regularization.py
python src/module_02_linear_optimization/11_pipeline_orchestration.py
python src/module_02_linear_optimization/12_coefficient_attribution.py

# Module 03: Tree Ensembles
python src/module_03_tree_ensembles/13_cart_decision_tree.py
python src/module_03_tree_ensembles/14_tree_pruning_splitting.py
python src/module_03_tree_ensembles/15_bias_variance_depth.py
python src/module_03_tree_ensembles/16_random_forest_bagging.py
python src/module_03_tree_ensembles/17_sequential_boosting.py
python src/module_03_tree_ensembles/18_modern_boosted_trees.py
python src/module_03_tree_ensembles/19_shap_explainability.py
python src/module_03_tree_ensembles/20_cross_model_tournament.py

# Module 04: Unsupervised Learning
python src/module_04_unsupervised_learning/21_kmeans_clustering.py
python src/module_04_unsupervised_learning/23_pca_decomposition.py
python src/module_04_unsupervised_learning/24_hierarchical_clustering.py
python src/module_04_unsupervised_learning/25_dbscan_density.py
python src/module_04_unsupervised_learning/26_tsne_manifold.py
python src/module_04_unsupervised_learning/27_umap_manifold.py
python src/module_04_unsupervised_learning/28_anomaly_detection.py
python src/module_04_unsupervised_learning/29_cluster_augmented_features.py

# Module 05: Model Evaluation & Optimization
python src/module_05_model_evaluation/30_split_discipline_leakage.py
python src/module_05_model_evaluation/31_cross_validation_schemes.py
python src/module_05_model_evaluation/32_pipeline_test_evaluation.py
python src/module_05_model_evaluation/33_nested_stratified_cv.py

# Showcase Visual Gallery
python src/module_05_visual_analytics/showcase_figure_generator.py
```

### 3. Launch the Interactive Web Dashboard
Launch the Flask intelligence server from root:

```bash
python app.py
# or
python main.py
```

Open your browser and navigate to:
- **Platform Overview:** `http://127.0.0.1:5001/`
- **Module 5 Workbench:** `http://127.0.0.1:5001/module5`
- **Cohort Dataset:** `http://127.0.0.1:5001/dataset`

---

## 📊 Curriculum Modules & Highlights

- **Module 1: Foundations (Sessions 01–04)**: Cohort data ingestion, zero-rule baseline classification (65.7%), exploratory distribution analysis, and IQR outlier boundaries.
- **Module 2: Linear Optimization (Sessions 05–12)**: Stratified partitioning, OLS regression, zero-leakage feature scaling, one-hot encoding, logistic decision boundaries, L1/L2 regularization, scikit-learn pipeline assembly, and coefficient odds-ratio interpretability.
- **Module 3: Tree Ensembles & Explainability (Sessions 13–20)**: CART decision trees, Information Gain vs Gini impurity, cost-complexity pruning, bias-variance curves, Random Forest bagging, AdaBoost stumps, gradient boosting residuals, modern gradient boosted trees (XGBoost, LightGBM, CatBoost) with early stopping, game-theoretic SHAP explainability (beeswarm & waterfall plots), and head-to-head tournament benchmarking.
- **Module 4: Unsupervised Segmentation & Manifold Learning (Sessions 21–29)**: K-Means clustering with Inertia Elbow and Silhouette sweeps (optimal K=2), PCA full-rank coordinate decomposition (PC1+PC2 capture 76.2% variance), Agglomerative Hierarchical clustering comparing 4 linkage dendrograms, non-parametric DBSCAN density clustering, t-SNE non-linear projection across perplexity sweeps, UMAP Riemannian manifold approximation with out-of-sample projection, unsupervised Isolation Forest and LOF anomaly detection (0.9535 ROC-AUC), and cluster-augmented semi-supervised feature engineering boosting tree splits.
- **Module 5: Model Evaluation, Validation & Optimization (Sessions 30–33)**: 80/20 train/test split discipline, mathematical disjoint set conditions, empirical taxonomy of 3 data leakage vectors, Stratified K-Fold CV with Standard Error (SE) confidence bounds, leakage-free Scikit-Learn Pipelines (`StandardScaler` + `LogisticRegression`), 10-student placement case study (`ExampleDataSetPP.csv`) evaluated on vault-locked test students S09 & S10 (100% test accuracy), and dual-loop Nested Cross-Validation mitigating hyperparameter selection bias.


---

## 👥 Authorship & Academic Notice
Developed for the Machine Learning Project-Based Learning Curriculum. All source code, architectures, and dashboards are independently structured and verified.
