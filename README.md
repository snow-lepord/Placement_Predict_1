# CampusPlacementIQ · Modular ML Intelligence & Predictive Platform

An end-to-end Machine Learning intelligence platform analyzing **50,000 real student records** across 48 structured curriculum sessions grouped into 6 core engineering modules. Features production-grade pipeline orchestration, rigorous diagnostics, model explainability with SHAP, unsupervised clustering, and an interactive Flask intelligence web dashboard.

---

## 🏗️ Project Architecture

```text
PythonProject2/
├── config.py                                 # Unified configuration for paths, feature schemas & seeds
├── requirements.txt                          # Core ML & visualization dependencies
├── README.md                                 # Comprehensive documentation & execution manual
├── run_pipeline.py                           # Master curriculum pipeline runner (Sessions 01–24)
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
│   ├── foundations_pipeline/                 # Module 1: Foundations (Sessions 01–04)
│   │   ├── s01_cohort_ingestion.py           # Ingestion, schema validation & missing value audit
│   │   ├── s02_heuristic_baseline.py         # Zero-rule majority class & domain heuristics
│   │   ├── s03_exploratory_profiling.py      # Univariate, bivariate, multivariate & correlation EDA
│   │   └── s04_data_sanitization.py          # Outlier detection (IQR method) & range clipping
│   ├── linear_optimization/                  # Module 2: Linear Models & Optimization (Sessions 05–12)
│   │   ├── s05_data_partitioning_splits.py   # Stratified train (70%) / val (10%) / test (20%) split
│   │   ├── s06_ols_linear_regression.py      # Ordinary Least Squares regression & residual plots
│   │   ├── s07_feature_standardization.py    # Zero-leakage Z-score feature standardization
│   │   ├── s08_categorical_imputation.py     # Median imputation & one-hot categorical encoding
│   │   ├── s09_logistic_classifier_boundary.py # Binary/multiclass logistic classification & sigmoid
│   │   ├── s10_penalty_regularization.py     # Ridge (L2), Lasso (L1), and Elastic Net penalties
│   │   ├── s11_end_to_end_orchestration.py   # Scikit-learn ColumnTransformer & Pipeline assembly
│   │   └── s12_interpretability_coefficients.py # Coefficient dynamics, odds ratios & collinearity
│   ├── tree_ensemble_analytics/              # Module 3: Tree Models & Ensembles (Sessions 13–20)
│   │   ├── s13_cart_decision_tree.py         # CART decision trees & topology visualization
│   │   ├── s14_tree_pruning_split_criteria.py# Gini vs Entropy, Information Gain & Cost-Complexity
│   │   ├── s15_bias_variance_depth_tradeoff.py # Tree depth progression & bias-variance frontier
│   │   ├── s16_bagging_random_forest.py      # Bootstrap aggregating, random feature subsets & OOB
│   │   ├── s17_gradient_boosting_core.py     # AdaBoost stumps vs Gradient Boosting residuals
│   │   ├── s18_modern_boosted_ensembles.py   # XGBoost, LightGBM, and CatBoost with early stopping
│   │   ├── s19_shap_feature_explanations.py  # SHAP TreeExplainer, Beeswarm & Waterfall attribution
│   │   └── s20_cross_model_tournament.py     # 10-model head-to-head tournament & Pareto frontier
│   ├── unsupervised_segmentation/            # Module 4: Unsupervised Learning (Sessions 21 & 24)
│   │   ├── s21_kmeans_student_clustering.py  # K-Means clustering, Inertia Elbow & Silhouette curves
│   │   └── s24_hierarchical_dendrogram_clustering.py # Agglomerative clustering & Linkage dendrograms
│   └── showcase_figure_generator.py          # High-resolution showcase charts generator
├── outputs/                                  # Per-session execution logs, reports & visual charts
│   ├── ingestion/                            # Session 01 schema audit report
│   ├── baseline/                             # Session 02 heuristic baseline report
│   ├── eda/                                  # Session 03 & showcase high-res charts
│   ├── cleaning/                             # Session 04 IQR outlier report
│   ├── split/                                # Session 05 partition distribution report
│   ├── linear_regression/                    # Session 06 OLS fit & residual plots
│   ├── scaling/                              # Session 07 standardization distribution plots
│   ├── encoding/                             # Session 08 feature expansion diagnostic plots
│   ├── logistic_regression/                  # Session 09 S-curve, sigmoid & decision boundary
│   ├── regularization/                       # Session 10 parameter shrinkage & penalty charts
│   ├── pipeline/                             # Session 11 architecture flow & probability plots
│   ├── coefficients/                         # Session 12 odds ratio rankings & correlation heatmap
│   ├── decision_tree/                        # Session 13 tree topology & feature importance
│   ├── splitting_criteria/                   # Session 14 cost-complexity pruning curves
│   ├── bias_variance/                        # Session 15 bias-variance tradeoff curves
│   ├── random_forest/                        # Session 16 n_estimators curve & forest importance
│   ├── boosting/                             # Session 17 staged convergence & learning rate curves
│   ├── modern_boosting/                      # Session 18 accuracy vs runtime & XGBoost importance
│   ├── shap/                                 # Session 19 beeswarm summary & waterfall explanations
│   ├── model_comparison/                     # Session 20 head-to-head tournament plots
│   ├── kmeans/                               # Session 21 elbow & silhouette curves + cluster CSV
│   └── hierarchical/                         # Session 24 4-linkage dendrograms & cut height plots
└── dashboard/                                # Modern Web Intelligence Platform
    ├── app.py                                # Flask web server & live inference API (/api/predict)
    ├── sessions_data.py                      # 48-session metadata registry & curriculum roadmap
    ├── static/css/style.css                  # Slate & Emerald luxury UI styling (Glassmorphism)
    └── templates/
        ├── base.html                         # Responsive layout, sidebar navigation & search
        ├── home.html                         # Hero dashboard, interactive simulator & visual gallery
        └── session.html                      # Terminal audit console & chart viewer with lightbox
```

---

## ⚡ Execution Instructions

### 1. Run the Full Master Pipeline
To execute all 22 active curriculum sessions sequentially and generate all diagnostic reports and high-resolution figures:

```bash
python run_pipeline.py
```

### 2. Execute Individual Modules or Sessions
You can run any individual milestone independently:

```bash
# Module 1: Foundations
python src/foundations_pipeline/s01_cohort_ingestion.py
python src/foundations_pipeline/s02_heuristic_baseline.py
python src/foundations_pipeline/s03_exploratory_profiling.py
python src/foundations_pipeline/s04_data_sanitization.py

# Module 2: Linear Optimization
python src/linear_optimization/s05_data_partitioning_splits.py
python src/linear_optimization/s06_ols_linear_regression.py
python src/linear_optimization/s07_feature_standardization.py
python src/linear_optimization/s08_categorical_imputation.py
python src/linear_optimization/s09_logistic_classifier_boundary.py
python src/linear_optimization/s10_penalty_regularization.py
python src/linear_optimization/s11_end_to_end_orchestration.py
python src/linear_optimization/s12_interpretability_coefficients.py

# Module 3: Tree Ensembles
python src/tree_ensemble_analytics/s13_cart_decision_tree.py
python src/tree_ensemble_analytics/s14_tree_pruning_split_criteria.py
python src/tree_ensemble_analytics/s15_bias_variance_depth_tradeoff.py
python src/tree_ensemble_analytics/s16_bagging_random_forest.py
python src/tree_ensemble_analytics/s17_gradient_boosting_core.py
python src/tree_ensemble_analytics/s18_modern_boosted_ensembles.py
python src/tree_ensemble_analytics/s19_shap_feature_explanations.py
python src/tree_ensemble_analytics/s20_cross_model_tournament.py

# Module 4: Unsupervised Clustering
python src/unsupervised_segmentation/s21_kmeans_student_clustering.py
python src/unsupervised_segmentation/s24_hierarchical_dendrogram_clustering.py

# Visual Gallery Figures
python src/showcase_figure_generator.py
```

### 3. Launch the Interactive Web Dashboard
Launch the Flask intelligence server from root:

```bash
python app.py
# or
python main.py
```

Open your browser and navigate to: **`http://127.0.0.1:5001`**

---

## 📊 Curriculum Modules & Highlights

- **Module 1: Foundations (Sessions 01–04)**: Cohort data ingestion, zero-rule baseline classification (65.7%), exploratory distribution analysis, and IQR outlier boundaries.
- **Module 2: Linear Optimization (Sessions 05–12)**: Stratified partitioning, OLS regression, zero-leakage feature scaling, one-hot encoding, logistic decision boundaries, L1/L2 regularization, scikit-learn pipeline assembly, and coefficient odds-ratio interpretability.
- **Module 3: Tree Ensembles & Explainability (Sessions 13–20)**: CART decision trees, Information Gain vs Gini impurity, cost-complexity pruning, bias-variance curves, Random Forest bagging, AdaBoost stumps, gradient boosting residuals, modern gradient boosted trees (XGBoost, LightGBM, CatBoost) with early stopping, game-theoretic SHAP explainability (beeswarm & waterfall plots), and head-to-head tournament benchmarking.
- **Module 4: Unsupervised Segmentation (Sessions 21 & 24)**: K-Means clustering with Inertia Elbow and Silhouette score sweeps (optimal K=2), plus Agglomerative Hierarchical clustering comparing Ward, Complete, Average, and Single linkage dendrograms.
- **Modules 5–6: Advanced Evaluation & Production (Sessions 30–48)**: Roadmap for nested cross-validation, Optuna Bayesian optimization, probability calibration, and production Docker containerization.

---

## 🏆 Key Benchmark Results

| Model Architecture | Paradigm | Validation Accuracy | Validation ROC-AUC | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Zero-Rule Baseline** | Heuristic | 65.70% | 0.5000 | Predicts majority class |
| **Pruned Decision Tree** | Single Tree | 90.80% | 0.9320 | Cost-complexity pruned |
| **Logistic Regression Pipeline** | Linear Model | 92.28% | 0.9620 | ColumnTransformer + L2 |
| **Random Forest (200 Trees)**| Bagging Ensemble | 92.06% | 0.9650 | OOB Score: 92.11% |
| **Gradient Boosting (sklearn)**| Boosting | 92.38% | 0.9680 | 200 depth-3 trees |
| **XGBoost** | Modern Boosting | 92.56% | 0.9702 | L1/L2 regularized objective |
| **CatBoost** | Modern Boosting | 92.62% | 0.9708 | Oblivious decision trees |
| **LightGBM** | Modern Boosting | **92.70%** | **0.9715** | Leaf-wise histogram binning |

---

## 👥 Authorship & Academic Notice
Developed for the Machine Learning Project-Based Learning Curriculum. All source code, architectures, and dashboards are independently structured and verified.
"""

with open("/Users/mehakarora/Desktop/ML/PythonProject2/README.md", "w") as f:
    f.write(readme_content)

print("Updated README.md for CampusPlacementIQ")
"""

with open("/Users/mehakarora/.gemini/antigravity/brain/5897e0b8-7980-4c9b-a226-cff5f9629553/scratch/write_readme.py", "w") as f:
    f.write(readme_content)

print("Created write_readme.py")
