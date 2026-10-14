"""
Curriculum Metadata & Roadmap Registry for CampusPlacementIQ Platform.
Tracks all 48 Sessions across 6 core modules, mapping reports, visual charts,
and deep conceptual breakdowns explaining the ML theory, methodology, and findings.
"""

MODULES = [
    {"id": 1, "name": "Foundations of ML & Data Analysis", "sessions": [1, 2, 3, 4]},
    {"id": 2, "name": "Linear Models & Optimization", "sessions": [5, 6, 7, 8, 9, 10, 11, 12]},
    {"id": 3, "name": "Tree-Based Models & Ensembles", "sessions": [13, 14, 15, 16, 17, 18, 19, 20]},
    {"id": 4, "name": "Unsupervised Learning & Clustering", "sessions": [21, 22, 23, 24, 25, 26, 27, 28, 29]},
    {"id": 5, "name": "Model Evaluation, Validation & Optimization", "sessions": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]},
    {"id": 6, "name": "ML Engineering & Production Serving", "sessions": [40, 41, 42, 43, 44, 45, 46, 47, 48]},
]

SESSIONS = {
    # -------------------------------------------------------------
    # Module 1: Foundations of ML & Data Analysis
    # -------------------------------------------------------------
    1: {
        "title": "Data Ingestion & Cohort Inspection",
        "built": True,
        "folder": "ingestion",
        "report": "ingestion_report.txt",
        "plots": [],
        "summary": "Ingests the 50,000-student raw cohort dataset (placement_raw.csv), inspects column schemas, evaluates memory footprint, audits missing value distributions, and verifies baseline class balance.",
        "key_concepts": ["Schema Validation", "Cohort Profiling", "Missing Data Auditing", "Target Class Distribution"],
        "methodology": [
            "Loaded 50,000 rows across 32 continuous and categorical features via Pandas",
            "Audited column memory usage (31.03 MB in memory)",
            "Identified 19,976 missing values across 5 elective columns (Workshops, Aptitude, Soft Skills, Coding, Mock Interviews)",
            "Audited target class balance: 32,856 placed (65.71%) vs 17,144 unplaced (34.29%)"
        ],
        "findings": "The cohort represents a realistic university placement drive with 65.71% positive placement rate and 5 elective features containing Missing at Random (MAR) patterns."
    },
    2: {
        "title": "Baseline Heuristic Model & Serving",
        "built": True,
        "folder": "baseline",
        "report": "baseline_report.txt",
        "plots": [],
        "summary": "Establishes the non-trivial lower-bound benchmark by predicting the majority class (Placed = 1) for every candidate. Any machine learning model must convincingly surpass this threshold to demonstrate true predictive value.",
        "key_concepts": ["Zero-Rule Classifier (ZeroR)", "Baseline Benchmark", "Majority Class Imbalance", "Trivial Accuracy Fallacy"],
        "methodology": [
            "Assigned constant prediction (y = 1) to all 50,000 students in the cohort",
            "Computed baseline accuracy, precision, recall, and F1-score",
            "Evaluated Area Under ROC Curve (ROC-AUC) for the constant heuristic"
        ],
        "findings": "Zero-Rule benchmark achieves 65.71% accuracy with ROC-AUC of 0.5000. All subsequent ML models must exceed 65.71% accuracy to justify complexity."
    },
    3: {
        "title": "Exploratory Data Analysis & Diagnostic Profiling",
        "built": True,
        "folder": "eda",
        "report": None,
        "plots": [
            "pie_placement_distribution.png",
            "donut_college_tier.png",
            "pie_stream_breakdown.png",
            "bar_placement_by_tier.png",
            "bar_stream_average_salary.png",
            "bar_backlogs_vs_placement.png",
            "03_univariate_numeric.png",
            "03_univariate_categorical.png",
            "03_bivariate_numeric_vs_target.png",
            "03_bivariate_categorical_vs_target.png",
            "03_bivariate_numeric_vs_numeric.png",
            "03_multivariate.png",
            "03_correlation_heatmap.png",
            "sgpa_trajectory_trend.png"
        ],
        "summary": "Conducts deep visual and statistical exploratory profiling across univariate distributions, academic stream breakdowns, college tier disparities, and bivariate correlations with placement outcomes.",
        "key_concepts": ["Univariate KDE Histograms", "Bivariate Target Segmentation", "Multicollinearity Diagnostics", "Semester SGPA Trajectories"],
        "methodology": [
            "Generated 14 multi-panel statistical visualization figures with Seaborn and Matplotlib",
            "Plotted correlation heatmaps across continuous academic metrics (SGPA 1-8, CGPA, Aptitude, Coding)",
            "Tracked student GPA trends across all 8 semesters partitioned by placement status"
        ],
        "findings": "CGPA (mean 7.7) and CodingTestScore exhibit strong positive correlation with placement. Active backlogs cut placement probability by over 40%."
    },
    4: {
        "title": "Data Sanitization & IQR Outlier Auditing",
        "built": True,
        "folder": "cleaning",
        "report": "cleaning_report.txt",
        "plots": [],
        "summary": "Executes data quality sanitization: audits duplicate records, detects continuous score anomalies via Interquartile Range (IQR), applies domain boundary clipping (0-100), and normalizes string casing.",
        "key_concepts": ["Interquartile Range (IQR)", "Outlier Auditing", "Domain Boundary Enforcement", "String Normalization"],
        "methodology": [
            "Verified duplicate row count across all 50,000 student entries",
            "Computed Q1, Q3, and IQR thresholds across test scores and CGPA",
            "Clamped percentage scores to strict valid [0, 100] domain bounds",
            "Cleaned and title-cased nominal text columns (Stream, City, Gender, CollegeTier)"
        ],
        "findings": "Zero duplicate records detected. Identified minor upper/lower outlier tails in assessment scores and normalized categorical strings into placement_clean.csv."
    },

    # -------------------------------------------------------------
    # Module 2: Linear Models & Optimization
    # -------------------------------------------------------------
    5: {
        "title": "Train, Validation & Test Splitting",
        "built": True,
        "folder": "split",
        "report": "split_report.txt",
        "plots": [],
        "summary": "Partitions the sanitized 50,000-student cohort into strict 70% Train (35,000), 15% Validation (7,500), and 15% Test (7,500) subsets using stratified sampling to guarantee identical placement ratios.",
        "key_concepts": ["Stratified Data Partitioning", "Generalization Set Isolation", "Target Ratio Preservation", "Evaluation Integrity"],
        "methodology": [
            "Executed two-stage stratified train_test_split with fixed random seed (42)",
            "Allocated 35,000 records for model optimization, 7,500 for tuning, 7,500 for final audit",
            "Audited exact 65.71% positive class proportion across all three independent partitions"
        ],
        "findings": "Stratification perfectly maintained the 65.71% placement ratio across train, val, and test splits without row overlap."
    },
    6: {
        "title": "Linear Regression & Cost Minimization",
        "built": True,
        "folder": "linear_regression",
        "report": "linear_regression_report.txt",
        "plots": [
            "06_linear_regression_fit.png",
            "06_residual_plot.png"
        ],
        "summary": "Models the continuous relationship between pre-placement Aptitude Test Scores and technical Coding Test Performance using Ordinary Least Squares (OLS) closed-form minimization.",
        "key_concepts": ["Ordinary Least Squares (OLS)", "Residual Diagnostics", "Homoscedasticity", "Coefficient of Determination (R²)", "Root Mean Squared Error (RMSE)"],
        "methodology": [
            "Fit OLS linear regression model to predict coding competency from general aptitude",
            "Computed residual errors (y_true - y_pred) and checked homoscedasticity assumptions",
            "Evaluated goodness-of-fit R² and residual standard error metrics"
        ],
        "findings": "OLS linear regression achieved R² = 0.657 with normally distributed residuals around zero, confirming aptitude strongly predicts coding scores."
    },
    7: {
        "title": "Feature Scaling & Leakage Prevention",
        "built": True,
        "folder": "scaling",
        "report": "scaling_report.txt",
        "plots": [
            "07_scaling_distribution_comparison.png"
        ],
        "summary": "Standardizes numeric features to zero mean and unit variance (Z-Score). Adheres strictly to the cardinal rule of ML: fitting transformation parameters exclusively on the training split to avoid data leakage.",
        "key_concepts": ["Z-Score Standardization", "Scale Invariance", "Data Leakage Prevention", "Out-of-Sample Transform"],
        "methodology": [
            "Computed mean and standard deviation strictly on the 35,000-row training partition",
            "Applied transformation Z = (X - μ) / σ outward to train, val, and test sets",
            "Compared original vs standardized probability density distributions"
        ],
        "findings": "Eliminated magnitude dominance (e.g. Attendance vs CGPA) while preventing leakage. Features now share μ = 0.0 and σ = 1.0."
    },
    8: {
        "title": "Imputation & Categorical Encoding",
        "built": True,
        "folder": "encoding",
        "report": "encoding_imputation_report.txt",
        "plots": [
            "08_imputation_and_features.png"
        ],
        "summary": "Resolves missing values across the 5 elective features via training-set medians, maps ordinal tiers (CollegeTier, CGPA_Tier), converts binary flags, and one-hot encodes nominal categoricals.",
        "key_concepts": ["Median Imputation", "Ordinal Integer Encoding", "One-Hot Encoding (OHE)", "Dimensionality Expansion"],
        "methodology": [
            "Filled missing cells using medians calculated strictly on train split",
            "Mapped ordinal tiers: Tier1 = 2, Tier2 = 1, Tier3 = 0",
            "Applied one-hot encoding across Stream, City, and Gender",
            "Exported 100% complete train_ready.csv, val_ready.csv, and test_ready.csv"
        ],
        "findings": "Expanded feature space from 32 to 45 encoded dimensions with exactly zero remaining missing values across all training splits."
    },
    9: {
        "title": "Logistic Regression & Decision Boundaries",
        "built": True,
        "folder": "logistic_regression",
        "report": "logistic_regression_report.txt",
        "plots": [
            "cgpa_vs_placement_curve.png",
            "sigmoid_function.png",
            "decision_boundary.png"
        ],
        "summary": "Trains a parametric binary logistic classifier using the sigmoid function to map linear combinations of student features into well-calibrated placement probabilities.",
        "key_concepts": ["Sigmoid Activation", "Binary Cross-Entropy (Log-Loss)", "Decision Boundary Hyperplane", "Odds Ratio Interpretation"],
        "methodology": [
            "Fit Scikit-Learn LogisticRegression with L-BFGS solver on 35,000 training examples",
            "Evaluated classification report (Precision, Recall, F1) on 7,500 validation students",
            "Visualized 2D decision boundary projection across CGPA and Coding Score"
        ],
        "findings": "Achieved 92.28% validation accuracy and 0.9620 ROC-AUC, outperforming the heuristic baseline by +26.57 percentage points."
    },
    10: {
        "title": "Regularization (Ridge, Lasso & Elastic Net)",
        "built": True,
        "folder": "regularization",
        "report": "regularization_report.txt",
        "plots": [
            "regression_coefficients.png",
            "logistic_penalty_comparison.png"
        ],
        "summary": "Investigates how L1 (Lasso) and L2 (Ridge) norm penalties constrain model coefficient magnitudes, mitigate multicollinearity among semester SGPAs, and perform automatic feature selection.",
        "key_concepts": ["L1 Regularization (Lasso / Sparsity)", "L2 Regularization (Ridge / Weight Shrinkage)", "Elastic Net Blend", "Penalty Parameter C"],
        "methodology": [
            "Swept inverse regularization strength C from 0.001 to 10.0 across L1 and L2 penalties",
            "Measured coefficient shrinkage and tracked zeroed-out weights under L1 penalty",
            "Evaluated impact on validation accuracy and log-loss"
        ],
        "findings": "L2 penalty achieved optimal validation accuracy (92.28%), while L1 penalty successfully zeroed out 8 redundant semester SGPA weights with negligible accuracy loss."
    },
    11: {
        "title": "End-to-End Scikit-Learn Pipeline Orchestration",
        "built": True,
        "folder": "pipeline",
        "report": "pipeline_report.txt",
        "plots": [
            "pipeline_prediction_distribution.png",
            "pipeline_architecture_flow.png"
        ],
        "summary": "Encapsulates the complete preprocessing and modeling workflow (imputation, scaling, one-hot encoding, and logistic regression) into a unified, atomic Scikit-Learn Pipeline.",
        "key_concepts": ["Composite Pipeline", "ColumnTransformer", "Atomic Preprocessing", "Production Deployment Readiness"],
        "methodology": [
            "Assembled ColumnTransformer handling numeric and categorical sub-pipelines",
            "Connected transformer directly to LogisticRegression estimator",
            "Fit entire pipeline on raw training data and evaluated on raw validation split"
        ],
        "findings": "The atomic pipeline reproduces 92.28% validation accuracy on raw data in a single transform-and-predict call, eliminating train-test leakage risk."
    },
    12: {
        "title": "Model Interpretability & Coefficient Dynamics",
        "built": True,
        "folder": "coefficients",
        "report": "coefficients_report.txt",
        "plots": [
            "top_coefficients.png",
            "feature_importance_coefficients.png",
            "sgpa_cgpa_multicollinearity_heatmap.png"
        ],
        "summary": "Extracts, exponentiates, and interprets the learned weights of the linear classifier to determine which academic, technical, and extracurricular factors most strongly influence placement.",
        "key_concepts": ["Odds Ratios (e^β)", "Feature Attribution", "Multicollinearity Variance Inflation", "Predictor Ranking"],
        "methodology": [
            "Extracted all 44 logistic regression coefficients and computed odds ratios",
            "Ranked features by magnitude of positive and negative influence",
            "Analyzed multicollinearity correlations between semester SGPAs and overall CGPA"
        ],
        "findings": "CGPA (β = +1.84) and CodingTestScore (β = +1.21) are the strongest positive placement drivers; HistoryOfBacklogs (β = -1.42) is the single largest penalty."
    },

    # -------------------------------------------------------------
    # Module 3: Tree-Based Models & Ensembles
    # -------------------------------------------------------------
    13: {
        "title": "Decision Tree Classification & Non-Linear Splitting",
        "built": True,
        "folder": "decision_tree",
        "report": "decision_tree_report.txt",
        "plots": [
            "tree_visualization.png",
            "feature_importance.png"
        ],
        "summary": "Introduces non-parametric classification via the CART algorithm, recursively partitioning student feature space along orthogonal splits to minimize Gini impurity.",
        "key_concepts": ["CART Algorithm", "Gini Impurity", "Recursive Binary Partitioning", "Overfitting in Unconstrained Trees"],
        "methodology": [
            "Fit unconstrained DecisionTreeClassifier on 35,000 student records",
            "Rendered multi-level tree split diagrams showing threshold criteria",
            "Evaluated training accuracy vs validation accuracy gap"
        ],
        "findings": "Unconstrained decision tree achieved 99.8% training accuracy but only 87.4% validation accuracy—a dramatic demonstration of severe tree overfitting."
    },
    14: {
        "title": "Tree Pruning & Splitting Criteria",
        "built": True,
        "folder": "splitting_criteria",
        "report": "splitting_criteria_report.txt",
        "plots": [
            "pruning_curve.png"
        ],
        "summary": "Applies Minimal Cost-Complexity Pruning via the ccp_alpha penalty to remove overfitted subtrees, and benchmarks Gini Impurity against Shannon Entropy / Information Gain.",
        "key_concepts": ["Cost-Complexity Pruning (ccp_alpha)", "Shannon Entropy", "Information Gain", "Subtree Regularization"],
        "methodology": [
            "Computed the cost-complexity pruning path and extracted effective alphas",
            "Trained decision trees across candidate alphas to identify the validation peak",
            "Walked through a 14-student hand-worked Information Gain split calculation"
        ],
        "findings": "Optimal ccp_alpha = 0.00022 reduced tree depth from 24 down to 9 and boosted validation accuracy from 87.4% to 90.80%."
    },
    15: {
        "title": "Bias-Variance Tradeoff Analysis",
        "built": True,
        "folder": "bias_variance",
        "report": "bias_variance_report.txt",
        "plots": [
            "bias_variance_curve.png"
        ],
        "summary": "Sweeps maximum tree depth from 1 to 25 to visualize the fundamental machine learning trade-off between underfitting (high bias) and overfitting (high variance).",
        "key_concepts": ["Bias-Variance Tradeoff", "Model Capacity", "Generalization Frontier", "Optimal Stopping Depth"],
        "methodology": [
            "Trained 25 distinct decision tree models with max_depth spanning 1 to 25",
            "Logged training and validation accuracy and log-loss curves",
            "Identified the depth inflection point where validation accuracy begins degrading"
        ],
        "findings": "Optimal tree depth is exactly 9. Beyond depth 9, training accuracy climbs toward 100% while validation accuracy plateaus and declines, proving variance inflation."
    },
    16: {
        "title": "Random Forests & Bagging Ensembles",
        "built": True,
        "folder": "random_forest",
        "report": "random_forest_report.txt",
        "plots": [
            "n_estimators_curve.png",
            "feature_importance.png"
        ],
        "summary": "Combines bootstrap aggregation (bagging) with random feature sub-sampling (sqrt(p)) to train an ensemble of 200 decorrelated decision trees, evaluating out-of-bag (OOB) error.",
        "key_concepts": ["Bootstrap Aggregation (Bagging)", "Feature Sub-Sampling", "Tree Decorrelation", "Out-of-Bag (OOB) Score"],
        "methodology": [
            "Trained RandomForestClassifier with 200 trees and bootstrap sampling",
            "Swept n_estimators from 10 to 300 to trace ensemble performance convergence",
            "Extracted Mean Decrease in Impurity (MDI) feature importances"
        ],
        "findings": "Random Forest achieved 92.06% validation accuracy and 92.11% OOB accuracy, dramatically suppressing single-tree variance."
    },
    17: {
        "title": "Boosting Fundamentals & AdaBoost",
        "built": True,
        "folder": "boosting",
        "report": "boosting_report.txt",
        "plots": [
            "staged_accuracy_curve.png",
            "learning_rate_curve.png"
        ],
        "summary": "Explores sequential ensemble boosting where each base learner focuses on the misclassified examples of previous iterations, contrasting AdaBoost with Gradient Boosting.",
        "key_concepts": ["Sequential Boosting", "Exponential Loss Function", "Sample Weight Re-weighting", "Decision Stumps"],
        "methodology": [
            "Trained AdaBoostClassifier using 200 depth-1 decision stumps",
            "Tracked staged validation scores across each sequential boosting iteration",
            "Evaluated the impact of learning rate shrinkage on convergence stability"
        ],
        "findings": "AdaBoost achieved 91.80% validation accuracy, proving that hundreds of weak learners (stumps) can aggregate into a highly capable non-linear classifier."
    },
    18: {
        "title": "Gradient Boosted Trees (XGBoost / LightGBM / CatBoost)",
        "built": True,
        "folder": "modern_boosting",
        "report": "modern_boosting_report.txt",
        "plots": [
            "accuracy_vs_time.png",
            "xgboost_feature_importance.png"
        ],
        "summary": "Side-by-side engineering benchmark of modern gradient boosting libraries: XGBoost (exact/approximate split), LightGBM (leaf-wise histogram binning), and CatBoost (ordered boosting).",
        "key_concepts": ["Second-Order Taylor Approximation", "Histogram Feature Binning", "Leaf-Wise Tree Growth", "Oblivious Symmetric Trees", "Early Stopping"],
        "methodology": [
            "Trained XGBoost, LightGBM, and CatBoost with 50-round early stopping on validation ROC-AUC",
            "Benchmarked wall-clock training latency, peak memory footprint, and final accuracy",
            "Extracted gain-based feature importances across all three libraries"
        ],
        "findings": "LightGBM achieved the highest accuracy in the curriculum (92.70% accuracy, 0.9715 ROC-AUC) in just 0.42 seconds, outperforming XGBoost (92.56%) and CatBoost (92.62%)."
    },
    19: {
        "title": "Feature Importance & SHAP Values",
        "built": True,
        "folder": "shap",
        "report": "shap_report.txt",
        "plots": [
            "shap_summary_beeswarm.png",
            "importance_comparison.png",
            "shap_waterfall_example.png"
        ],
        "summary": "Applies game-theoretic Shapley Additive Explanations (SHAP via TreeExplainer) to provide mathematically sound, consistent local and global feature attribution for ensemble predictions.",
        "key_concepts": ["Shapley Values", "Game-Theoretic Feature Attribution", "TreeExplainer", "Global Beeswarm Plot", "Local Waterfall Breakdown"],
        "methodology": [
            "Fit SHAP TreeExplainer on the best-performing LightGBM ensemble",
            "Computed exact Shapley values across 1,000 validation student profiles",
            "Generated global Beeswarm summary plots and individual student Waterfall explainability diagrams"
        ],
        "findings": "CGPA exhibits the highest mean absolute SHAP value (|phi| = 1.45), followed by CodingTestScore (|phi| = 0.92). Non-linear inflection confirms CGPA >= 7.5 yields massive placement probability surge."
    },
    20: {
        "title": "Cross-Model Benchmark Tournament",
        "built": True,
        "folder": "model_comparison",
        "report": "model_comparison_report.txt",
        "plots": [
            "all_models_accuracy.png",
            "head_to_head.png"
        ],
        "summary": "Comprehensive 10-model tournament benchmarking all supervised algorithms built across Sessions 01–19 on accuracy, ROC-AUC, F1-score, log-loss, and inference latency on the unseen test set.",
        "key_concepts": ["Model Selection Matrix", "Pareto Frontier (Accuracy vs Latency)", "Multi-Metric Evaluation", "Curriculum Culmination"],
        "methodology": [
            "Evaluated 10 distinct models on the held-out 7,500-student test partition",
            "Compiled cross-model comparison matrix ranking accuracy, ROC-AUC, log loss, and runtime",
            "Generated bar chart rankings and head-to-head radar plots"
        ],
        "findings": "LightGBM ranks #1 overall (92.70% accuracy, 0.9715 ROC-AUC), closely followed by CatBoost (92.62%) and XGBoost (92.56%), with sub-millisecond inference latency."
    },

    # -------------------------------------------------------------
    # Module 4: Unsupervised Learning & Clustering
    # -------------------------------------------------------------
    21: {
        "title": "K-Means Clustering & Student Segmentation",
        "built": True,
        "folder": "kmeans",
        "report": "kmeans_report.txt",
        "plots": [
            "elbow_and_silhouette_curves.png"
        ],
        "summary": "Performs unsupervised student cohort segmentation without label supervision, sweeping K from 2 to 10 to evaluate Within-Cluster Sum of Squares (Inertia) and Silhouette coefficients.",
        "key_concepts": ["K-Means Clustering", "Inertia (Elbow Method)", "Silhouette Coefficient", "Centroid Profiling", "Unsupervised Discovery"],
        "methodology": [
            "Fit KMeans across K = 2 through 10 on standardized continuous features",
            "Plotted dual Inertia Elbow and Silhouette score evaluation curves",
            "Profiled cluster centroid means to characterize discovered student archetypes"
        ],
        "findings": "Optimal clustering occurs at K = 2 (Silhouette: 0.352), naturally separating high-performing placement candidates (89% placed) from lower-performing candidates (31% placed) without label access."
    },
    22: {
        "title": "K-Means++ & Scalability Optimization",
        "built": True,
        "folder": "kmeans_plusplus",
        "report": "kmeans_plusplus_report.txt",
        "plots": [
            "kmeans_pp_vs_random_stability.png",
            "minibatch_scalability_benchmark.png"
        ],
        "summary": "Explores D²-weighted probabilistic centroid initialization (K-Means++) to prevent poor local minima traps and accelerate Lloyd's convergence, alongside Mini-Batch K-Means scalability benchmarking across 35,000 student profiles.",
        "key_concepts": ["K-Means++ D² Probabilistic Seeding", "Mini-Batch K-Means (batch=512)", "Local Minima Escape", "Lloyd Convergence Speed", "Computational Complexity O(NKTd)"],
        "methodology": [
            "Benchmarked uniform random initialization vs K-Means++ across 25 independent random seeds at K=5 on 10,000 standardized student records",
            "Measured final Within-Cluster Sum of Squares (Inertia) dispersion and Lloyd's iterations to centroid equilibrium",
            "Scaled cohort size from N = 2,000 to N = 35,000 comparing full-batch K-Means against Mini-Batch K-Means (batch_size = 512)",
            "Quantified per-student WCSS distortion drift and Adjusted Rand Index (ARI) partition agreement"
        ],
        "findings": "K-Means++ consistently avoids high-inertia local minima traps and reduces Lloyd's iterations to convergence, while Mini-Batch K-Means achieves a massive wall-clock speedup at N = 35,000 with negligible per-sample distortion drift and high ARI partition alignment."
    },
    23: {
        "title": "Principal Component Analysis & Variance Decomposition",
        "built": True,
        "folder": "pca",
        "report": "pca_report.txt",
        "plots": [
            "scree_and_cumulative_variance.png",
            "loadings_biplot.png",
            "pca_2d_by_placement.png"
        ],
        "summary": "Investigates orthogonal coordinate rotation via PCA across 20 standardized continuous attributes, quantifying variance decay, scree elbow inflection, feature loadings biplots, and validating 2D manifold projection.",
        "key_concepts": ["Principal Component Analysis (PCA)", "Orthogonal Coordinate Rotation", "Explained Variance Ratio", "Scree Plot & Cumulative Thresholds", "Feature Loadings Biplot"],
        "methodology": [
            "Fit full-rank PCA (20 components) on 35,000 median-imputed, standardized student profiles",
            "Traced scree plot and cumulative variance across all 20 principal components",
            "Extracted eigenvector loadings for PC1 and PC2 with vector biplot visualization",
            "Projected 50,000 students onto the 2D PC1-PC2 manifold colored by actual placement ground truth"
        ],
        "findings": "PC1 captures 64.25% of cohort variance (driven by academic grades: SGPAs & CGPA), while PC2 captures 11.90% (driven by coding and aptitude scores). Together, PC1 + PC2 capture 76.15% of total variance, validating 2D cluster visualizations."
    },
    24: {
        "title": "Hierarchical Agglomerative Clustering",
        "built": True,
        "folder": "hierarchical",
        "report": "hierarchical_report.txt",
        "plots": [
            "dendrograms_by_linkage.png",
            "silhouette_by_cut_height.png"
        ],
        "summary": "Builds bottom-up agglomerative hierarchical cluster trees, comparing Ward's minimum variance, Complete, Average, and Single linkage criteria via dendrogram tree structures.",
        "key_concepts": ["Agglomerative Hierarchical Clustering", "Linkage Criteria (Ward, Complete, Average, Single)", "Dendrogram Cut Height", "Adjusted Rand Index (ARI)"],
        "methodology": [
            "Computed pairwise Euclidean distance matrix on 1,000-student cohort sample",
            "Constructed dendrogram visualizations across Ward, Complete, Average, and Single linkage",
            "Evaluated cluster alignment against K-Means via Adjusted Rand Index (ARI)"
        ],
        "findings": "Ward linkage produced the most compact, balanced student clusters; dendrogram cut height of h = 38 yielded 2 clusters with ARI = 0.81 alignment to K-Means."
    },
    25: {
        "title": "DBSCAN Density-Based Clustering & Anomaly Detection",
        "built": True,
        "folder": "dbscan",
        "report": "dbscan_report.txt",
        "plots": [
            "k_distance_plot.png",
            "dbscan_clusters_pca_2d.png"
        ],
        "summary": "Executes Density-Based Spatial Clustering (DBSCAN) across 20 continuous attributes, discovering dense student cohorts without pre-specifying K, finding non-spherical clusters, and isolating anomalous noise points.",
        "key_concepts": ["Density Reachability & Connectivity", "Epsilon (eps) Radius", "min_samples (Core Condition)", "Core vs Border vs Noise Points", "K-Distance Knee Heuristic", "Unsupervised Outlier Detection"],
        "methodology": [
            "Subsampled 5,000 students from the training partition to model high-dimensional Euclidean neighborhoods",
            "Fit NearestNeighbors with min_samples = 40 (2 x n_features) and generated k-distance rank plot",
            "Extracted optimal eps = 2.92 using the Kneedle maximum perpendicular distance algorithm",
            "Fit DBSCAN with ball_tree algorithm and evaluated overlap with ground-truth IsAnomaly tags",
            "Conducted epsilon sensitivity analysis and projected clusters and noise onto 2D PCA space"
        ],
        "findings": "DBSCAN identified 1 core dense cluster (97.8% of students) and 112 noise outliers (2.2%). 93.8% of DBSCAN's noise points matched ground-truth anomalies in the cohort (capturing 59.3% of all anomalies blind to labels)."
    },
    26: {
        "title": "t-SNE Non-Linear Manifold Visualization & Perplexity Dynamics",
        "built": True,
        "folder": "tsne",
        "report": "tsne_report.txt",
        "plots": [
            "tsne_perplexity_comparison.png",
            "tsne_placement_manifold_2d.png"
        ],
        "summary": "Maps high-dimensional student cohort geometry onto 2D non-linear manifolds using t-SNE, minimizing Kullback-Leibler (KL) divergence with Student-t heavy tails to resolve the crowding problem.",
        "key_concepts": ["Pairwise Gaussian Affinities", "Student-t Kernel (df=1)", "Kullback-Leibler (KL) Divergence", "Perplexity Tuning", "Crowding Problem Resolution", "PCA Pre-Reduction"],
        "methodology": [
            "Subsampled 3,500 students and pre-reduced 20 continuous attributes to 10 PCs via PCA (preserving 96.07% variance)",
            "Systematically swept perplexity across [5, 30, 50, 100] to evaluate local vs. global manifold topology",
            "Traced KL divergence convergence (optimal KL = 1.1093 at perplexity = 30 in 8.21s)",
            "Projected 2D embedding with KDE density contours and cohort centroids"
        ],
        "findings": "Perplexity = 5 fragments homogeneous student groups into micro-clusters, while Perplexity = 30 and 50 reveal a clean non-linear bifurcation between a dense 'Placed' archipelago and an 'Unplaced' outer periphery."
    },
    27: {
        "title": "UMAP Uniform Manifold Approximation & Projection",
        "built": True,
        "folder": "umap",
        "report": "umap_report.txt",
        "plots": [
            "umap_hyperparameter_grid.png",
            "umap_vs_tsne_vs_pca_benchmark.png"
        ],
        "summary": "Leverages fuzzy simplicial sets and Riemannian geometry to achieve rapid non-linear dimensionality reduction that preserves both local neighborhoods and global distance relationships, enabling out-of-sample projection.",
        "key_concepts": ["Riemannian Metric Spaces", "Fuzzy Simplicial Sets", "Local Connectivity (rho_i)", "Fuzzy Cross-Entropy Objective", "Global Structure Preservation", "Parametric Out-of-Sample Transform"],
        "methodology": [
            "Subsampled 5,000 students and swept n_neighbors in [5, 15, 50] and min_dist in [0.01, 0.10, 0.25, 0.50]",
            "Benchmarked computational latency and pairwise distance Spearman correlation against t-SNE and PCA",
            "Demonstrated out-of-sample parametric projection by transforming 1,000 held-out validation students in 6.97s"
        ],
        "findings": "UMAP runs over 2.2x faster than t-SNE (5.86s vs 13.11s) while achieving superior global distance correlation (r = 0.6089 vs 0.5727). Its parametric transform allows projecting new incoming students without re-fitting the entire cohort."
    },
    28: {
        "title": "Unsupervised Anomaly & Academic Trajectory Outlier Detection",
        "built": True,
        "folder": "anomaly",
        "report": "anomaly_detection_report.txt",
        "plots": [
            "anomaly_score_distribution.png",
            "anomaly_detection_pca_projection.png"
        ],
        "summary": "Isolates abnormal student academic trajectories without supervision using Isolation Forests (iTrees) and Local Outlier Factor (LOF), evaluated blind against ground-truth cohort anomalies.",
        "key_concepts": ["Isolation Trees (iTrees)", "Average Path Length E(h(x))", "Normalized Anomaly Score s(x,n)", "Local Outlier Factor (LOF)", "Academic Trajectory Collapse", "Discrepancy Auditing"],
        "methodology": [
            "Trained 250 Isolation Forest iTrees completely blind to labels across 35,000 students (contamination = 2.5%)",
            "Trained Local Outlier Factor with k=35 neighbors on a 5,000-student density neighborhood",
            "Audited detection accuracy against ground-truth IsAnomaly flags (ROC-AUC = 0.9535, PR-AUC = 0.4852)",
            "Extracted and profiled the top 5 most severe academic trajectory anomaly archetypes"
        ],
        "findings": "Isolation Forest achieved 0.9535 ROC-AUC blind to labels, catching 35.8% of anomalies at top-2.5% threshold. Flagged students exhibit extreme metric discordance (e.g. 9.7+ CGPA with bottom 1% coding scores) or abrupt semester GPA crashes."
    },
    29: {
        "title": "Cluster-Augmented Semi-Supervised Feature Engineering",
        "built": True,
        "folder": "cluster_features",
        "report": "cluster_augmented_report.txt",
        "plots": [
            "cluster_feature_importance.png",
            "augmented_model_performance_lift.png"
        ],
        "summary": "Bridges unsupervised clustering and supervised modeling by generating 9 synthetic topological features (K-Means centroid distances and GMM soft posteriors) to boost gradient boosted tree classification.",
        "key_concepts": ["Semi-Supervised Feature Augmentation", "Centroid Distance Encoding d(x, mu_k)", "Gaussian Mixture Model (GMM) Posteriors", "Representation Learning", "Feature Importance Attribution"],
        "methodology": [
            "Fit KMeans(K=4) and GaussianMixture(K=4) strictly on training data (zero-leakage discipline)",
            "Engineered 4 centroid Euclidean distance features, 4 GMM soft Bayesian posterior probabilities, and 1 cluster archetype ID",
            "Trained baseline vs. cluster-augmented Logistic Regression and LightGBM models on held-out validation data",
            "Evaluated validation accuracy, F1-score, ROC-AUC, and feature split importance rankings"
        ],
        "findings": "Synthetic cluster features Dist_To_Cluster_2 (371 splits) and Dist_To_Cluster_1 (251 splits) ranked as the #2 and #4 most important features in LightGBM, acting as holistic spatial landmarks that summarize multiple academic metrics simultaneously."
    },


    # -------------------------------------------------------------
    # Module 5: Model Evaluation, Validation & Optimization
    # -------------------------------------------------------------
    30: {
        "title": "Train/Val/Test Split Discipline & Data Leakage Prevention",
        "built": True,
        "folder": "split_leakage",
        "report": "split_leakage_report.txt",
        "plots": [
            "split_architecture_diagram.png",
            "data_leakage_distributions.png"
        ],
        "summary": "Establishes rigorous three-way disjoint partitioning (80% development vs 20% vault-locked test set), enforces mathematical disjoint conditions, and dissects Preprocessing, Temporal, and Target Leakage contamination mechanisms.",
        "key_concepts": [
            "Three-Way Disjoint Partitioning",
            "Vault-Locked Test Set",
            "Preprocessing Leakage (Global Scaling)",
            "Temporal / Sequential Contamination",
            "Target / Proxy Feature Leakage",
            "Optimistic Evaluation Bias"
        ],
        "methodology": [
            "Partitioned 50,000-student cohort into 80% Development (40,000 students) and 20% Vault-Locked Test (10,000 students)",
            "Sub-partitioned Development set into 60% Train (30,000) for parameter fitting (W, b) and 20% Validation (10,000) for hyperparameter tuning",
            "Enforced pairwise disjoint mathematical axioms: D_train ∩ D_val = ∅, D_train ∩ D_test = ∅, D_val ∩ D_test = ∅",
            "Simulated Preprocessing Leakage by fitting StandardScaler globally on all 50k records vs. fold-isolated training fitting",
            "Quantified empirical distribution shift and test accuracy differences across leakage vectors"
        ],
        "findings": "Sealing the 20% test partition (10,000 records) guarantees an unbiased generalization benchmark. Global preprocessing bleeds test statistics (mean, std) into training features, artificially deflating error estimates and creating an illusion of high model performance."
    },
    31: {
        "title": "K-Fold & Stratified Cross-Validation Schemes",
        "built": True,
        "folder": "cross_validation",
        "report": "cv_schemes_report.txt",
        "plots": [
            "kfold_vs_stratified_variance.png",
            "class_ratio_preservation.png"
        ],
        "summary": "Benchmarks Standard K-Fold against Stratified K-Fold cross-validation across K=5 folds, deriving sample variance, standard deviation, and standard error (SE = sigma / sqrt(K)) confidence bounds.",
        "key_concepts": [
            "K-Fold Rotational Partitioning",
            "Stratified K-Fold",
            "Class Prior Preservation (|F_{k,c}|/|F_k| = p_c)",
            "Sample Variance & Standard Deviation",
            "Standard Error of the Mean (SE)",
            "95% Student's t Confidence Interval",
            "Evaluation Variance Reduction"
        ],
        "methodology": [
            "Evaluated 2,500 students with 65.6% positive placement balance across K=5 cross-validation folds",
            "Audited fold-by-fold class prior preservation: Standard K-Fold fluctuated between 64.0% and 67.4% Placed, while Stratified K-Fold matched the exact 65.60% ground truth in every fold",
            "Calculated fold evaluation metrics: Mean CV Accuracy, sample variance, standard deviation, and standard error",
            "Derived 95% Confidence Interval using Student's t critical value (t_0.025, df=4 = 2.776)"
        ],
        "findings": "Stratified K-Fold reduced cross-validation score variance by 26% (Std Dev dropped from 1.69% to 1.25%, with Standard Error SE = 0.56%), completely eliminating class prior shift across validation folds."
    },
    32: {
        "title": "Scikit-Learn Pipeline & Held-Out Test Evaluation",
        "built": True,
        "folder": "pipeline_eval",
        "report": "pipeline_evaluation_report.txt",
        "plots": [
            "case_study_decision_boundary.png",
            "model_coefficients_case_study.png",
            "test_predictions_probabilities.png"
        ],
        "summary": "Encapsulates StandardScaler and LogisticRegression within a Scikit-Learn Pipeline to eliminate preprocessing leakage, executing a full 10-student case study (ExampleDataSetPP.csv) with 4-fold stratified CV and vault-locked test evaluation on S09 & S10.",
        "key_concepts": [
            "Scikit-Learn Pipeline Architecture",
            "Fold-Isolated Feature Transformation",
            "10-Student Placement Case Study",
            "4-Fold Stratified Development Validation",
            "Feature Attribution Weights (w_CGPA, w_Coding, etc.)",
            "Vault-Locked Held-Out Test Generalization",
            "Calibrated Probability Estimation"
        ],
        "methodology": [
            "Ingested the exact 10-student case study dataset (ExampleDataSetPP.csv) across 6 core predictive features: CGPA, Aptitude, Coding, Communication, Internship, Projects",
            "Sealed 20% Test Set (S09 & S10) and allocated remaining 80% (S01–S08) to Development",
            "Built make_pipeline(StandardScaler(), LogisticRegression()) ensuring zero transformation leakage",
            "Executed 4-fold Stratified CV on Development set (each validation fold containing 1 Placed and 1 Not Placed student)",
            "Retrained final model on full development set (S01–S08), extracted feature weights, and evaluated isolated test students S09 & S10"
        ],
        "findings": "Pipeline achieved 100% 4-Fold CV accuracy across development folds. Retrained model achieved 100% held-out test accuracy on S09 (88.2% predicted placement probability, Placed = 1) and S10 (4.1% probability, Not Placed = 0). CGPA (+0.463) and Internship (+0.781) are the primary positive placement drivers."
    },
    33: {
        "title": "Nested Stratified Cross-Validation & Tuning",
        "built": True,
        "folder": "nested_cv",
        "report": "nested_cv_report.txt",
        "plots": [
            "nested_cv_inner_outer_scores.png",
            "hyperparameter_validation_curve.png"
        ],
        "summary": "Implements dual-loop Nested Stratified CV (Outer K=5, Inner K=3) to optimize hyperparameter C over {0.01, 0.1, 1.0, 10.0} and eliminate hyperparameter selection bias in validation reporting.",
        "key_concepts": [
            "Dual-Loop Nested Cross-Validation",
            "Inner Tuning Loop (K_inner = 3)",
            "Outer Evaluation Loop (K_outer = 5)",
            "Hyperparameter Selection Bias",
            "Unbiased Generalization Score Estimation",
            "Double-Optimization Complexity (N_fits = 5 * (3 * 4 + 1) = 65 fits)"
        ],
        "methodology": [
            "Subsampled 1,000 students from the cohort to benchmark dual-loop optimization efficiency",
            "Configured outer StratifiedKFold (K=5) for performance estimation and inner StratifiedKFold (K=3) for grid search over C in {0.01, 0.1, 1.0, 10.0}",
            "Executed 65 total model fits across inner folds and outer test rotations",
            "Compared optimistic non-nested best CV score against unbiased nested outer test F1-score"
        ],
        "findings": "Non-nested CV yields an optimistically biased score of 92.93% F1, whereas Nested CV provides an unbiased generalization benchmark of 92.79% ± 0.0162 F1 (90.70% accuracy) across unseen outer folds, validating deployment readiness."
    },
    34: {
        "title": "ROC & Precision-Recall Curve Diagnostics",
        "built": True,
        "folder": "roc_pr_curves",
        "report": "roc_pr_report.txt",
        "plots": [
            "roc_and_pr_curves_comparison.png",
            "threshold_optimization_tradeoff.png"
        ],
        "summary": "Analyzes Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves across Logistic Regression, Random Forest, and LightGBM, optimizing operational probability thresholds via Youden's J index and F1 maximization.",
        "key_concepts": ["ROC-AUC vs PR-AUC (Average Precision)", "True Positive Rate (TPR) & False Positive Rate (FPR)", "Youden's J Statistic", "F1-Optimal Threshold Tuning", "Rare-Class Imbalance Diagnostics"],
        "methodology": [
            "Trained Logistic Regression, Random Forest (150 trees), and LightGBM on standardized training features and evaluated posterior probabilities on 5,000 validation students",
            "Plotted full ROC and Precision-Recall curves and computed ROC-AUC and Average Precision (PR-AUC)",
            "Simulated a 10% rare Tier-1 screening imbalance regime to prove why PR-AUC exposes false-positive inflation while ROC-AUC remains invariant",
            "Swept classification thresholds τ from 0.05 to 0.95 to identify Youden's J optimal cutoff and the F1-maximizing operational threshold"
        ],
        "findings": "LightGBM achieved the highest discrimination (0.978+ ROC-AUC, 0.988+ PR-AUC). Under 10% class imbalance, ROC-AUC remained unchanged at ~0.978 while PR-AUC dropped to ~0.86, proving PR-AUC is essential for selective Tier-1 candidate screening."
    },
    35: {
        "title": "Regression Loss Functions & Robust Metrics",
        "built": True,
        "folder": "regression_loss",
        "report": "regression_loss_report.txt",
        "plots": [
            "loss_functions_and_outlier_sensitivity.png",
            "salary_residual_robustness.png"
        ],
        "summary": "Compares L2 Squared Error (OLS/Ridge), L1 Absolute Error (Median Quantile Regression), and Huber Hybrid Loss for predicting continuous student salary packages in the presence of extreme international offer outliers.",
        "key_concepts": ["L2 Squared Error (MSE/RMSE)", "L1 Absolute Error (MAE & Quantile q=0.5)", "Huber Hybrid Loss (δ=1.35)", "Mean Absolute Percentage Error (MAPE)", "Super-Dream Salary Outlier Robustness"],
        "methodology": [
            "Extracted placed student compensation records and injected 3% extreme super-dream/international package outliers (45–65 LPA) into the training partition",
            "Trained L2 Ridge Regression, HuberRegressor (δ = 1.35), and L1 QuantileRegressor (q = 0.50) on both clean and outlier-contaminated training sets",
            "Evaluated RMSE inflation, MAE, and MAPE on 1,500 held-out validation placed students"
        ],
        "findings": "3% extreme salary outliers severely inflated L2 (OLS/Ridge) validation RMSE because quadratic loss squares large residuals, whereas Huber Loss (δ = 1.35) and L1 Quantile Regression bounded outlier gradient pull and preserved stable validation RMSE and MAE."
    },
    36: {
        "title": "Grid Search vs Random Search vs Bayesian Optimization",
        "built": True,
        "folder": "search_schemes",
        "report": "search_schemes_report.txt",
        "plots": [
            "search_convergence_comparison.png",
            "hyperparameter_space_exploration.png"
        ],
        "summary": "Benchmarks sample efficiency and best-so-far validation F1 convergence across exhaustive Cartesian Grid Search, Randomized Search, and Gaussian Process Expected Improvement (EI) Bayesian Optimization.",
        "key_concepts": ["Cartesian Grid Curse of Dimensionality", "Bergstra-Bengio Random Search Theorem", "Gaussian Process (Matern 5/2) Surrogate", "Expected Improvement (EI) Acquisition", "Exploration vs Exploitation"],
        "methodology": [
            "Allocated an identical 27-evaluation budget to tune Random Forest hyperparameters (max_depth, min_samples_split, max_features)",
            "Executed a 3x3x3 Cartesian Grid Search, 27-trial continuous Randomized Search, and 27-trial GP-Bayesian Expected Improvement loop",
            "Tracked cumulative best validation F1-score trajectory and 2D hyperparameter landscape coverage"
        ],
        "findings": "Grid Search wasted evaluations repeating identical max_depth values across low-impact secondary axes, whereas Gaussian Process Bayesian Optimization concentrated evaluations inside the high-F1 basin (max_depth ≈ 11–14), achieving the highest validation F1 within 27 trials."
    },
    37: {
        "title": "Learning Curves & Sample Complexity Analysis",
        "built": True,
        "folder": "learning_curves",
        "report": "learning_curves_report.txt",
        "plots": [
            "learning_curves_bias_variance_regimes.png",
            "generalization_gap_saturation.png"
        ],
        "summary": "Plots empirical training and validation learning curves from N = 250 to N = 35,000 students to diagnose high-bias capacity bottlenecks versus high-variance memorization gaps.",
        "key_concepts": ["Empirical Learning Curves", "Sample Complexity & Data Saturation", "High-Bias Plateau Signature", "High-Variance Memorization Gap", "Asymptotic Generalization Convergence"],
        "methodology": [
            "Trained High-Bias Logistic Regression (C=0.05), High-Variance Unpruned Decision Tree (max_depth=None), and Balanced Random Forest (max_depth=10) across 8 logarithmic sample sizes (250 to 35,000 students)",
            "Computed Training F1, Held-Out Validation F1, and the Generalization Gap (Train F1 - Val F1) at each sample scale",
            "Identified sample saturation inflection points for linear vs ensemble architectures"
        ],
        "findings": "Linear Logistic Regression saturates by N = 2,500 students (generalization gap < 0.5%), proving additional data cannot overcome linear capacity limits. Unpruned trees maintain a >10% overfitting gap even at N = 35,000, while depth-regulated Random Forests converge to >93.8% F1 with a <1.5% gap."
    },
    38: {
        "title": "Hyperparameter Optimization via Optuna",
        "built": True,
        "folder": "optuna_tuning",
        "report": "optuna_tuning_report.txt",
        "plots": [
            "optuna_tpe_optimization_history.png",
            "optuna_hyperparameter_importance.png"
        ],
        "summary": "Employs sequential Tree-Structured Parzen Estimator (TPE) density-ratio sampling and automated Median Pruning at intermediate boosting checkpoints to optimize LightGBM across 40 trials.",
        "key_concepts": ["Tree-Structured Parzen Estimator (TPE l(x)/g(x))", "Intermediate Checkpoint Median Pruning", "Log-Uniform Learning Rate Sampling", "Functional ANOVA (fANOVA) Importance"],
        "methodology": [
            "Parameterized a 5-dimensional LightGBM search space: learning_rate, num_leaves, colsample_bytree, subsample, and reg_lambda",
            "Executed 40 sequential TPE trials with intermediate validation ROC-AUC checks at 25 boosting trees to prune underperforming trials early",
            "Fit a Functional ANOVA (fANOVA) surrogate forest over trial history to quantify each hyperparameter's variance contribution"
        ],
        "findings": "Automated Median Pruning terminated over 40% of unpromising trials at tree 25, saving ~30% compute time while discovering a champion LightGBM configuration (ROC-AUC = 0.9784, F1 = 94.46%). fANOVA revealed learning_rate and num_leaves drive >70% of performance variance."
    },
    39: {
        "title": "Probability Calibration & Brier Score Diagnostics",
        "built": True,
        "folder": "calibration",
        "report": "calibration_report.txt",
        "plots": [
            "reliability_calibration_curves.png",
            "brier_score_and_ece_comparison.png"
        ],
        "summary": "Evaluates posterior probability reliability using Calibration Curves, Brier Score Loss, and Expected Calibration Error (ECE), applying Platt Scaling (Sigmoid) and Isotonic Regression via CalibratedClassifierCV.",
        "key_concepts": ["Reliability Diagrams (Calibration Curves)", "Platt Scaling (Parametric Sigmoid)", "Isotonic Regression (Non-Parametric Monotonic)", "Brier Score Loss Decomposition", "Expected Calibration Error (ECE)"],
        "methodology": [
            "Audited uncalibrated Gaussian Naive Bayes and Random Forest classifiers on 5,000 validation students across 10 uniform probability bins",
            "Applied 3-fold CalibratedClassifierCV using both 'sigmoid' (Platt Scaling) and 'isotonic' regression",
            "Quantified Brier Score Loss, Log-Loss, and Expected Calibration Error (ECE) before and after calibration"
        ],
        "findings": "Uncalibrated Gaussian NB suffers from extreme bimodal probability distortion (ECE > 8.5%); Isotonic Calibration slashes its Brier Score by >25% and ECE by >75%. Isotonic-Calibrated Random Forest achieves a Brier Score of 0.056 and ECE < 1.2%, ensuring predicted probabilities reflect true empirical placement rates."
    },

    # -------------------------------------------------------------
    # Module 6: ML Engineering & Production Serving
    # -------------------------------------------------------------
    40: {
        "title": "Production Pipeline Assembly & Packaging",
        "built": True,
        "folder": "prod_packaging",
        "report": "prod_packaging_report.txt",
        "plots": [
            "artifact_compression_benchmark.png",
            "pipeline_stage_latency_and_metrics.png"
        ],
        "summary": "Packages the complete preprocessing ColumnTransformer and tuned LightGBM classifier into a versioned production artifact (v2.4.0) with SHA-256 cryptographic digest verification and compression benchmarking.",
        "key_concepts": ["Unified ColumnTransformer + Estimator Artifact", "Semantic Versioning (v2.4.0)", "SHA-256 Artifact Integrity Digest", "Zlib Compression Level Trade-offs", "Cold-Start Deserialization SLA"],
        "methodology": [
            "Encapsulated median/mode imputation, StandardScaler, OneHotEncoder(handle_unknown='ignore'), and tuned LightGBM into a single atomic Scikit-Learn Pipeline",
            "Benchmarked Joblib serialization across Zlib compression levels 0, 1, 3, 6, and 9 for disk size, write time, and cold-start load latency",
            "Generated model_manifest.json recording schema contracts, SHA-256 checksum, and held-out vault test benchmarks (N = 10,000)"
        ],
        "findings": "Zlib compression level 3 reduces artifact footprint by >60% while keeping cold-start deserialization under 15 ms. On the 10,000-student vault test set, the packaged v2.4.0 artifact achieves 92.68% accuracy, 94.46% F1, and 97.84% ROC-AUC in 82 ms batch execution."
    },
    41: {
        "title": "Training-Serving Skew Detection",
        "built": True,
        "folder": "skew_detection",
        "report": "skew_detection_report.txt",
        "plots": [
            "serving_skew_impact_and_recovery.png",
            "probability_distortion_scatter.png"
        ],
        "summary": "Audits live inference payloads for Training-Serving Skew (100-point scale CGPA entry, fractional [0,1] test scores, missing fields, and unseen stream categories) and implements pre-inference schema guardrails.",
        "key_concepts": ["Training-Serving Skew Taxonomy", "Unit Scale Mismatch Detection", "Pre-Inference Schema Guardrails", "Unknown Category Routing", "Silent Degradation Prevention"],
        "methodology": [
            "Simulated 3,000 live web inference requests with 25% contaminated by realistic frontend payload skews (CGPA entered on 100-pt scale, Coding score on [0,1] scale, unseen Stream strings, and missing optional fields)",
            "Quantified accuracy and F1 collapse when skewed payloads are fed directly into the model without unit validation",
            "Implemented automated pre-inference guardrails that detect and normalize scale mismatches prior to pipeline execution"
        ],
        "findings": "Unmitigated unit skew distorts predicted probabilities and degrades F1-score across affected cohorts. Automated pre-inference guardrails intercept 100% of unit and schema anomalies, restoring full baseline performance (92.7% Accuracy, 94.5% F1)."
    },
    42: {
        "title": "Model Serialization & ONNX Runtime Portability",
        "built": True,
        "folder": "serialization",
        "report": "serialization_report.txt",
        "plots": [
            "serialization_format_benchmark.png",
            "batch_vs_single_latency_curve.png"
        ],
        "summary": "Benchmarks 5 model serialization and execution formats—Python Pickle, Uncompressed Joblib, Compressed Joblib, Native C++ LightGBM Booster, and Portable FP32 Graph JSON—across disk size, cold-start load, and single-row vs batch latency.",
        "key_concepts": ["Cross-Platform Model Serialization", "Native C++ Booster Execution", "Portable FP32 Graph Weights", "Single-Row Wrapper Overhead Amortization", "Numerical Equivalence (<1e-12)"],
        "methodology": [
            "Exported trained placement models into 5 serialization formats (.pkl, .joblib c=0, .joblib c=3, LightGBM native .txt, and portable FP32 JSON weight graph)",
            "Measured artifact size (KB), cold-start deserialization latency (ms), and 10,000-student batch inference latency (ms)",
            "Swept inference batch sizes from B = 1 to B = 10,000 to measure per-student latency amortization in microseconds"
        ],
        "findings": "Bypassing Scikit-Learn's Python DataFrame validation wrapper via native booster handles or portable FP32 matrix kernels slashes single-instance (B=1) latency by over 15x while maintaining bit-exact numerical equivalence (max probability delta < 1e-12)."
    },
    43: {
        "title": "FastAPI & Microservice Inference Serving",
        "built": True,
        "folder": "microservice_serving",
        "report": "microservice_serving_report.txt",
        "plots": [
            "microservice_latency_percentiles.png",
            "concurrent_throughput_scaling.png"
        ],
        "summary": "Stress-tests the production REST inference endpoints (/api/predict and /api/module5/predict) across 800 live HTTP requests, auditing p50/p90/p95/p99 tail latency against a strict 10.0 ms SLA and modeling multi-worker concurrency scaling.",
        "key_concepts": ["REST Microservice Inference Endpoints", "Tail Latency Percentiles (p50, p95, p99)", "In-Memory Model Caching (_M5_CACHE)", "Sub-10ms Real-Time SLA", "Horizontal Worker Throughput Scaling"],
        "methodology": [
            "Executed 400 randomized student payload POST requests against /api/predict and 400 against /api/module5/predict using the live Flask WSGI test client",
            "Computed empirical p50, p90, p95, and p99 response time percentiles and verified HTTP 200 JSON schema contracts",
            "Modeled horizontal throughput scaling (requests/sec) across 1, 2, 4, 8, and 16 concurrent worker processes"
        ],
        "findings": "In-memory pipeline caching keeps /api/predict p95 latency under 0.6 ms and /api/module5/predict p95 latency under 2.5 ms—more than 4x faster than the 10.0 ms production SLA ceiling—supporting >2,500 requests/sec on an 8-worker node."
    },
    44: {
        "title": "Automated Unit & Integration Testing",
        "built": True,
        "folder": "automated_testing",
        "report": "automated_testing_report.txt",
        "plots": [
            "automated_test_suite_results.png",
            "metamorphic_testing_coverage.png"
        ],
        "summary": "Executes an 8-gate automated ML testing suite covering simplex probability bounds, deterministic idempotence, all-NaN & unseen-category resilience, directional metamorphic monotonicity, and REST API contract assertions.",
        "key_concepts": ["ML Testing Pyramid", "Simplex Probability Axiom [0,1]", "Deterministic Idempotence", "Metamorphic Directional Monotonicity", "Minimum F1/AUC Quality Gates"],
        "methodology": [
            "Built 8 automated test gates across 5 verification layers: Axiom & Bounds, Edge-Case Input, Behavioral/Metamorphic, Performance Gate, and Integration API",
            "Verified that all-NaN numeric payloads and unseen categorical tokens ('Quantum Computing', 'Atlantis') impute cleanly without raising exceptions",
            "Executed metamorphic directional tests confirming that increasing CGPA + Coding + Internships monotonically raises placement probability while active backlogs lower it"
        ],
        "findings": "All 8/8 automated test assertions passed in under 60 ms total execution time. Metamorphic sweeps confirmed strict monotonic probability lift (+54% mean placement probability shift from CGPA 5.5 to 9.4) and zero NaN emissions."
    },
    45: {
        "title": "Data & Concept Drift Monitoring",
        "built": True,
        "folder": "drift_monitoring",
        "report": "drift_monitoring_report.txt",
        "plots": [
            "psi_feature_drift_dashboard.png",
            "concept_drift_retraining_recovery.png"
        ],
        "summary": "Monitors incoming student cohorts using two-sample Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) across 10 decile bins, firing automated retraining triggers when PSI exceeds the 0.25 critical threshold.",
        "key_concepts": ["Population Stability Index (PSI)", "Two-Sample Kolmogorov-Smirnov (KS) Test", "Covariate Shift vs Concept Drift", "PSI Alert Thresholds (0.10 Warning, 0.25 Critical)", "Automated Retraining Recovery"],
        "methodology": [
            "Compared 5,000 baseline training students against a simulated 2028 incoming cohort exhibiting AI-boom skill inflation (+14.5 Coding pts, +0.62 CGPA shift)",
            "Computed 10-bin Population Stability Index (PSI) and two-sample KS statistics with p-values across 6 core numeric features",
            "Simulated longitudinal quarterly F1 decay from 2026-Q1 to 2028-Q2 comparing a frozen legacy model against PSI-triggered automated retraining"
        ],
        "findings": "CodingTestScore (PSI > 0.45) and CGPA (PSI > 0.30) breached the PSI >= 0.25 critical threshold with KS p-values < 1e-50. While the frozen model's F1 decayed to 86.4%, PSI-triggered automated retraining restored cohort F1 to 94.5%."
    },
    46: {
        "title": "MLflow Experiment Tracking & Registry",
        "built": True,
        "folder": "experiment_tracking",
        "report": "experiment_tracking_report.txt",
        "plots": [
            "registry_version_lineage.png",
            "champion_challenger_promotion_audit.png"
        ],
        "summary": "Maintains a structured Model Registry catalog tracking hyperparameter lineage, validation metrics, and latency across 6 major model versions (v0.1.0 to v2.4.0), governing lifecycle promotion from Archived and Staging to Production.",
        "key_concepts": ["Experiment Lineage Catalog", "Model Registry Lifecycle Stages", "Champion vs Challenger Promotion Gates", "Pareto Accuracy-Latency Frontier", "Reproducible Run Auditing"],
        "methodology": [
            "Logged 6 milestone model versions spanning v0.1.0 (Zero-Rule Baseline), v1.0.0 (Logistic Regression), v1.2.0 (CART), v1.5.0 (Random Forest), v2.1.0 (XGBoost), and v2.4.0 (TPE-Tuned LightGBM)",
            "Evaluated Staging challengers against Production promotion criteria across 5 axes: Accuracy, F1-Score, ROC-AUC, Calibration (100 - ECE), and SLA Speed Score",
            "Exported model_registry_catalog.json recording full run lineage and stage transitions"
        ],
        "findings": "v2.4.0 LightGBM TPE-Tuned satisfied all 5 Champion promotion gates, outperforming Staging candidate v1.5.0 Random Forest by +0.64% F1 and +0.69% ROC-AUC while executing 3.2x faster (1.48 ms vs 4.85 ms)."
    },
    47: {
        "title": "Docker Containerization & CI/CD Deployment",
        "built": True,
        "folder": "docker_cicd",
        "report": "docker_cicd_report.txt",
        "plots": [
            "docker_image_footprint_and_cicd.png",
            "canary_deployment_rollout.png"
        ],
        "summary": "Generates a hardened multi-stage production Dockerfile (non-root UID 10001, healthcheck probes), cuts container image bloat by 65.3%, and audits a 6-stage GitHub Actions CI/CD pipeline with progressive canary rollout.",
        "key_concepts": ["Multi-Stage Dockerfile Architecture", "Image Layer Footprint Optimization", "Non-Root Container Hardening (UID 10001)", "6-Stage GitHub Actions CI/CD Gate", "Zero-Downtime Canary Rollout"],
        "methodology": [
            "Authored Dockerfile.production separating the compiler/wheel builder stage from the lean python:3.11-slim runtime image",
            "Benchmarked container layer footprints across Unoptimized Full (1,057 MB), Single-Stage Slim (737 MB), and Multi-Stage Production (367 MB)",
            "Audited a 6-stage automated CI/CD workflow (Lint -> Schema Check -> ML Tests -> Docker Build -> Healthcheck -> Canary Shift)"
        ],
        "findings": "Stripping compiler toolchains and pip caches via multi-stage builds reduced container image size from 1,057 MB to 367 MB (-65.3%). The complete 6-stage CI/CD pipeline executes in 65.0 seconds and safely shifts 100% of production traffic over a 30-minute canary window."
    },
    48: {
        "title": "Final Capstone Reporting & Executive Summary",
        "built": True,
        "folder": "capstone_summary",
        "report": "capstone_summary_report.txt",
        "plots": [
            "capstone_progression_and_fairness.png",
            "university_roi_intervention_lift.png"
        ],
        "summary": "Synthesizes all 48 curriculum sessions across Modules 1–6, auditing cross-module accuracy progression (+26.97% lift over baseline), Equal Opportunity demographic fairness across Gender and College Tier, and University Placement Cell bootcamp ROI.",
        "key_concepts": ["48-Session Curriculum Synthesis", "Cross-Module Generalization Progression", "Equal Opportunity & Demographic Parity Audit", "At-Risk Cohort Identification", "University Placement Cell ROI Simulation"],
        "methodology": [
            "Consolidated held-out test benchmarks across all 6 curriculum modules from Session 02 ZeroR Baseline (65.71% Acc) to Session 40 Tuned LightGBM (92.68% Acc, 94.46% F1)",
            "Conducted an Ethical AI fairness audit on 10,000 test students measuring Equal Opportunity Recall (TPR) and F1 parity across Gender and CollegeTier subgroups",
            "Simulated a targeted university bootcamp intervention (+18 Coding pts, +15 Aptitude/Interview pts, 1 Internship) on borderline at-risk students (predicted probability 25%–50%)"
        ],
        "findings": "Across 48 completed sessions, model accuracy rose by +26.97% (65.71% to 92.68%) with a Gender Equal Opportunity gap of <0.5% (strictly passing the 1.5% fairness standard). Targeted bootcamp intervention converts over 75% of borderline at-risk students into placement-ready candidates."
    },
}
