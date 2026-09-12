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
    {"id": 5, "name": "Evaluation, Diagnostics & Model Selection", "sessions": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]},
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
        "built": False,
        "summary": "Explores probabilistic centroid initialization (K-Means++) to prevent poor local minima traps, and evaluates Mini-Batch K-Means for scaling to multi-million record student cohorts.",
        "key_concepts": ["K-Means++ Initialization", "Mini-Batch K-Means", "Local Minima Avoidance", "Computational Complexity (O(nkt))"]
    },
    23: {
        "title": "Principal Component Analysis (PCA)",
        "built": False,
        "summary": "Applies orthogonal linear transformation via Singular Value Decomposition (SVD) to project 45 high-dimensional student features onto principal components explaining maximal variance.",
        "key_concepts": ["Eigenvectors & Eigenvalues", "Explained Variance Ratio", "Scree Plot", "Dimensionality Reduction"]
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
        "title": "DBSCAN Density-Based Clustering",
        "built": False,
        "summary": "Applies Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to discover arbitrary-shaped student clusters and isolate anomalous student outliers.",
        "key_concepts": ["Epsilon (eps) Neighborhood", "MinPts", "Core, Border & Noise Points", "Non-Spherical Clusters"]
    },
    26: {
        "title": "t-SNE Non-Linear Manifold Visualization",
        "built": False,
        "summary": "Uses t-Distributed Stochastic Neighbor Embedding (t-SNE) to map high-dimensional student feature relationships onto 2D interactive scatter plots preserving local neighborhood structure.",
        "key_concepts": ["Kullback-Leibler (KL) Divergence", "Perplexity Tuning", "Non-Linear Manifold Projection", "Crowding Problem"]
    },
    27: {
        "title": "UMAP Uniform Manifold Approximation",
        "built": False,
        "summary": "Leverages Uniform Manifold Approximation and Projection (UMAP) based on Riemannian geometry to achieve faster dimensionality reduction that preserves both local and global data topology.",
        "key_concepts": ["Fuzzy Simplicial Sets", "Topological Data Analysis", "UMAP vs t-SNE", "Global Structure Preservation"]
    },
    28: {
        "title": "Anomaly & Outlier Detection in Academic Trajectories",
        "built": False,
        "summary": "Employs Isolation Forests and Local Outlier Factors (LOF) to flag anomalous student academic records, such as sudden extreme drops in semester GPA or conflicting test results.",
        "key_concepts": ["Isolation Forest", "Path Length Anomaly Score", "Local Outlier Factor (LOF)", "Academic Integrity Auditing"]
    },
    29: {
        "title": "Cluster-Augmented Feature Engineering",
        "built": False,
        "summary": "Feeds unsupervised cluster assignments and distances-to-centroids as synthetic features back into supervised gradient boosted models to boost classification accuracy.",
        "key_concepts": ["Semi-Supervised Feature Augmentation", "Distance-to-Centroid Encoding", "Representation Learning", "Hybrid Modeling"]
    },

    # -------------------------------------------------------------
    # Module 5: Evaluation, Diagnostics & Model Selection (Roadmap)
    # -------------------------------------------------------------
    30: {
        "title": "Data Leakage Diagnostics & Audit Protocols",
        "built": False,
        "summary": "Explores subtle data leakage vectors (temporal leakage, target leakage, transformation leakage) and builds automated unit test assertions to detect contamination in production pipelines.",
        "key_concepts": ["Target Leakage", "Temporal Contamination", "Split Inversion", "Pipeline Unit Testing"]
    },
    31: {
        "title": "K-Fold Cross-Validation Schemes",
        "built": False,
        "summary": "Implements Stratified K-Fold and Repeated K-Fold Cross-Validation to generate unbiased variance bounds and confidence intervals for student placement predictions.",
        "key_concepts": ["Stratified K-Fold", "Cross-Validation Variance", "Sample Complexity", "Out-of-Fold (OOF) Predictions"]
    },
    32: {
        "title": "Nested Cross-Validation for Unbiased Model Selection",
        "built": False,
        "summary": "Separates hyperparameter tuning (inner fold) from generalizability estimation (outer fold) using Nested Cross-Validation to eliminate optimization selection bias.",
        "key_concepts": ["Inner vs Outer CV Loops", "Selection Bias Mitigation", "Unbiased Error Estimation"]
    },
    33: {
        "title": "Confusion Matrix & Optimal Decision Thresholds",
        "built": False,
        "summary": "Optimizes classification probability thresholds beyond 0.50 based on campus placement costs, balancing False Negatives (missed student support) against False Positives.",
        "key_concepts": ["Cost-Sensitive Learning", "Threshold Sweeping", "Precision-Recall Tradeoff", "F-Beta Optimization"]
    },
    34: {
        "title": "ROC & Precision-Recall Curve Diagnostics",
        "built": False,
        "summary": "Analyzes Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves, demonstrating why PR-AUC is more informative in imbalanced candidate screening scenarios.",
        "key_concepts": ["ROC-AUC vs PR-AUC", "True Positive Rate (TPR)", "False Positive Rate (FPR)", "Class Imbalance Diagnostics"]
    },
    35: {
        "title": "Regression Loss Functions & Robust Metrics",
        "built": False,
        "summary": "Compares Mean Squared Error (MSE), Mean Absolute Error (MAE), and Huber Loss for predicting continuous placement salary packages in the presence of extreme package outliers.",
        "key_concepts": ["Huber Loss", "L1 vs L2 Loss Sensitivity", "Mean Absolute Percentage Error (MAPE)", "Quantile Regression"]
    },
    36: {
        "title": "Grid Search vs Random Search vs Bayesian Optimization",
        "built": False,
        "summary": "Benchmarks computational efficiency between exhaustive Grid Search, Randomized Search, and Gaussian Process Bayesian Optimization for tuning ensemble hyperparameters.",
        "key_concepts": ["Exploration vs Exploitation", "Gaussian Process Surrogates", "Acquisition Functions (Expected Improvement)"]
    },
    37: {
        "title": "Learning Curves & Sample Complexity Analysis",
        "built": False,
        "summary": "Plots empirical learning curves across varying training dataset sizes (1,000 to 35,000 students) to diagnose whether models are bottlenecked by data volume or model capacity.",
        "key_concepts": ["Empirical Learning Curves", "Data Saturation", "Sample Complexity", "High Bias vs High Variance Signatures"]
    },
    38: {
        "title": "Hyperparameter Optimization via Optuna",
        "built": False,
        "summary": "Employs Optuna with Tree-structured Parzen Estimator (TPE) and Median Pruning to optimize LightGBM learning rate, num_leaves, and colsample_bytree in under 100 trials.",
        "key_concepts": ["Optuna TPE Sampler", "Hyperband / Median Pruning", "Search Space Parameterization"]
    },
    39: {
        "title": "Probability Calibration & Brier Score Diagnostics",
        "built": False,
        "summary": "Calibrates predicted placement probabilities using Platt Scaling (Logistic) and Isotonic Regression to ensure a predicted 80% placement probability matches an empirical 80% cohort rate.",
        "key_concepts": ["Platt Scaling", "Isotonic Regression", "Brier Score", "Reliability Diagrams (Calibration Curves)"]
    },

    # -------------------------------------------------------------
    # Module 6: ML Engineering & Production Serving (Roadmap)
    # -------------------------------------------------------------
    40: {
        "title": "Production Pipeline Assembly & Packaging",
        "built": False,
        "summary": "Packages the complete trained ML pipeline, configuration constants, and inference schemas into a pip-installable Python wheel package with strict semantic versioning.",
        "key_concepts": ["Python Packaging (pyproject.toml)", "Schema Enforcement (Pydantic)", "Model Versioning", "Reproducible Wheels"]
    },
    41: {
        "title": "Training-Serving Skew Detection",
        "built": False,
        "summary": "Implements validation checks to detect discrepancies between training feature pipelines and live web inference inputs, preventing silent performance degradation.",
        "key_concepts": ["Feature Schema Drift", "Data Type Mismatches", "Missing Value Defaults", "Pipeline Invariant Assertions"]
    },
    42: {
        "title": "Model Serialization & ONNX Runtime Portability",
        "built": False,
        "summary": "Converts scikit-learn and LightGBM models into the Open Neural Network Exchange (ONNX) format for ultra-fast, cross-platform, hardware-accelerated model execution.",
        "key_concepts": ["ONNX Format", "ONNX Runtime Execution", "Model Serialization", "Hardware Acceleration"]
    },
    43: {
        "title": "FastAPI & Microservice Inference Serving",
        "built": False,
        "summary": "Constructs a production-grade asynchronous REST API using FastAPI, with automatic OpenAPI documentation, Pydantic input validation, and sub-10ms response latency.",
        "key_concepts": ["FastAPI Async Serving", "Pydantic Input Validation", "REST Endpoints", "Latency Benchmarking"]
    },
    44: {
        "title": "Automated Unit & Integration Testing",
        "built": False,
        "summary": "Builds a comprehensive test suite with pytest, testing pipeline transformations, model output bounds, edge-case student inputs, and regression performance assertions.",
        "key_concepts": ["pytest Test Suite", "Boundary & Edge Case Testing", "Behavioral Testing", "Continuous Integration (CI)"]
    },
    45: {
        "title": "Data & Concept Drift Monitoring",
        "built": False,
        "summary": "Implements Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI) to detect distribution shifts in incoming student cohorts over future academic years.",
        "key_concepts": ["Population Stability Index (PSI)", "Kolmogorov-Smirnov Test", "Concept Drift", "Model Retraining Triggers"]
    },
    46: {
        "title": "MLflow Experiment Tracking & Registry",
        "built": False,
        "summary": "Integrates MLflow to log hyperparameters, training metrics, ROC curves, and model artifacts, managing models through Staging, Production, and Archived lifecycles.",
        "key_concepts": ["MLflow Tracking", "Artifact Storage", "Model Registry", "Deployment Lifecycle Management"]
    },
    47: {
        "title": "Docker Containerization & CI/CD Deployment",
        "built": False,
        "summary": "Containerizes the complete intelligence platform into a lightweight Docker image, establishing GitHub Actions CI/CD workflows for automated testing and cloud deployment.",
        "key_concepts": ["Multi-Stage Dockerfile", "GitHub Actions CI/CD", "Container Orchestration", "Cloud Deployment"]
    },
    48: {
        "title": "Final Capstone Reporting & Executive Summary",
        "built": False,
        "summary": "Synthesizes the entire 48-session machine learning journey into an executive technical audit, evaluating model performance, business ROI for colleges, and student career impact.",
        "key_concepts": ["Executive Capstone Report", "Business Impact Analysis", "Ethical AI & Fairness Audit", "Curriculum Summary"]
    },
}
