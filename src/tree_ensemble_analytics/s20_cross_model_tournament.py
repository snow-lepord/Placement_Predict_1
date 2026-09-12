# Session 20: Cross-Model Tournament & Benchmark
# Rigorous side-by-side evaluation across linear models, decision trees, bagging, and modern boosting.
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import config

os.makedirs(config.MODEL_COMPARISON_DIR, exist_ok=True)

# 1. Linear Track (Pipeline from Session 11)
train_raw = pd.read_csv(os.path.join(config.SPLITS_DIR, "train.csv"))
val_raw = pd.read_csv(os.path.join(config.SPLITS_DIR, "val.csv"))
linear_features = config.NUMERIC_COLUMNS + config.CATEGORY_COLUMNS + config.YES_NO_COLUMNS + ["CollegeTier", "CGPA_Tier"]
x_train_raw, y_train_raw = train_raw[linear_features], train_raw[config.TARGET_COLUMN]
x_val_raw, y_val_raw = val_raw[linear_features], val_raw[config.TARGET_COLUMN]

numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
ordinal_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OrdinalEncoder(categories=[["Tier3", "Tier2", "Tier1"], ["Low", "Mid", "High"]]))])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, config.NUMERIC_COLUMNS),
    ("ordinal", ordinal_pipe, ["CollegeTier", "CGPA_Tier"]),
    ("categorical", cat_pipe, config.CATEGORY_COLUMNS + config.YES_NO_COLUMNS),
])
linear_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE))
])

t0 = time.time()
linear_pipeline.fit(x_train_raw, y_train_raw)
fit_linear = time.time() - t0
lin_preds = linear_pipeline.predict(x_val_raw)
lin_probs = linear_pipeline.predict_proba(x_val_raw)[:, 1]

linear_results = {
    "val_accuracy": accuracy_score(y_val_raw, lin_preds),
    "val_auc": roc_auc_score(y_val_raw, lin_probs),
    "val_log_loss": log_loss(y_val_raw, lin_probs),
    "fit_seconds": fit_linear,
    "n_learned_parameters": linear_pipeline.named_steps["clf"].coef_.shape[1] + 1
}

# 2. Tree Ensembles (Prepared Encoded Splits)
train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train_ready.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val_ready.csv"))
drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
feat_cols = [c for c in train.columns if c not in drop_cols]
x_tr, y_tr = train[feat_cols], train[config.TARGET_COLUMN]
x_va, y_va = val[feat_cols], val[config.TARGET_COLUMN]

# LightGBM best model
t0 = time.time()
lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4, random_state=config.RANDOM_STATE, verbose=-1)
lgbm.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])
fit_tree = time.time() - t0
tree_preds = lgbm.predict(x_va)
tree_probs = lgbm.predict_proba(x_va)[:, 1]

tree_results = {
    "val_accuracy": accuracy_score(y_va, tree_preds),
    "val_auc": roc_auc_score(y_va, tree_probs),
    "val_log_loss": log_loss(y_va, tree_probs),
    "fit_seconds": fit_tree,
    "n_trees": lgbm.best_iteration_
}

# 3. Comprehensive Tournament of All Models
all_models = {
    "Baseline Heuristic": 0.6570,
    "Logistic Regression (Plain)": 0.9154,
    "Logistic Regression (Pipeline)": linear_results["val_accuracy"],
    "Decision Tree (Pruned)": 0.9080,
    "Random Forest (200 Trees)": 0.9210,
    "AdaBoost": 0.9160,
    "Gradient Boosting (sklearn)": 0.9238,
    "XGBoost": accuracy_score(y_va, XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4, early_stopping_rounds=30, random_state=config.RANDOM_STATE, eval_metric="logloss").fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False).predict(x_va)),
    "LightGBM": tree_results["val_accuracy"],
    "CatBoost": accuracy_score(y_va, CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=4, early_stopping_rounds=30, random_seed=config.RANDOM_STATE, verbose=False).fit(x_tr, y_tr, eval_set=(x_va, y_va), verbose=False).predict(x_va)),
}

# 1. All models horizontal bar chart
plt.figure(figsize=(9, 5.5))
models_sorted = pd.Series(all_models).sort_values()
bars = plt.barh(models_sorted.index, models_sorted.values, color="#4f46e5")
plt.xlim(0.6, 0.96)
plt.xlabel("Validation Accuracy")
plt.title("Tournament Benchmark: Validation Accuracy across 10 Model Architectures", fontweight="bold")
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.003, bar.get_y() + bar.get_height()/2.0, f"{w:.4f}", va="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.MODEL_COMPARISON_DIR, "all_models_accuracy.png"), dpi=120)
plt.close()

# 2. Head-to-head comparison
plt.figure(figsize=(7, 4.5))
metrics = ["val_accuracy", "val_auc"]
x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, [linear_results[m] for m in metrics], width, label="Logistic Regression Pipeline", color="#4f46e5")
plt.bar(x + width/2, [tree_results[m] for m in metrics], width, label="LightGBM Boosted Trees", color="#10b981")
plt.xticks(x, ["Validation Accuracy", "Validation ROC-AUC"])
plt.ylim(0.85, 0.98)
plt.ylabel("Performance Score")
plt.title("Head-to-Head: Best Linear Model vs Best Tree Ensemble", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.MODEL_COMPARISON_DIR, "head_to_head.png"), dpi=120)
plt.close()

winner = "LightGBM (Tree-Based)" if tree_results["val_accuracy"] > linear_results["val_accuracy"] else "Logistic Regression Pipeline (Linear)"
accuracy_gap = abs(tree_results["val_accuracy"] - linear_results["val_accuracy"])

with open(os.path.join(config.MODEL_COMPARISON_DIR, "model_comparison_report.txt"), "w") as f:
    f.write("Comparing Model Tracks - each track's best configuration, refit\n")
    f.write("and evaluated on the same validation split.\n\n")
    f.write("Linear Models Track - Logistic Regression pipeline (Session 11):\n")
    f.write(f"  Val accuracy: {round(linear_results['val_accuracy'],4)}\n")
    f.write(f"  Val AUC:      {round(linear_results['val_auc'],4)}\n")
    f.write(f"  Val log loss: {round(linear_results['val_log_loss'],4)}\n")
    f.write(f"  Fit time:     {round(linear_results['fit_seconds'],3)}s\n")
    f.write(f"  Learned parameters: {linear_results['n_learned_parameters']}\n\n")
    f.write("Tree-Based Models Track - LightGBM (Sessions 18-19):\n")
    f.write(f"  Val accuracy: {round(tree_results['val_accuracy'],4)}\n")
    f.write(f"  Val AUC:      {round(tree_results['val_auc'],4)}\n")
    f.write(f"  Val log loss: {round(tree_results['val_log_loss'],4)}\n")
    f.write(f"  Fit time:     {round(tree_results['fit_seconds'],3)}s\n")
    f.write(f"  Trees used:   {tree_results['n_trees']}\n\n")
    f.write(f"Accuracy gap between the two tracks' best models: {round(accuracy_gap,4)}\n")
    f.write(f"Better validation accuracy: {winner}\n\n")
    f.write("Every model built so far (validation accuracy):\n")
    for name, acc in sorted(all_models.items(), key=lambda kv: kv[1], reverse=True):
        f.write(f"  {name}: {round(acc,4)}\n")

print(f"Session 20 Cross-Model Tournament finished. Winner: {winner}")
