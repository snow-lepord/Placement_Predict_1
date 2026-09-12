# Session 18: Modern Boosted Trees (XGBoost, LightGBM, CatBoost)
# Leverages histogram binning, second-order gradients, and early stopping.
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
import config

os.makedirs(config.MODERN_BOOSTING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train_ready.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val_ready.csv"))

drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
feature_cols = [c for c in train.columns if c not in drop_cols]
x_train, y_train = train[feature_cols], train[config.TARGET_COLUMN]
x_val, y_val = val[feature_cols], val[config.TARGET_COLUMN]

results = {}

# 1. XGBoost
t0 = time.time()
xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    early_stopping_rounds=30,
    random_state=config.RANDOM_STATE,
    eval_metric="logloss"
)
xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
fit_xgb = time.time() - t0
results["XGBoost"] = {
    "train_acc": accuracy_score(y_train, xgb.predict(x_train)),
    "val_acc": accuracy_score(y_val, xgb.predict(x_val)),
    "n_trees_used": xgb.best_iteration,
    "fit_seconds": fit_xgb
}

# 2. LightGBM
t0 = time.time()
lgb = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    random_state=config.RANDOM_STATE,
    verbose=-1
)
lgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])
fit_lgb = time.time() - t0
results["LightGBM"] = {
    "train_acc": accuracy_score(y_train, lgb.predict(x_train)),
    "val_acc": accuracy_score(y_val, lgb.predict(x_val)),
    "n_trees_used": lgb.best_iteration_,
    "fit_seconds": fit_lgb
}

# 3. CatBoost
t0 = time.time()
cb = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=4,
    early_stopping_rounds=30,
    random_seed=config.RANDOM_STATE,
    verbose=False
)
cb.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)
fit_cb = time.time() - t0
results["CatBoost"] = {
    "train_acc": accuracy_score(y_train, cb.predict(x_train)),
    "val_acc": accuracy_score(y_val, cb.predict(x_val)),
    "n_trees_used": cb.get_best_iteration(),
    "fit_seconds": fit_cb
}

# Accuracy vs Training Time scatter
plt.figure(figsize=(8, 5))
names = list(results.keys())
val_accs = [results[n]["val_acc"] for n in names]
fit_times = [results[n]["fit_seconds"] for n in names]
plt.scatter(fit_times, val_accs, s=140, color=["#4f46e5", "#06b6d4", "#10b981"], zorder=3)
for n, t, a in zip(names, fit_times, val_accs):
    plt.annotate(f"{n} ({a:.4f})", (t, a), textcoords="offset points", xytext=(8, 5), fontweight="bold")
plt.xlabel("Training Runtime (Seconds with Early Stopping)")
plt.ylabel("Validation Accuracy")
plt.title("Modern Boosted Trees Pareto Efficiency: Accuracy vs Runtime", fontweight="bold")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.MODERN_BOOSTING_DIR, "accuracy_vs_time.png"), dpi=120)
plt.close()

# Feature importance
xgb_importance = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 4.5))
xgb_importance.sort_values().plot(kind="barh", color="#6366f1")
plt.xlabel("Relative Feature Importance")
plt.title("XGBoost Top 10 Feature Importance Ranking", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.MODERN_BOOSTING_DIR, "xgboost_feature_importance.png"), dpi=120)
plt.close()

SKLEARN_GB_VAL_ACC = 0.9238
SKLEARN_GB_TREES = 200

with open(os.path.join(config.MODERN_BOOSTING_DIR, "modern_boosting_report.txt"), "w") as f:
    f.write("Modern Boosted Trees - all three use early stopping (30 rounds,\n")
    f.write("watching validation loss), so trees used varies per library:\n\n")
    f.write(f"sklearn GradientBoosting (Session 17 baseline): val accuracy {SKLEARN_GB_VAL_ACC}, {SKLEARN_GB_TREES} trees (fixed)\n\n")
    for name, r in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Train accuracy: {round(r['train_acc'],4)}\n")
        f.write(f"  Val accuracy:   {round(r['val_acc'],4)}\n")
        f.write(f"  Trees used (early stopping): {r['n_trees_used']}\n")
        f.write(f"  Fit time: {round(r['fit_seconds'],2)}s\n\n")
    f.write("Top 10 features by XGBoost importance:\n")
    for feature, score in xgb_importance.items():
        f.write(f"  {feature}: {round(score,4)}\n")
    f.write("\nWhat each library adds over sklearn's GradientBoostingClassifier:\n")
    f.write("  XGBoost:  regularized objective (L1/L2 on leaf weights), second-order\n")
    f.write("            gradient approximation, level-wise tree growth.\n")
    f.write("  LightGBM: leaf-wise tree growth and histogram-binned splits - built\n")
    f.write("            for speed on large datasets.\n")
    f.write("  CatBoost: native categorical handling via ordered target encoding,\n")
    f.write("            symmetric (oblivious) trees for fast inference.\n")

print("Session 18 Modern Boosted Trees comparison complete.")
