# Session 19: SHAP (SHapley Additive exPlanations) & Individual Attribution
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.stats import spearmanr
import config

os.makedirs(config.SHAP_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train_ready.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val_ready.csv"))

drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
feature_cols = [c for c in train.columns if c not in drop_cols]
x_train, y_train = train[feature_cols], train[config.TARGET_COLUMN]
x_val, y_val = val[feature_cols], val[config.TARGET_COLUMN]

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    random_state=config.RANDOM_STATE,
    verbose=-1,
)
model.fit(x_train, y_train, eval_set=[(x_val, y_val)], callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])

# Compute SHAP Values
explainer = shap.TreeExplainer(model)
sample_val = x_val.sample(1000, random_state=config.RANDOM_STATE)
shap_values = explainer.shap_values(sample_val)

if isinstance(shap_values, list):
    shap_vals_matrix = shap_values[1]
else:
    shap_vals_matrix = shap_values

# 1. Summary Beeswarm Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_matrix, sample_val, show=False, max_display=12)
plt.title("SHAP Global Summary Beeswarm Plot", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(config.SHAP_DIR, "shap_summary_beeswarm.png"), dpi=130, bbox_inches="tight")
plt.close()

# 2. Importance Comparison: Impurity vs Mean |SHAP|
builtin_imp = pd.Series(model.feature_importances_, index=feature_cols)
builtin_norm = builtin_imp / builtin_imp.sum()
shap_imp = pd.Series(np.abs(shap_vals_matrix).mean(axis=0), index=feature_cols)
shap_norm = shap_imp / shap_imp.sum()

comp_df = pd.DataFrame({"Impurity (Gini/Split)": builtin_norm, "Mean |SHAP|": shap_norm})
top_shap_cols = shap_norm.sort_values(ascending=False).head(10).index
comp_plot = comp_df.loc[top_shap_cols]

plt.figure(figsize=(8, 5))
comp_plot.sort_values("Mean |SHAP|").plot(kind="barh", color=["#64748b", "#4f46e5"], width=0.7)
plt.xlabel("Normalized Importance Metric")
plt.title("Feature Attribution: Impurity-Based vs SHAP Values", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.SHAP_DIR, "importance_comparison.png"), dpi=120)
plt.close()

# 3. Individual Waterfall Plot
example_idx = 0
example_pred = model.predict_proba(x_val.iloc[[example_idx]])[0, 1]
single_shap = explainer(x_val.iloc[[example_idx]])

plt.figure()
shap.plots.waterfall(single_shap[0], show=False, max_display=12)
plt.tight_layout()
plt.savefig(os.path.join(config.SHAP_DIR, "shap_waterfall_example.png"), dpi=130, bbox_inches="tight")
plt.close()

top_builtin = builtin_norm.sort_values(ascending=False).head(10)
top_shap = shap_norm.sort_values(ascending=False).head(10)
rank_corr, _ = spearmanr(builtin_norm, shap_norm)

base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = base_val[1]
base_val_prob = 1 / (1 + np.exp(-base_val))

with open(os.path.join(config.SHAP_DIR, "shap_report.txt"), "w") as f:
    f.write(f"Explaining LightGBM ({model.best_iteration_} trees, Session 18's model)\n\n")
    f.write("Top 10 by built-in (impurity-based) importance, normalized:\n")
    for feat, sc in top_builtin.items():
        f.write(f"  {feat}: {round(sc,4)}\n")
    f.write("\nTop 10 by mean |SHAP value|, normalized:\n")
    for feat, sc in top_shap.items():
        f.write(f"  {feat}: {round(sc,4)}\n")
    f.write(f"\nSpearman rank correlation between the two rankings: {round(rank_corr,4)}\n")
    f.write("\nExample row explained (validation row 0):\n")
    f.write(f"  Predicted placement probability: {round(example_pred,4)}\n")
    f.write(f"  Base value: {round(base_val,4)} log-odds ({round(base_val_prob,4)} as a probability)\n")
    f.write("\nWhy use SHAP over built-in importance:\n")
    f.write("  - Built-in importance is a single number per feature, averaged\n")
    f.write("    over the whole training set, biased toward continuous/high-\n")
    f.write("    cardinality features, and has no sign (no direction).\n")
    f.write("  - SHAP gives one value per feature PER ROW, has a sign showing\n")
    f.write("    which direction the feature pushed that specific prediction,\n")
    f.write("    and the row's SHAP values sum exactly to (prediction - base\n")
    f.write("    value) - so it's an actual explanation, not just a ranking.\n")

print("Session 19 SHAP feature explanations saved successfully.")
