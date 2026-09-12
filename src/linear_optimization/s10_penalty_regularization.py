# Session 10: Regularization & Parameter Shrinkage (Ridge, Lasso, Elastic Net)
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import config

os.makedirs(config.REGULARIZATION_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train_ready.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val_ready.csv"))

drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]

# Part 1: Regression Regularization (Target: CodingTestScore)
reg_target = "CodingTestScore"
reg_features = [c for c in train.columns if c not in drop_cols + [reg_target]]
x_train_reg, y_train_reg = train[reg_features], train[reg_target]
x_val_reg, y_val_reg = val[reg_features], val[reg_target]

reg_models = {
    "Linear (no regularization)": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.01),
    "Elastic Net (L1 + L2)": ElasticNet(alpha=0.01, l1_ratio=0.5),
}

reg_results = {}
for name, m in reg_models.items():
    m.fit(x_train_reg, y_train_reg)
    preds = m.predict(x_val_reg)
    mse = mean_squared_error(y_val_reg, preds)
    r2 = r2_score(y_val_reg, preds)
    zeros = (abs(m.coef_) < 1e-6).sum()
    reg_results[name] = {"mse": mse, "r2": r2, "zero_coefs": zeros}

plt.figure(figsize=(9, 5))
plt.plot(reg_models["Linear (no regularization)"].coef_, label="Linear (OLS)", alpha=0.7, color="#64748b")
plt.plot(reg_models["Ridge (L2)"].coef_, label="Ridge (L2 Shrinkage)", alpha=0.8, color="#4f46e5")
plt.plot(reg_models["Lasso (L1)"].coef_, label="Lasso (L1 Sparsity)", alpha=0.8, color="#10b981")
plt.axhline(0, color="gray", linewidth=1, linestyle="--")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Weight")
plt.title("Regularization Shrinkage: Linear vs Ridge vs Lasso Coefficients", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.REGULARIZATION_DIR, "regression_coefficients.png"), dpi=120)
plt.close()

# Part 2: Logistic Regression Penalties
clf_features = [c for c in train.columns if c not in drop_cols]
x_train_clf, y_train_clf = train[clf_features], train[config.TARGET_COLUMN]
x_val_clf, y_val_clf = val[clf_features], val[config.TARGET_COLUMN]

clf_models = {
    "L2 (Ridge-style)": LogisticRegression(solver="saga", l1_ratio=0.0, max_iter=2000, random_state=config.RANDOM_STATE),
    "L1 (Lasso-style)": LogisticRegression(solver="saga", l1_ratio=1.0, max_iter=2000, random_state=config.RANDOM_STATE),
    "Elastic Net": LogisticRegression(solver="saga", l1_ratio=0.5, max_iter=2000, random_state=config.RANDOM_STATE),
}

clf_results = {}
for name, m in clf_models.items():
    m.fit(x_train_clf, y_train_clf)
    acc = accuracy_score(y_val_clf, m.predict(x_val_clf))
    zeros = (abs(m.coef_[0]) < 1e-6).sum()
    clf_results[name] = {"accuracy": acc, "zero_coefs": zeros}

plt.figure(figsize=(7, 4.5))
bars = plt.bar(clf_results.keys(), [v["accuracy"] for v in clf_results.values()], color=["#4f46e5", "#06b6d4", "#10b981"])
plt.ylabel("Validation Accuracy")
plt.title("Classification Accuracy across Penalty Formulations", fontweight="bold")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval - 0.05, f"{yval:.4f}", ha="center", va="bottom", color="white", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.REGULARIZATION_DIR, "logistic_penalty_comparison.png"), dpi=120)
plt.close()

with open(os.path.join(config.REGULARIZATION_DIR, "regularization_report.txt"), "w") as f:
    f.write("Regression (target: CodingTestScore)\n")
    for name, r in reg_results.items():
        f.write(f"  {name}: MSE={round(r['mse'],4)}  R2={round(r['r2'],4)}  zero-weight features={r['zero_coefs']}\n")
    f.write("\nLogistic Regression (target: PlacementStatus)\n")
    for name, r in clf_results.items():
        f.write(f"  {name}: Accuracy={round(r['accuracy'],4)}  zero-weight features={r['zero_coefs']}\n")

print("Session 10 Regularization benchmarks completed.")
