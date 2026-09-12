# Session 09: Logistic Regression Classification & Decision Boundaries
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import config

os.makedirs(config.LOGISTIC_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val.csv"))

feature_cols = config.NUMERIC_COLUMNS
x_train, y_train = train[feature_cols].copy(), train[config.TARGET_COLUMN]
x_val, y_val = val[feature_cols].copy(), val[config.TARGET_COLUMN]

for col in feature_cols:
    fill_value = x_train[col].median()
    x_train[col] = x_train[col].fillna(fill_value)
    x_val[col] = x_val[col].fillna(fill_value)

placed_pct = (y_train == 1).mean() * 100

# 1. CGPA vs Placement empirical S-curve
bins = pd.cut(train["CGPA"], bins=15)
fraction_placed = train.groupby(bins, observed=True)[config.TARGET_COLUMN].mean()
bin_centers = [interval.mid for interval in fraction_placed.index]

plt.figure(figsize=(7, 5))
plt.plot(bin_centers, fraction_placed.values, marker="o", color="#4f46e5", linewidth=2)
plt.xlabel("CGPA")
plt.ylabel("Fraction Placed")
plt.title("CGPA vs Placement Empirical S-Curve", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.LOGISTIC_DIR, "cgpa_vs_placement_curve.png"), dpi=120)
plt.close()

# 2. Sigmoid plot
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z_values = np.linspace(-8, 8, 100)
plt.figure(figsize=(7, 5))
plt.plot(z_values, sigmoid(z_values), color="#06b6d4", linewidth=2.5)
plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
plt.axvline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Log-Odds (z = w^T x + b)")
plt.ylabel("Probability sigma(z)")
plt.title("The Standard Sigmoid Activation Function", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.LOGISTIC_DIR, "sigmoid_function.png"), dpi=120)
plt.close()

# 3. Fit Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
model.fit(x_train, y_train)
val_accuracy = accuracy_score(y_val, model.predict(x_val))
val_log_loss = log_loss(y_val, model.predict_proba(x_val))

# 4. 2D Decision boundary plot
boundary_model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
boundary_model.fit(x_train[["CGPA", "CodingTestScore"]].to_numpy(), y_train)

x_min, x_max = x_train["CGPA"].min() - 0.5, x_train["CGPA"].max() + 0.5
y_min, y_max = x_train["CodingTestScore"].min() - 5, x_train["CodingTestScore"].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_predictions = boundary_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, grid_predictions, alpha=0.25, levels=1, colors=["#f43f5e", "#10b981"])
plt.scatter(x_val["CGPA"], x_val["CodingTestScore"], c=y_val, cmap="coolwarm", alpha=0.3, s=12)
plt.xlabel("CGPA")
plt.ylabel("Coding Test Score")
plt.title("Linear Decision Boundary (CGPA vs Coding Score)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.LOGISTIC_DIR, "decision_boundary.png"), dpi=120)
plt.close()

# AUC per feature
auc_features = ["CGPA", "MockInterviewScore", "CodingTestScore", "AptitudeTestScore", "AttendancePercent"]
auc_scores = {col: roc_auc_score(y_train, x_train[col]) for col in auc_features}

# Scaler comparisons
results = {}
results["Unscaled"] = accuracy_score(y_val, LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE).fit(x_train, y_train).predict(x_val))

ss = StandardScaler()
results["StandardScaler"] = accuracy_score(y_val, LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE).fit(ss.fit_transform(x_train), y_train).predict(ss.transform(x_val)))

mm = MinMaxScaler()
results["MinMaxScaler"] = accuracy_score(y_val, LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE).fit(mm.fit_transform(x_train), y_train).predict(mm.transform(x_val)))

# Softmax 3-class evaluation
salary_median = train.loc[train["Salary Package"] > 0, "Salary Package"].median()
def tier_fn(s):
    if s == 0: return "Not Placed"
    if s < salary_median: return "Standard Package"
    return "Premium Package"

y_train_tier = train["Salary Package"].apply(tier_fn)
y_val_tier = val["Salary Package"].apply(tier_fn)
softmax_model = LogisticRegression(solver="lbfgs", max_iter=1000)
softmax_model.fit(ss.fit_transform(x_train), y_train_tier)
tier_accuracy = accuracy_score(y_val_tier, softmax_model.predict(ss.transform(x_val)))

with open(os.path.join(config.LOGISTIC_DIR, "logistic_regression_report.txt"), "w") as f:
    f.write(f"Placed: {round(placed_pct,1)}%  Not Placed: {round(100-placed_pct,1)}%\n\n")
    f.write("Single-feature AUC:\n")
    for col, auc in auc_scores.items():
        f.write(f"  {col}: {round(auc,4)}\n")
    f.write(f"\nFull model validation accuracy: {round(val_accuracy,4)}\n")
    f.write(f"Full model cross-entropy (log loss): {round(val_log_loss,4)}\n")
    f.write("\nScaler comparison:\n")
    for label, acc in results.items():
        f.write(f"  {label}: {round(acc,4)}\n")
    f.write(f"\nSoftmax 3-class accuracy: {round(tier_accuracy,4)}\n")

print(f"Session 09 Logistic Regression complete. Val Acc: {round(val_accuracy, 4)}")
