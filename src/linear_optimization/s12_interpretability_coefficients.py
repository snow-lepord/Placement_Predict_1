# Session 12: Interpreting Model Coefficients & Attribution
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
import config

os.makedirs(config.COEFFICIENTS_DIR, exist_ok=True)

# Linear regression plain units interpretation
train_raw = pd.read_csv(os.path.join(config.SPLITS_DIR, "train.csv"))
demo = train_raw[["AptitudeTestScore", "CodingTestScore"]].dropna()
linreg = LinearRegression()
linreg.fit(demo[["AptitudeTestScore"]], demo["CodingTestScore"])
slope = linreg.coef_[0]

# Logistic regression standardized coefficients
train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train_ready.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val_ready.csv"))

drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
feature_cols = [c for c in train.columns if c not in drop_cols]
x_train, y_train = train[feature_cols], train[config.TARGET_COLUMN]

model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
model.fit(x_train, y_train)

coef_table = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": model.coef_[0],
})
coef_table["odds_ratio"] = np.exp(coef_table["coefficient"])
coef_table["abs_coefficient"] = coef_table["coefficient"].abs()
coef_table = coef_table.sort_values("abs_coefficient", ascending=False)

# Top coefficients bar chart
top10 = coef_table.head(10).sort_values("coefficient")
plt.figure(figsize=(8, 5))
colors = ["#f43f5e" if c < 0 else "#10b981" for c in top10["coefficient"]]
plt.barh(top10["feature"], top10["coefficient"], color=colors)
plt.axvline(0, color="gray", linewidth=1, linestyle="--")
plt.xlabel("Logistic Regression Coefficient (Standardized)")
plt.title("Top 10 Feature Coefficients Impacting Placement", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.COEFFICIENTS_DIR, "top_coefficients.png"), dpi=120)
plt.savefig(os.path.join(config.COEFFICIENTS_DIR, "feature_importance_coefficients.png"), dpi=120)
plt.close()

# Multicollinearity heatmap
sgpa_cols = [f"SGPA_Sem{i}" for i in range(1, 9)] + ["CGPA"]
plt.figure(figsize=(8, 6))
sns.heatmap(train_raw[sgpa_cols].corr(), annot=True, cmap="vlag", fmt=".2f", center=0)
plt.title("Multicollinearity: SGPA Semesters vs Cumulative CGPA", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.COEFFICIENTS_DIR, "sgpa_cgpa_multicollinearity_heatmap.png"), dpi=120)
plt.close()

with open(os.path.join(config.COEFFICIENTS_DIR, "coefficients_report.txt"), "w") as f:
    f.write(f"Linear regression: each 1 point of AptitudeTestScore adds {round(slope,3)} to CodingTestScore\n\n")
    f.write("Top 10 features by coefficient size (logistic regression):\n")
    for _, row in coef_table.head(10).iterrows():
        direction = "increases odds of placement" if row["coefficient"] > 0 else "decreases odds of placement"
        f.write(f"  {row['feature']:<30} coef={round(row['coefficient'],4):>8}  odds_ratio={round(row['odds_ratio'],4):>7}  ({direction})\n")

print("Session 12 Model interpretability coefficients saved successfully.")
