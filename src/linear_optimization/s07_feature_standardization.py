# Session 07: Feature Standardization & Leakage Prevention
# Fits transformation parameters strictly on the training partition.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import config

os.makedirs(config.SCALING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val.csv"))
test = pd.read_csv(os.path.join(config.SPLITS_DIR, "test.csv"))

cols = config.NUMERIC_COLUMNS

# Compute scaling parameters on training split only
mean = train[cols].mean()
std = train[cols].std()
min_val = train[cols].min()
max_val = train[cols].max()

comparison = pd.DataFrame({
    "original": train["CGPA"].head(),
    "standardized": ((train["CGPA"] - mean["CGPA"]) / std["CGPA"]).head(),
    "min_max": ((train["CGPA"] - min_val["CGPA"]) / (max_val["CGPA"] - min_val["CGPA"])).head(),
})

# Diagnostic plot: scaling comparison
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.kdeplot(train["CGPA"], ax=axes[0], color="#4f46e5", fill=True)
axes[0].set_title("Original CGPA Distribution", fontweight="bold")
sns.kdeplot((train["CGPA"] - mean["CGPA"]) / std["CGPA"], ax=axes[1], color="#10b981", fill=True)
axes[1].set_title("Standardized CGPA (Mean=0, Std=1)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.SCALING_DIR, "07_scaling_distribution_comparison.png"), dpi=120)
plt.close()

# Apply standardization
train[cols] = (train[cols] - mean) / std
val[cols] = (val[cols] - mean) / std
test[cols] = (test[cols] - mean) / std

train.to_csv(os.path.join(config.SPLITS_DIR, "train_scaled.csv"), index=False)
val.to_csv(os.path.join(config.SPLITS_DIR, "val_scaled.csv"), index=False)
test.to_csv(os.path.join(config.SPLITS_DIR, "test_scaled.csv"), index=False)

with open(os.path.join(config.SCALING_DIR, "scaling_report.txt"), "w") as f:
    f.write("Method used for the pipeline: Standardization\n")
    f.write("Comparison on CGPA (first 5 rows):\n")
    f.write(comparison.round(3).to_string())

print("Session 07 Feature scaling completed without data leakage.")
