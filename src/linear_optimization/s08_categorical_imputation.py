# Session 08: Categorical Encoding & Median Imputation
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

os.makedirs(config.ENCODING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train_scaled.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val_scaled.csv"))
test = pd.read_csv(os.path.join(config.SPLITS_DIR, "test_scaled.csv"))

original_scale = pd.read_csv(os.path.join(config.SPLITS_DIR, "train.csv"))
mean_val = original_scale["AptitudeTestScore"].mean()
median_val = original_scale["AptitudeTestScore"].median()

# Median imputation for numeric features
for col in config.NUMERIC_COLUMNS:
    fill_value = train[col].median()
    train[col] = train[col].fillna(fill_value)
    val[col] = val[col].fillna(fill_value)
    test[col] = test[col].fillna(fill_value)

# Mode imputation for text columns
for col in config.TEXT_COLUMNS:
    fill_value = train[col].mode()[0]
    train[col] = train[col].fillna(fill_value)
    val[col] = val[col].fillna(fill_value)
    test[col] = test[col].fillna(fill_value)

# Binary binary mapping
for col in config.YES_NO_COLUMNS:
    train[col] = train[col].replace({"Yes": 1, "No": 0})
    val[col] = val[col].replace({"Yes": 1, "No": 0})
    test[col] = test[col].replace({"Yes": 1, "No": 0})

# Ordinal mapping
tier_map = {"Tier3": 0, "Tier2": 1, "Tier1": 2}
cgpa_tier_map = {"Low": 0, "Mid": 1, "High": 2}
for d in (train, val, test):
    d["CollegeTier"] = d["CollegeTier"].replace(tier_map)
    d["CGPA_Tier"] = d["CGPA_Tier"].replace(cgpa_tier_map)

# One-Hot Encoding
train = pd.get_dummies(train, columns=config.CATEGORY_COLUMNS)
val = pd.get_dummies(val, columns=config.CATEGORY_COLUMNS).reindex(columns=train.columns, fill_value=0)
test = pd.get_dummies(test, columns=config.CATEGORY_COLUMNS).reindex(columns=train.columns, fill_value=0)

train.to_csv(os.path.join(config.SPLITS_DIR, "train_ready.csv"), index=False)
val.to_csv(os.path.join(config.SPLITS_DIR, "val_ready.csv"), index=False)
test.to_csv(os.path.join(config.SPLITS_DIR, "test_ready.csv"), index=False)

# Diagnostic chart: feature expansion
plt.figure(figsize=(7, 4))
plt.bar(["Before Encoding", "After One-Hot Encoding"], [original_scale.shape[1], train.shape[1]], color=["#6366f1", "#10b981"])
plt.ylabel("Total Column Count")
plt.title("Dimensionality Expansion via One-Hot Encoding", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.ENCODING_DIR, "08_imputation_and_features.png"), dpi=120)
plt.close()

with open(os.path.join(config.ENCODING_DIR, "encoding_imputation_report.txt"), "w") as f:
    f.write(f"AptitudeTestScore original-scale mean: {round(mean_val,3)}, median: {round(median_val,3)}\n")
    f.write(f"Final train shape: {train.shape}\n")

print(f"Session 08 Encoding & Imputation complete. Final train dimensions: {train.shape}")
