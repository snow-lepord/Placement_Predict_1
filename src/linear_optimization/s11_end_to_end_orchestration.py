# Session 11: End-to-End Scikit-Learn Pipeline Orchestration
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import config

os.makedirs(config.PIPELINE_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val.csv"))

feature_cols = config.NUMERIC_COLUMNS + config.CATEGORY_COLUMNS + config.YES_NO_COLUMNS + ["CollegeTier", "CGPA_Tier"]
x_train, y_train = train[feature_cols], train[config.TARGET_COLUMN]
x_val, y_val = val[feature_cols], val[config.TARGET_COLUMN]

numeric_steps = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

ordinal_steps = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OrdinalEncoder(categories=[["Tier3", "Tier2", "Tier1"], ["Low", "Mid", "High"]])),
])

categorical_steps = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_steps, config.NUMERIC_COLUMNS),
    ("ordinal", ordinal_steps, ["CollegeTier", "CGPA_Tier"]),
    ("categorical", categorical_steps, config.CATEGORY_COLUMNS + config.YES_NO_COLUMNS),
])

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
])

pipeline.fit(x_train, y_train)
val_accuracy = accuracy_score(y_val, pipeline.predict(x_val))
val_probs = pipeline.predict_proba(x_val)[:, 1]

# Prediction distribution plot
plt.figure(figsize=(7, 4.5))
plt.hist(val_probs, bins=25, color="#4f46e5", edgecolor="white")
plt.axvline(0.5, color="#f43f5e", linestyle="--", linewidth=1.5, label="Decision Threshold (0.50)")
plt.xlabel("Predicted Placement Probability")
plt.ylabel("Student Count")
plt.title("Pipeline Validation Probability Distribution", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.PIPELINE_DIR, "pipeline_prediction_distribution.png"), dpi=120)
plt.close()

# Architecture diagram
fig, ax = plt.subplots(figsize=(9, 3))
ax.axis("off")
flow_text = (
    "Raw Data Split\n(train.csv / val.csv)\n      │\n      ▼\n"
    "ColumnTransformer\n├── Numeric -> Median Impute -> StandardScaler\n"
    "├── Ordinal -> Mode Impute -> OrdinalEncoder\n"
    "└── Categorical -> Mode Impute -> OneHotEncoder\n      │\n      ▼\n"
    "LogisticRegression Classifier\n(Global Fit & Validation Evaluation)"
)
ax.text(0.5, 0.5, flow_text, ha="center", va="center", family="monospace", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.8", fc="#f8fafc", ec="#cbd5e1", lw=1.5))
plt.tight_layout()
plt.savefig(os.path.join(config.PIPELINE_DIR, "pipeline_architecture_flow.png"), dpi=120)
plt.close()

with open(os.path.join(config.PIPELINE_DIR, "pipeline_report.txt"), "w") as f:
    f.write(f"Pipeline validation accuracy: {round(val_accuracy, 4)}\n")
    f.write("Preprocessing steps:\n")
    f.write(f"  Numeric columns ({len(config.NUMERIC_COLUMNS)}): SimpleImputer(median) -> StandardScaler\n")
    f.write("  Ordinal columns (2): SimpleImputer(most_frequent) -> OrdinalEncoder\n")
    f.write(f"  Categorical columns ({len(config.CATEGORY_COLUMNS) + len(config.YES_NO_COLUMNS)}): SimpleImputer(most_frequent) -> OneHotEncoder\n")
    f.write("Model: LogisticRegression(max_iter=1000)\n")

print(f"Session 11 Pipeline fit complete. Val Acc: {round(val_accuracy, 4)}")
