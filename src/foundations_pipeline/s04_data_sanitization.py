# Session 04: Cohort Sanitization & IQR Outlier Auditing
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import config

os.makedirs(config.CLEANING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(config.CLEAN_DATA_PATH), exist_ok=True)

df = pd.read_csv(config.RAW_DATA_PATH)
rows_before = df.shape[0]

# Remove duplicate entries
df = df.drop_duplicates()

# IQR Outlier identification
outlier_check_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
outliers_found = {}
for col in outlier_check_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    outliers_found[col] = count

# Range validity enforcement
score_cols = ["AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "AttendancePercent"]
for col in score_cols:
    df[col] = df[col].clip(0, 100)

# Normalize string casing
for col in config.TEXT_COLUMNS:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

df.to_csv(config.CLEAN_DATA_PATH, index=False)

with open(os.path.join(config.CLEANING_DIR, "cleaning_report.txt"), "w") as f:
    f.write(f"Rows before: {rows_before}\nRows after: {df.shape[0]}\n\n")
    f.write("Outliers found per column (IQR method):\n")
    for col in outlier_check_cols:
        f.write(f"{col}: {outliers_found[col]}\n")

print(f"Session 04 Sanitization complete. Clean data saved to {config.CLEAN_DATA_PATH}")
