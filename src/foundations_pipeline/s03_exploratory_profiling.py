# Session 03: Exploratory Data Profiling & Distribution Diagnostics
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import config

os.makedirs(config.EDA_DIR, exist_ok=True)

df = pd.read_csv(config.RAW_DATA_PATH)
df["PlacementLabel"] = df[config.TARGET_COLUMN].replace({1: "Placed", 0: "Not Placed"})

# Palette styling
sns.set_theme(style="whitegrid", palette="muted")

# 1. Univariate Numeric
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["CGPA"].dropna(), bins=20, ax=axes[0], kde=True, color="#4f46e5")
axes[0].set_title("Distribution of CGPA", fontweight="bold")
sns.histplot(df["AptitudeTestScore"].dropna(), bins=20, ax=axes[1], kde=True, color="#06b6d4")
axes[1].set_title("Distribution of Aptitude Test Score", fontweight="bold")
sns.histplot(df["CodingTestScore"].dropna(), bins=20, ax=axes[2], kde=True, color="#10b981")
axes[2].set_title("Distribution of Coding Test Score", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "03_univariate_numeric.png"), dpi=120)
plt.close()

# 2. Univariate Categorical
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x="Stream", data=df, ax=axes[0], palette="viridis")
axes[0].set_title("Student Distribution by Academic Stream", fontweight="bold")
axes[0].tick_params(axis="x", rotation=25)
sns.countplot(x="Gender", data=df, ax=axes[1], palette="Set2")
axes[1].set_title("Gender Breakdown in Cohort", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "03_univariate_categorical.png"), dpi=120)
plt.close()

# 3. Bivariate Numeric vs Target
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(x="PlacementLabel", y="CGPA", data=df, ax=axes[0], palette=["#f43f5e", "#10b981"])
axes[0].set_title("CGPA vs Placement Outcome", fontweight="bold")
sns.boxplot(x="PlacementLabel", y="CodingTestScore", data=df, ax=axes[1], palette=["#f43f5e", "#10b981"])
axes[1].set_title("Coding Test Score vs Placement Outcome", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "03_bivariate_numeric_vs_target.png"), dpi=120)
plt.close()

# 4. Bivariate Categorical vs Target
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pd.crosstab(df["HistoryOfBacklogs"], df["PlacementLabel"]).plot(kind="bar", ax=axes[0], color=["#f43f5e", "#10b981"])
axes[0].set_title("Backlogs History vs Placement Rate", fontweight="bold")
axes[0].tick_params(axis="x", rotation=0)
pd.crosstab(df["CollegeTier"], df["PlacementLabel"]).plot(kind="bar", ax=axes[1], color=["#f43f5e", "#10b981"])
axes[1].set_title("College Tier vs Placement Rate", fontweight="bold")
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "03_bivariate_categorical_vs_target.png"), dpi=120)
plt.close()

# 5. Bivariate Numeric vs Numeric
plt.figure(figsize=(7, 5))
sns.scatterplot(x="CGPA", y="CodingTestScore", hue="PlacementLabel", data=df.sample(2000, random_state=42), alpha=0.5, s=18, palette={"Placed": "#10b981", "Not Placed": "#f43f5e"})
plt.title("CGPA vs Coding Test Score (Sampled Cohort)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "03_bivariate_numeric_vs_numeric.png"), dpi=120)
plt.close()

# 6. Multivariate
sample_df = df.sample(3000, random_state=42)
grid = sns.FacetGrid(sample_df, col="Stream", hue="PlacementLabel", col_wrap=3, height=3.2, palette={"Placed": "#10b981", "Not Placed": "#f43f5e"})
grid.map(sns.scatterplot, "CGPA", "CodingTestScore", alpha=0.4, s=14)
grid.add_legend()
grid.savefig(os.path.join(config.EDA_DIR, "03_multivariate.png"), dpi=120)
plt.close()

# 7. Correlation Heatmap
corr_cols = ["CGPA", "AttendancePercent", "Internships", "Projects", "AptitudeTestScore", "SoftSkillsRating", "CodingTestScore", "MockInterviewScore"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="mako", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Key Numerical Indicators", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "03_correlation_heatmap.png"), dpi=120)
plt.close()

print("Session 03 Exploratory Profiling plots generated successfully.")
