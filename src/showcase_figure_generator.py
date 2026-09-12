# Visual Gallery Showcase Generator
# Produces high-resolution visual analytics for the CampusPlacementIQ dashboard.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import config

os.makedirs(config.EDA_DIR, exist_ok=True)

df = pd.read_csv(config.RAW_DATA_PATH)
df["PlacementLabel"] = df[config.TARGET_COLUMN].replace({1: "Placed", 0: "Not Placed"})

# Palette definitions
COLORS_PRIMARY = ["#10b981", "#f43f5e"]
PALETTE_TIER = ["#6366f1", "#06b6d4", "#f59e0b"]
PALETTE_STREAM = sns.color_palette("mako", n_colors=df["Stream"].nunique())

# 1. Placement Outcome Distribution Pie Chart
plt.figure(figsize=(6.5, 6.5))
placed_counts = df["PlacementLabel"].value_counts()
plt.pie(placed_counts, labels=placed_counts.index, autopct="%1.1f%%", startangle=140,
        colors=COLORS_PRIMARY, explode=(0.04, 0), shadow=True,
        textprops={"fontsize": 12, "fontweight": "bold", "color": "#1e293b"})
plt.title("Cohort Placement Distribution (50,000 Students)", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "pie_placement_distribution.png"), dpi=130)
plt.close()

# 2. College Tier Donut Chart
plt.figure(figsize=(6.5, 6.5))
tier_counts = df["CollegeTier"].value_counts()
wedges, texts, autotexts = plt.pie(tier_counts, labels=tier_counts.index, autopct="%1.1f%%",
                                   startangle=90, colors=PALETTE_TIER,
                                   wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2),
                                   textprops={"fontsize": 11, "fontweight": "bold"})
for at in autotexts:
    at.set_color("white")
plt.title("Student Distribution by College Tier", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "donut_college_tier.png"), dpi=130)
plt.close()

# 3. Stream Breakdown Pie Chart
plt.figure(figsize=(7, 7))
stream_counts = df["Stream"].value_counts()
plt.pie(stream_counts, labels=stream_counts.index, autopct="%1.1f%%", startangle=120,
        colors=PALETTE_STREAM, textprops={"fontsize": 10, "fontweight": "medium"})
plt.title("Academic Stream Representation", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "pie_stream_breakdown.png"), dpi=130)
plt.close()

# 4. Placement Rate by College Tier
tier_rates = df.groupby("CollegeTier")[config.TARGET_COLUMN].mean() * 100
plt.figure(figsize=(7, 4.5))
bars = plt.bar(tier_rates.index, tier_rates.values, color=["#6366f1", "#06b6d4", "#10b981"], width=0.55)
plt.ylabel("Placement Rate (%)")
plt.title("Placement Success Rate by Institutional Tier", fontsize=12, fontweight="bold")
plt.ylim(0, 100)
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, y + 2, f"{y:.1f}%", ha="center", fontweight="bold")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "bar_placement_by_tier.png"), dpi=130)
plt.close()

# 5. Average Salary by Stream
placed_df = df[df["Salary Package"] > 0]
stream_sal = placed_df.groupby("Stream")["Salary Package"].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 4.5))
bars = plt.barh(stream_sal.index, stream_sal.values, color="#4f46e5", height=0.6)
plt.xlabel("Average Salary Package (LPA)")
plt.title("Mean Compensation Package by Academic Stream", fontsize=12, fontweight="bold")
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.1, bar.get_y() + bar.get_height()/2.0, f"₹{w:.2f}L", va="center", fontweight="bold")
plt.xlim(0, max(stream_sal.values) + 1.5)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "bar_stream_average_salary.png"), dpi=130)
plt.close()

# 6. Backlogs Impact on Placement
backlog_rate = df.groupby("HistoryOfBacklogs")[config.TARGET_COLUMN].mean() * 100
plt.figure(figsize=(6, 4.5))
bars = plt.bar(backlog_rate.index, backlog_rate.values, color=["#10b981", "#f43f5e"], width=0.5)
plt.ylabel("Placement Rate (%)")
plt.title("Impact of Active/Historical Backlogs on Placement", fontsize=12, fontweight="bold")
plt.ylim(0, 100)
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, y + 2, f"{y:.1f}%", ha="center", fontweight="bold")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "bar_backlogs_vs_placement.png"), dpi=130)
plt.close()

# 7. SGPA Trajectory Trend (Sem 1 to Sem 8)
sgpa_cols = [f"SGPA_Sem{i}" for i in range(1, 9)]
semesters = [f"Sem {i}" for i in range(1, 9)]
mean_placed = df[df[config.TARGET_COLUMN] == 1][sgpa_cols].mean().values
mean_unplaced = df[df[config.TARGET_COLUMN] == 0][sgpa_cols].mean().values

plt.figure(figsize=(8.5, 4.8))
plt.plot(semesters, mean_placed, marker="o", color="#10b981", linewidth=2.5, label="Placed Students (Cohort Mean)")
plt.plot(semesters, mean_unplaced, marker="s", color="#f43f5e", linewidth=2.5, linestyle="--", label="Unplaced Students (Cohort Mean)")
plt.xlabel("Academic Semester Progression")
plt.ylabel("Mean SGPA")
plt.title("Semester-by-Semester SGPA Trajectory by Placement Outcome", fontsize=12, fontweight="bold")
plt.ylim(5.0, 9.0)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.EDA_DIR, "sgpa_trajectory_trend.png"), dpi=130)
plt.close()

print("All showcase figures generated successfully.")
