# Session 21: K-Means Clustering & Cohort Segmentation
# Sweeps cluster count K from 2 to 10 using Inertia (Elbow) and Silhouette validation.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import config

os.makedirs(config.KMEANS_DIR, exist_ok=True)

K_RANGE = range(2, 11)
SILHOUETTE_SAMPLE_SIZE = 5000

df = pd.read_csv(os.path.join(config.SPLITS_DIR, "train.csv"))
feature_cols = config.NUMERIC_COLUMNS

X_raw = df[feature_cols].copy()
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Cluster sweep
k_values, inertias, silhouettes = [], [], []
rng = np.random.default_rng(config.RANDOM_STATE)
sil_idx = rng.choice(len(X_scaled), size=min(SILHOUETTE_SAMPLE_SIZE, len(X_scaled)), replace=False)
X_sil_sample = X_scaled[sil_idx]

for k in K_RANGE:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=config.RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_sil_sample, labels[sil_idx])
    k_values.append(k)
    inertias.append(km.inertia_)
    silhouettes.append(sil)

results_df = pd.DataFrame({"K": k_values, "Inertia": inertias, "Silhouette": silhouettes})
results_df.to_csv(os.path.join(config.KMEANS_DIR, "kmeans_k_selection_results.csv"), index=False)

best_k_silhouette = results_df.loc[results_df["Silhouette"].idxmax(), "K"]
# Simple elbow detection
d1 = np.diff(inertias)
d2 = np.diff(d1)
elbow_k = list(K_RANGE)[np.argmax(d2) + 1] if len(d2) > 0 else 3
OPTIMAL_K = int(best_k_silhouette)

with open(os.path.join(config.KMEANS_DIR, "optimal_k.txt"), "w") as f:
    f.write(f"Optimal K selected = {OPTIMAL_K}\n")
    f.write("Selection rule      = maximum silhouette score across K=2..10\n")
    f.write(f"Silhouette at K*    = {results_df['Silhouette'].max():.4f}\n")
    f.write(f"Elbow-method K      = {elbow_k}\n")

# Visualization: Elbow & Silhouette curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(results_df["K"], results_df["Inertia"], marker="o", color="#4f46e5", linewidth=2.2, markersize=7)
axes[0].set_title("Inertia Reduction (Elbow Method)", fontweight="bold")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Within-Cluster Sum of Squares (Inertia)")
axes[0].set_xticks(list(K_RANGE))
axes[0].grid(True, alpha=0.3)

axes[1].plot(results_df["K"], results_df["Silhouette"], marker="s", color="#10b981", linewidth=2.2, markersize=7)
axes[1].axvline(x=OPTIMAL_K, color="#f43f5e", linestyle="--", label=f"Optimal K = {OPTIMAL_K}")
axes[1].set_title("Silhouette Coefficient Trajectory", fontweight="bold")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Mean Silhouette Score")
axes[1].set_xticks(list(K_RANGE))
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(config.KMEANS_DIR, "elbow_and_silhouette_curves.png"), dpi=130)
plt.close()

# Fit final model at optimal K
final_km = KMeans(n_clusters=OPTIMAL_K, init="k-means++", n_init=10, random_state=config.RANDOM_STATE)
df["Cluster"] = final_km.fit_predict(X_scaled)
df.to_csv(os.path.join(config.KMEANS_DIR, "placement_data_with_clusters.csv"), index=False)

placement_by_cluster = df.groupby("Cluster")[config.TARGET_COLUMN].mean()

with open(os.path.join(config.KMEANS_DIR, "kmeans_report.txt"), "w") as f:
    f.write(f"K-Means clustering, K swept from 2 to 10, fit on {len(X_scaled):,} students\n")
    f.write(f"using {len(feature_cols)} scaled, median-imputed numeric features.\n\n")
    f.write("K, Inertia, Silhouette (sampled):\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nOptimal K by max silhouette: {best_k_silhouette} (silhouette = {results_df['Silhouette'].max():.4f})\n")
    f.write(f"Elbow-method K (max curvature): {elbow_k}\n")
    f.write(f"Recorded optimal K: {OPTIMAL_K}\n\n")
    f.write(f"Final cluster sizes (K={OPTIMAL_K}):\n")
    f.write(df["Cluster"].value_counts().sort_index().to_string())
    f.write("\n\nPlacement rate by cluster (unsupervised structure vs ground truth):\n")
    for cid, rate in placement_by_cluster.items():
        f.write(f"  Cluster {cid}: {round(rate * 100, 1)}% placed\n")

print(f"Session 21 K-Means complete. Optimal K={OPTIMAL_K}, Silhouette={results_df['Silhouette'].max():.4f}")
