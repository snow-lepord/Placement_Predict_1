# Session 24: Hierarchical Agglomerative Clustering & Dendrogram Diagnostics
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
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
import config

os.makedirs(config.HIERARCHICAL_DIR, exist_ok=True)

SUBSAMPLE_SIZE = 2000

df = pd.read_csv(os.path.join(config.SPLITS_DIR, "train.csv"))
feature_cols = config.NUMERIC_COLUMNS
x_raw = df[feature_cols].copy()

imputer = SimpleImputer(strategy="median")
x_imputed = pd.DataFrame(imputer.fit_transform(x_raw), columns=feature_cols, index=x_raw.index)

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x_imputed), columns=feature_cols, index=x_raw.index)

sample_idx = df.sample(n=SUBSAMPLE_SIZE, random_state=config.RANDOM_STATE).index
x_sample = x_scaled.loc[sample_idx]
placement_sample = df.loc[sample_idx, config.TARGET_COLUMN].values

dist_matrix = pdist(x_sample, metric="euclidean")

methods = ["ward", "complete", "average", "single"]
linkage_matrices = {}
cophenetic_scores = {}

for m in methods:
    Z = linkage(dist_matrix if m != "ward" else x_sample, method=m)
    linkage_matrices[m] = Z
    c, _ = cophenet(Z, dist_matrix)
    cophenetic_scores[m] = c

# 1. Four Linkage Dendrograms Subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, m in enumerate(methods):
    ax = axes[idx]
    dendrogram(linkage_matrices[m], p=30, truncate_mode="lastp", ax=ax, no_labels=True, color_threshold=0)
    ax.set_title(f"{m.capitalize()} Linkage (Cophenetic Corr: {cophenetic_scores[m]:.3f})", fontweight="bold")
    ax.set_ylabel("Merge Distance / Dissimilarity")

plt.tight_layout()
plt.savefig(os.path.join(config.HIERARCHICAL_DIR, "dendrograms_by_linkage.png"), dpi=130)
plt.close()

# 2. Silhouette across cut heights
k_range = range(2, 11)
silhouette_by_k = []
for k in k_range:
    labels = fcluster(linkage_matrices["ward"], t=k, criterion="maxclust")
    silhouette_by_k.append(silhouette_score(x_sample, labels))

best_k_idx = int(np.argmax(silhouette_by_k))
best_k_hier = list(k_range)[best_k_idx]

plt.figure(figsize=(8, 4.5))
plt.plot(list(k_range), silhouette_by_k, marker="o", color="#4f46e5", linewidth=2.2)
plt.axvline(best_k_hier, color="#f43f5e", linestyle="--", label=f"Optimal Cut (K={best_k_hier})")
plt.xlabel("Number of Clusters (Cut Height K)")
plt.ylabel("Mean Silhouette Score")
plt.title("Hierarchical Cluster Quality across Cut Heights (Ward Linkage)", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.HIERARCHICAL_DIR, "silhouette_by_cut_height.png"), dpi=120)
plt.close()

single_labels = fcluster(linkage_matrices["single"], t=4, criterion="maxclust")
ward_labels = fcluster(linkage_matrices["ward"], t=4, criterion="maxclust")
single_cluster_sizes = pd.Series(single_labels).value_counts().sort_index()
ward_cluster_sizes = pd.Series(ward_labels).value_counts().sort_index()

SESSION21_OPTIMAL_K = 2
hier_labels_at_2 = fcluster(linkage_matrices["ward"], t=SESSION21_OPTIMAL_K, criterion="maxclust")
kmeans_sample = KMeans(n_clusters=SESSION21_OPTIMAL_K, init="k-means++", n_init=10, random_state=config.RANDOM_STATE)
kmeans_labels = kmeans_sample.fit_predict(x_sample)
ari = adjusted_rand_score(hier_labels_at_2, kmeans_labels)

placement_by_hier_cluster = pd.Series(placement_sample).groupby(hier_labels_at_2).mean()

with open(os.path.join(config.HIERARCHICAL_DIR, "hierarchical_report.txt"), "w") as f:
    f.write(f"Hierarchical (agglomerative) clustering, {SUBSAMPLE_SIZE:,}-row subsample\n")
    f.write(f"of the {len(df):,}-row training split, {len(feature_cols)} scaled, median-\n")
    f.write("imputed numeric features (same pipeline as Session 21's K-Means).\n\n")
    f.write("Cophenetic correlation by linkage method (how faithfully the tree\n")
    f.write("preserves actual pairwise distances):\n")
    for method, score in cophenetic_scores.items():
        f.write(f"  {method}: {round(score, 4)}\n")
    f.write(f"\nSingle linkage cut at K=4, cluster sizes: {single_cluster_sizes.to_dict()}\n")
    f.write("  (dominant cluster + near-singletons = chaining effect)\n")
    f.write(f"Ward linkage cut at K=4, cluster sizes: {ward_cluster_sizes.to_dict()}\n")
    f.write("  (balanced sizes - Ward resists chaining)\n\n")
    f.write("Silhouette score by K, cutting Ward's single fitted tree:\n")
    for k, sil in zip(k_range, silhouette_by_k):
        f.write(f"  K={k}: {round(sil, 4)}\n")
    f.write(f"Best K for Ward linkage: {best_k_hier}\n\n")
    f.write(f"Adjusted Rand Index vs K-Means (both K={SESSION21_OPTIMAL_K}, same subsample): {round(ari, 4)}\n\n")
    f.write(f"Placement rate by hierarchical cluster (K={SESSION21_OPTIMAL_K}, label not used to build clusters):\n")
    for cluster_id, rate in placement_by_hier_cluster.items():
        f.write(f"  Cluster {cluster_id}: {round(rate * 100, 1)}% placed\n")
    f.write("\nHierarchical vs K-Means, the core tradeoff:\n")
    f.write("  Hierarchical builds one tree and lets K be chosen after the fact\n")
    f.write("  by cutting at different heights - no refit needed per K. K-Means\n")
    f.write("  needs a full refit for every K tried (Session 21's K=2..10 sweep)\n")
    f.write("  but scales to far larger n, since it never needs the full pairwise\n")
    f.write("  distance matrix hierarchical clustering requires.\n")

print(f"Session 24 Hierarchical Clustering report and dendrograms generated. ARI with K-Means: {round(ari, 4)}")
