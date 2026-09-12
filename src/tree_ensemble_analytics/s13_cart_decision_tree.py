# Session 13: CART Decision Tree Classification & Visual Splitting
# Demonstrates tree splitting thresholds, unconstrained vs shallow architectures, and feature importance.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import config

os.makedirs(config.DECISION_TREE_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train_ready.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val_ready.csv"))

drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
feature_cols = [c for c in train.columns if c not in drop_cols]
x_train, y_train = train[feature_cols], train[config.TARGET_COLUMN]
x_val, y_val = val[feature_cols], val[config.TARGET_COLUMN]

# Unconstrained Tree
full_tree = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
full_tree.fit(x_train, y_train)
train_acc_full = accuracy_score(y_train, full_tree.predict(x_train))
val_acc_full = accuracy_score(y_val, full_tree.predict(x_val))

# Constrained Shallow Tree (max_depth=3)
shallow_tree = DecisionTreeClassifier(max_depth=3, random_state=config.RANDOM_STATE)
shallow_tree.fit(x_train, y_train)
train_acc_shallow = accuracy_score(y_train, shallow_tree.predict(x_train))
val_acc_shallow = accuracy_score(y_val, shallow_tree.predict(x_val))

# Tree visualization
plt.figure(figsize=(20, 10))
plot_tree(shallow_tree, feature_names=feature_cols, class_names=["Not Placed", "Placed"], filled=True, fontsize=9, rounded=True)
plt.title("CART Decision Tree Topology (max_depth=3)", fontsize=14, fontweight="bold")
plt.savefig(os.path.join(config.DECISION_TREE_DIR, "tree_visualization.png"), dpi=130, bbox_inches="tight")
plt.close()

# Feature importance plot
top_features = pd.Series(full_tree.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
plt.figure(figsize=(7, 4.5))
top_features.sort_values().plot(kind="barh", color="#4f46e5")
plt.xlabel("Gini Impurity Reduction Importance")
plt.title("Top 10 Decision Tree Features", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.DECISION_TREE_DIR, "feature_importance.png"), dpi=120)
plt.close()

with open(os.path.join(config.DECISION_TREE_DIR, "decision_tree_report.txt"), "w") as f:
    f.write("Unconstrained tree (no depth limit):\n")
    f.write(f"  Train accuracy: {round(train_acc_full, 4)}\n")
    f.write(f"  Val accuracy:   {round(val_acc_full, 4)}\n")
    f.write(f"  Tree depth:     {full_tree.get_depth()}\n")
    f.write("  -> Massive generalization gap indicates pure sample memorization.\n\n")
    f.write("Shallow tree (max_depth=3):\n")
    f.write(f"  Train accuracy: {round(train_acc_shallow, 4)}\n")
    f.write(f"  Val accuracy:   {round(val_acc_shallow, 4)}\n\n")
    f.write("Top 10 features by Gini importance:\n")
    for feat, val in top_features.items():
        f.write(f"  {feat:<25}: {round(val, 4)}\n")

print("Session 13 Decision Tree report and artifacts generated.")
