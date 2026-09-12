# Session 16: Random Forests & Bagging Ensembles
# Ensembles decorrelate predictions via bootstrap aggregation and random feature subsets.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import config

os.makedirs(config.RANDOM_FOREST_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train_ready.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val_ready.csv"))

drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
feature_cols = [c for c in train.columns if c not in drop_cols]
x_train, y_train = train[feature_cols], train[config.TARGET_COLUMN]
x_val, y_val = val[feature_cols], val[config.TARGET_COLUMN]

BEST_SINGLE_TREE_DEPTH = 6
single_tree = DecisionTreeClassifier(max_depth=BEST_SINGLE_TREE_DEPTH, random_state=config.RANDOM_STATE)
single_tree.fit(x_train, y_train)
train_acc_tree = accuracy_score(y_train, single_tree.predict(x_train))
val_acc_tree = accuracy_score(y_val, single_tree.predict(x_val))

# Bagging Classifier
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=config.RANDOM_STATE),
    n_estimators=200,
    oob_score=True,
    random_state=config.RANDOM_STATE,
    n_jobs=-1
)
bagging.fit(x_train, y_train)
train_acc_bag = accuracy_score(y_train, bagging.predict(x_train))
val_acc_bag = accuracy_score(y_val, bagging.predict(x_val))

# Random Forest Classifier
forest = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    oob_score=True,
    random_state=config.RANDOM_STATE,
    n_jobs=-1
)
forest.fit(x_train, y_train)
train_acc_rf = accuracy_score(y_train, forest.predict(x_train))
val_acc_rf = accuracy_score(y_val, forest.predict(x_val))

# Sweep n_estimators
tree_counts = [10, 25, 50, 100, 150, 200]
rf_val_accuracies = []
for n in tree_counts:
    rf_sub = RandomForestClassifier(n_estimators=n, max_features="sqrt", random_state=config.RANDOM_STATE, n_jobs=-1)
    rf_sub.fit(x_train, y_train)
    rf_val_accuracies.append(accuracy_score(y_val, rf_sub.predict(x_val)))

plt.figure(figsize=(8, 4.5))
plt.plot(tree_counts, rf_val_accuracies, marker="o", color="#4f46e5", linewidth=2)
plt.xlabel("Number of Estimators (n_estimators)")
plt.ylabel("Validation Accuracy")
plt.title("Random Forest Generalization vs Ensemble Size", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.RANDOM_FOREST_DIR, "n_estimators_curve.png"), dpi=120)
plt.close()

top_features = pd.Series(forest.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
plt.figure(figsize=(8, 5))
top_features.sort_values().plot(kind="barh", color="#06b6d4")
plt.xlabel("Feature Importance (Gini Impurity Decrease)")
plt.title("Random Forest Top 15 Feature Importances", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.RANDOM_FOREST_DIR, "feature_importance.png"), dpi=120)
plt.close()

bag_importance = pd.Series(
    sum(tree.feature_importances_ for tree in bagging.estimators_) / len(bagging.estimators_),
    index=feature_cols,
)

with open(os.path.join(config.RANDOM_FOREST_DIR, "random_forest_report.txt"), "w") as f:
    f.write(f"Single tree (max_depth={BEST_SINGLE_TREE_DEPTH}, Session 15's best depth):\n")
    f.write(f"  Train accuracy: {round(train_acc_tree,4)}  Val accuracy: {round(val_acc_tree,4)}\n\n")
    f.write(f"Bagging (200 unconstrained trees):\n")
    f.write(f"  Train accuracy: {round(train_acc_bag,4)}  Val accuracy: {round(val_acc_bag,4)}  OOB: {round(bagging.oob_score_,4)}\n\n")
    f.write(f"Random Forest (200 trees, max_features='sqrt'):\n")
    f.write(f"  Train accuracy: {round(train_acc_rf,4)}  Val accuracy: {round(val_acc_rf,4)}  OOB: {round(forest.oob_score_,4)}\n\n")
    f.write("Train-val gap (each ensemble tree is individually unconstrained,\n")
    f.write("so its own gap is larger than the single pruned tree's - the win\n")
    f.write("is in val accuracy from averaging, not a smaller per-tree gap):\n")
    f.write(f"  Single tree:   {round(train_acc_tree - val_acc_tree,4)}\n")
    f.write(f"  Bagging:       {round(train_acc_bag - val_acc_bag,4)}\n")
    f.write(f"  Random Forest: {round(train_acc_rf - val_acc_rf,4)}\n\n")
    f.write("Validation accuracy vs n_estimators:\n")
    for n, acc in zip(tree_counts, rf_val_accuracies):
        f.write(f"  n_estimators={n}: {round(acc,4)}\n")
    f.write(f"\nTop 15 features by Random Forest importance:\n")
    for feature, score in top_features.items():
        f.write(f"  {feature}: {round(score,4)}\n")
    f.write(f"\nTop feature's importance share - Bagging: {bag_importance.max():.4f} ({bag_importance.idxmax()})")
    f.write(f"  vs Random Forest: {top_features.max():.4f} ({top_features.idxmax()})\n")
    f.write("Random Forest spreads importance more evenly across correlated\n")
    f.write("features because max_features randomly excludes the dominant\n")
    f.write("feature from many splits, forcing trees to use alternatives.\n")

print(f"Session 16 Random Forest complete. RF Val Acc: {round(val_acc_rf, 4)}, OOB: {round(forest.oob_score_, 4)}")
