# Session 15: Bias-Variance Tradeoff via Tree Depth Progression
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import config

os.makedirs(config.BIAS_VARIANCE_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train_ready.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val_ready.csv"))

drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
feature_cols = [c for c in train.columns if c not in drop_cols]
x_train, y_train = train[feature_cols], train[config.TARGET_COLUMN]
x_val, y_val = val[feature_cols], val[config.TARGET_COLUMN]

depths = list(range(1, 26))
train_accs, val_accs = [], []

for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=config.RANDOM_STATE)
    t.fit(x_train, y_train)
    train_accs.append(accuracy_score(y_train, t.predict(x_train)))
    val_accs.append(accuracy_score(y_val, t.predict(x_val)))

best_idx = val_accs.index(max(val_accs))
best_depth = depths[best_idx]
best_val_acc = val_accs[best_idx]

plt.figure(figsize=(9, 5))
plt.plot(depths, train_accs, marker="o", label="Training Accuracy", color="#4f46e5", markersize=4)
plt.plot(depths, val_accs, marker="s", label="Validation Accuracy", color="#10b981", markersize=4)
plt.axvline(best_depth, color="#f43f5e", linestyle="--", label=f"Optimal Depth (d={best_depth})")
plt.xlabel("Max Tree Depth")
plt.ylabel("Accuracy")
plt.title("Bias-Variance Frontier across Tree Depth Hierarchy", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.BIAS_VARIANCE_DIR, "bias_variance_curve.png"), dpi=120)
plt.close()

gaps = [t - v for t, v in zip(train_accs, val_accs)]
with open(os.path.join(config.BIAS_VARIANCE_DIR, "bias_variance_report.txt"), "w") as f:
    f.write(f"Best depth: {best_depth} (validation accuracy = {round(best_val_acc, 4)})\n\n")
    f.write(f"Gap at depth 1 (high bias regime): {round(gaps[0], 4)}\n")
    f.write(f"Gap at depth {depths[-1]} (high variance regime): {round(gaps[-1], 4)}\n\n")
    f.write("Depth progression:\n")
    f.write("Depth | Train Acc | Val Acc | Generalization Gap\n")
    for d, tr, va, g in zip(depths, train_accs, val_accs, gaps):
        f.write(f"{d:>5} | {round(tr,4):>9} | {round(va,4):>7} | {round(g,4):>18}\n")

print(f"Session 15 Bias-Variance Tradeoff complete. Best depth: {best_depth} (Val: {round(best_val_acc, 4)})")
