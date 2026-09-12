# Session 17: Sequential Boosting Foundations (AdaBoost & Gradient Boosting)
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import config

os.makedirs(config.BOOSTING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train_ready.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val_ready.csv"))

drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
feature_cols = [c for c in train.columns if c not in drop_cols]
x_train, y_train = train[feature_cols], train[config.TARGET_COLUMN]
x_val, y_val = val[feature_cols], val[config.TARGET_COLUMN]

# 1. AdaBoost with Decision Stumps
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=config.RANDOM_STATE),
    n_estimators=200,
    learning_rate=1.0,
    random_state=config.RANDOM_STATE,
)
ada.fit(x_train, y_train)
train_acc_ada = accuracy_score(y_train, ada.predict(x_train))
val_acc_ada = accuracy_score(y_val, ada.predict(x_val))

# 2. Gradient Boosting Classifier
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=config.RANDOM_STATE,
)
gb.fit(x_train, y_train)
train_acc_gb = accuracy_score(y_train, gb.predict(x_train))
val_acc_gb = accuracy_score(y_val, gb.predict(x_val))

# Staged accuracy comparison curve
ada_staged = [accuracy_score(y_val, p) for p in ada.staged_predict(x_val)]
gb_staged = [accuracy_score(y_val, p) for p in gb.staged_predict(x_val)]

plt.figure(figsize=(8, 5))
plt.plot(range(1, 201), ada_staged, label="AdaBoost (Stumps)", color="#f43f5e", alpha=0.8)
plt.plot(range(1, 201), gb_staged, label="Gradient Boosting (Depth 3)", color="#4f46e5", linewidth=2)
plt.xlabel("Boosting Iterations (Number of Sequential Trees)")
plt.ylabel("Validation Accuracy")
plt.title("Staged Convergence: AdaBoost vs Gradient Boosting", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.BOOSTING_DIR, "staged_accuracy_curve.png"), dpi=120)
plt.close()

# Sweep learning rate for Gradient Boosting
learning_rates = [0.01, 0.05, 0.1, 0.5]
lr_val_accuracies = []
for lr in learning_rates:
    m = GradientBoostingClassifier(n_estimators=200, learning_rate=lr, max_depth=3, random_state=config.RANDOM_STATE)
    m.fit(x_train, y_train)
    lr_val_accuracies.append(accuracy_score(y_val, m.predict(x_val)))

plt.figure(figsize=(7, 4.5))
plt.bar([str(lr) for lr in learning_rates], lr_val_accuracies, color="#06b6d4")
plt.ylim(0.85, max(lr_val_accuracies) + 0.03)
plt.xlabel("Learning Rate eta")
plt.ylabel("Validation Accuracy")
plt.title("Gradient Boosting Sensitivity to Learning Rate", fontweight="bold")
for i, v in enumerate(lr_val_accuracies):
    plt.text(i, v + 0.003, f"{v:.4f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.BOOSTING_DIR, "learning_rate_curve.png"), dpi=120)
plt.close()

RANDOM_FOREST_VAL_ACC = 0.921

with open(os.path.join(config.BOOSTING_DIR, "boosting_report.txt"), "w") as f:
    f.write("AdaBoost (200 depth-1 stumps):\n")
    f.write(f"  Train accuracy: {round(train_acc_ada,4)}  Val accuracy: {round(val_acc_ada,4)}\n\n")
    f.write("Gradient Boosting (200 trees, depth=3, learning_rate=0.1):\n")
    f.write(f"  Train accuracy: {round(train_acc_gb,4)}  Val accuracy: {round(val_acc_gb,4)}\n\n")
    f.write("Validation accuracy vs learning rate (200 trees):\n")
    for lr, acc in zip(learning_rates, lr_val_accuracies):
        f.write(f"  learning_rate={lr}: {round(acc,4)}\n")
    f.write("\nComparison with Session 16:\n")
    f.write(f"  Random Forest (Session 16, parallel/independent trees): {RANDOM_FOREST_VAL_ACC}\n")
    f.write(f"  AdaBoost (sequential, reweighted rows):                 {round(val_acc_ada,4)}\n")
    f.write(f"  Gradient Boosting (sequential, fits residuals):         {round(val_acc_gb,4)}\n\n")
    f.write("Note: boosting trains trees sequentially (each depends on the\n")
    f.write("previous one's output), so unlike Session 16's bagging/forest it\n")
    f.write("cannot be parallelized across trees with n_jobs.\n")

print("Session 17 Boosting Basics report and figures saved.")
