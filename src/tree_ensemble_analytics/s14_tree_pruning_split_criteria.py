# Session 14: Tree Pruning & Information Splitting Criteria
# Hand-calculated toy comparison (Gini, Entropy, MSE) + Cost-Complexity Pruning (ccp_alpha).
import sys, os, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import config

os.makedirs(config.SPLITTING_DIR, exist_ok=True)

# Toy cohort for pedagogical verification
students = pd.DataFrame({
    "CGPA":        [8.6, 9.4, 9.0, 5.2, 5.3, 5.0, 6.2, 6.2, 8.7, 7.5, 6.7, 7.5, 6.8, 6.8],
    "Internships": [2,   2,   2,   0,   0,   1,   1,   0,   2,   1,   1,   1,   0,   0],
    "Backlogs":    ["No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No"],
    "Aptitude":    [73,  76,  81,  48,  44,  83,  63,  52,  77,  70,  96,  84,  48,  73],
    "Placed":      ["Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No"],
})

def entropy(labels):
    counts = labels.value_counts()
    probs = counts / len(labels)
    return -sum(p * math.log2(p) for p in probs)

def gini(labels):
    counts = labels.value_counts()
    probs = counts / len(labels)
    return 1 - sum(p ** 2 for p in probs)

def mse(labels):
    num = labels.map({"Yes": 1, "No": 0})
    return ((num - num.mean()) ** 2).mean()

def weighted_eval(l1, l2, fn):
    n = len(l1) + len(l2)
    return (len(l1)/n)*fn(l1) + (len(l2)/n)*fn(l2)

root_entropy = entropy(students["Placed"])
root_gini = gini(students["Placed"])
root_mse = mse(students["Placed"])

# Split on CGPA <= 7.15
cgpa_split = students["CGPA"] <= 7.15
left = students.loc[cgpa_split, "Placed"]
right = students.loc[~cgpa_split, "Placed"]
w_entropy = weighted_eval(left, right, entropy)
w_gini = weighted_eval(left, right, gini)
w_mse = weighted_eval(left, right, mse)
info_gain = root_entropy - w_entropy

candidates = [
    ("CGPA <= 7.15", students["CGPA"] <= 7.15),
    ("Internships == 0", students["Internships"] == 0),
    ("Backlogs == Yes", students["Backlogs"] == "Yes"),
    ("Aptitude <= 65", students["Aptitude"] <= 65),
]
candidate_results = [(name, root_entropy - weighted_eval(students.loc[mask, "Placed"], students.loc[~mask, "Placed"], entropy)) for name, mask in candidates]
candidate_results.sort(key=lambda x: x[1], reverse=True)

# Full dataset criterion evaluation
train = pd.read_csv(os.path.join(config.SPLITS_DIR, "train_ready.csv"))
val = pd.read_csv(os.path.join(config.SPLITS_DIR, "val_ready.csv"))
drop_cols = ["StudentID", "IsAnomaly", "Salary Package", config.TARGET_COLUMN]
feature_cols = [c for c in train.columns if c not in drop_cols]
x_train, y_train = train[feature_cols], train[config.TARGET_COLUMN]
x_val, y_val = val[feature_cols], val[config.TARGET_COLUMN]

criterion_results = {}
for crit in ["gini", "entropy"]:
    t = DecisionTreeClassifier(criterion=crit, max_depth=6, random_state=config.RANDOM_STATE)
    t.fit(x_train, y_train)
    criterion_results[crit] = accuracy_score(y_val, t.predict(x_val))

# Cost-complexity pruning
path = DecisionTreeClassifier(random_state=config.RANDOM_STATE).cost_complexity_pruning_path(x_train, y_train)
alphas = path.ccp_alphas[:-1:max(1, len(path.ccp_alphas) // 20)]
train_scores, val_scores = [], []
for alpha in alphas:
    m = DecisionTreeClassifier(ccp_alpha=alpha, random_state=config.RANDOM_STATE)
    m.fit(x_train, y_train)
    train_scores.append(accuracy_score(y_train, m.predict(x_train)))
    val_scores.append(accuracy_score(y_val, m.predict(x_val)))

best_idx = val_scores.index(max(val_scores))
best_alpha = alphas[best_idx]

plt.figure(figsize=(8, 5))
plt.plot(alphas, train_scores, marker="o", label="Train Accuracy", color="#4f46e5", markersize=4)
plt.plot(alphas, val_scores, marker="s", label="Validation Accuracy", color="#10b981", markersize=4)
plt.axvline(best_alpha, color="#f43f5e", linestyle="--", label=f"Optimal alpha ({best_alpha:.5f})")
plt.xlabel("Cost-Complexity Alpha (ccp_alpha)")
plt.ylabel("Accuracy")
plt.title("Minimal Cost-Complexity Pruning Trajectory", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(config.SPLITTING_DIR, "pruning_curve.png"), dpi=120)
plt.close()

with open(os.path.join(config.SPLITTING_DIR, "splitting_criteria_report.txt"), "w") as f:
    f.write("PART 1: worked example - 14 student dataset\n")
    f.write(f"Root entropy: {round(root_entropy,4)}\nRoot Gini: {round(root_gini,4)}\nRoot MSE: {round(root_mse,4)}\n")
    f.write(f"\nCGPA <= 7.15 split:\n")
    f.write(f"  Weighted entropy: {round(w_entropy,4)}  Information Gain: {round(info_gain,4)}\n")
    f.write(f"  Weighted Gini: {round(w_gini,4)}  Weighted MSE: {round(w_mse,4)}\n")
    f.write("\nCandidate splits by Information Gain:\n")
    for name, gain in candidate_results:
        f.write(f"  {name}: {round(gain,4)}\n")
    f.write("\nPART 2: entropy vs Gini (real dataset, max_depth=6)\n")
    for crit, acc in criterion_results.items():
        f.write(f"  {crit}: {round(acc,4)}\n")
    f.write("\nPART 3: pruning\n")
    f.write(f"  Best ccp_alpha: {round(best_alpha,5)} (val accuracy = {round(val_scores[best_idx],4)})\n")

print(f"Session 14 Pruning & Splitting Criteria report saved. Best alpha: {round(best_alpha, 5)}")
