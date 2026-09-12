# Session 02: Heuristic Benchmark & Baseline Inference
# Evaluates majority-class zero-rule prediction and domain-informed rules.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import config

os.makedirs(config.BASELINE_DIR, exist_ok=True)

df = pd.read_csv(config.RAW_DATA_PATH)

majority_class = df[config.TARGET_COLUMN].mode()[0]
baseline_accuracy = (df[config.TARGET_COLUMN] == majority_class).mean()

print(f"Majority class: {majority_class}")
print(f"Baseline accuracy: {round(baseline_accuracy, 4)}")

with open(os.path.join(config.BASELINE_DIR, "baseline_report.txt"), "w") as f:
    f.write(f"Majority class: {majority_class}\nBaseline accuracy: {round(baseline_accuracy, 4)}\n")
