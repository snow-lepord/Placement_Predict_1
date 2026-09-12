# Session 05: Stratified Data Partitioning (Train/Val/Test)
# 70% train, 10% validation, 20% test with stratified target distribution.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from sklearn.model_selection import train_test_split
import config

os.makedirs(config.SPLIT_DIR, exist_ok=True)
os.makedirs(config.SPLITS_DIR, exist_ok=True)

df = pd.read_csv(config.CLEAN_DATA_PATH)

train_val, test = train_test_split(
    df, test_size=0.2, stratify=df[config.TARGET_COLUMN], random_state=config.RANDOM_STATE
)
train, val = train_test_split(
    train_val, test_size=0.125, stratify=train_val[config.TARGET_COLUMN], random_state=config.RANDOM_STATE
)

train.to_csv(os.path.join(config.SPLITS_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(config.SPLITS_DIR, "val.csv"), index=False)
test.to_csv(os.path.join(config.SPLITS_DIR, "test.csv"), index=False)

print(f"Stratified Partitioning Complete -> Train: {train.shape[0]} | Val: {val.shape[0]} | Test: {test.shape[0]}")

with open(os.path.join(config.SPLIT_DIR, "split_report.txt"), "w") as f:
    f.write(f"Train: {train.shape[0]}\nValidation: {val.shape[0]}\nTest: {test.shape[0]}\n")
