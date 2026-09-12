"""
Session 01: Cohort Ingestion & Schema Audit
Pipeline: Ingests raw cohort records, validates schemas, and audits missing values.
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

def run_cohort_ingestion():
    os.makedirs(config.INGESTION_DIR, exist_ok=True)
    df = pd.read_csv(config.RAW_DATA_PATH)
    
    report_path = os.path.join(config.INGESTION_DIR, "ingestion_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("CAMPUSPLACEMENTIQ — COHORT INGESTION & AUDIT REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source Dataset: {config.RAW_DATA_PATH}\n")
        f.write(f"Total Student Records: {df.shape[0]:,}\n")
        f.write(f"Feature Dimensions:    {df.shape[1]} columns\n")
        f.write(f"Memory Footprint:      {df.memory_usage(deep=True).sum() / 1e6:.2f} MB\n\n")
        
        f.write("--- Missing Value Summary ---\n")
        missing = df.isnull().sum()
        missing_nonzero = missing[missing > 0]
        if len(missing_nonzero) == 0:
            f.write("Zero missing values detected across all raw columns.\n")
        else:
            for col, count in missing_nonzero.items():
                f.write(f"  - {col}: {count:,} missing ({count/len(df)*100:.2f}%)\n")
        
        f.write("\n--- Data Type Distribution ---\n")
        dtype_counts = df.dtypes.value_counts()
        for dt, cnt in dtype_counts.items():
            f.write(f"  - {dt}: {cnt} columns\n")
            
        f.write("\n--- Target Variable Distribution (PlacementStatus) ---\n")
        if config.TARGET_COLUMN in df.columns:
            vc = df[config.TARGET_COLUMN].value_counts()
            for val, cnt in vc.items():
                f.write(f"  - {val}: {cnt:,} ({cnt/len(df)*100:.2f}%)\n")
        f.write("\nAudit completed successfully. Pipeline ready for heuristic baseline.\n")

if __name__ == "__main__":
    run_cohort_ingestion()