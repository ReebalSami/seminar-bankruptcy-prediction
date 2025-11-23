import os
import pandas as pd
from pathlib import Path

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
R01 = BASE / "results/01_data_preparation"
DUP_XLSX = R01 / "01a_duplicate_removal.xlsx"
OUT_XLSX = R01 / "01b_outlier_treatment.xlsx"
IMP_XLSX = R01 / "01c_imputation_report.xlsx"


def read_sheets(path):
    try:
        xl = pd.ExcelFile(path)
        return {name: xl.parse(name) for name in xl.sheet_names}
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return {}


def audit_phase01():
    print("=== AUDIT PHASE 01: DATA PREPARATION ===\n")

    # 01a: Duplicates
    dup = read_sheets(DUP_XLSX)
    if 'Summary' in dup:
        print("-- 01a Duplicate Removal (Summary) --")
        df = dup['Summary']
        metrics = dict(zip(df['Metric'], df['Value']))
        print(metrics)
        # Extract core facts
        original = metrics.get('Original Observations')
        after = metrics.get('After Removal')
        removed = metrics.get('Duplicates Removed')
        print(f"Original: {original}, After: {after}, Removed: {removed}")
        print(f"Rates: Original={metrics.get('Original Bankruptcy Rate (%)')}, After={metrics.get('After Removal Rate (%)')}, Change={metrics.get('Rate Change (pp)')}")

    # 01b: Outliers Winsorization
    out = read_sheets(OUT_XLSX)
    if 'Summary' in out and 'Winsorization_Stats' in out:
        print("\n-- 01b Outlier Treatment (Summary) --")
        s = out['Summary']
        s_metrics = dict(zip(s['Metric'], s['Value']))
        print({k: s_metrics.get(k) for k in [
            'Total Observations','Features Winsorized','Lower Percentile','Upper Percentile',
            'Total Values Capped','Avg % Affected per Feature','Max % Affected (Single Feature)','Min % Affected (Single Feature)'
        ]})
        # Sanity from stats table
        stats_df = out['Winsorization_Stats']
        total_affected = int(stats_df['N_Affected'].sum())
        avg_pct = float(stats_df['Pct_Affected'].mean())
        min_pct = float(stats_df['Pct_Affected'].min())
        max_pct = float(stats_df['Pct_Affected'].max())
        print(f"Computed totals: Values capped={total_affected:,}, Avg % affected={avg_pct:.2f}%, Min %={min_pct:.2f}%, Max %={max_pct:.2f}%")

    # 01c: Imputation
    imp = read_sheets(IMP_XLSX)
    if 'Summary' in imp and 'Imputation_Stats' in imp and 'Quality_Assessment' in imp:
        print("\n-- 01c Imputation (Summary) --")
        s = imp['Summary']
        s_metrics = dict(zip(s['Metric'], s['Value']))
        print({k: s_metrics.get(k) for k in [
            'Total Observations','Features Imputed','Imputation Method','Estimator',
            'Total Missing Values Before','Total Missing Values After','Avg Quality Score'
        ]})
        # A37 quality (if present)
        qa = imp['Quality_Assessment']
        if 'Feature' in qa.columns:
            a37 = qa[qa['Feature'] == 'A37']
            if not a37.empty:
                row = a37.iloc[0]
                print(f"A37 Quality: Score={row['Quality_Score']:.1f}, Rating={row['Rating']}, Missing_Pct={row['Missing_Pct']:.2f}")


if __name__ == "__main__":
    audit_phase01()
