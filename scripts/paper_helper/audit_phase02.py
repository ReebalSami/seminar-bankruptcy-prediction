import json
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
R02 = BASE / "results/02_exploratory_analysis"
FACTS = R02 / "phase02_facts.json"

CORR_FILES = {
    1: R02 / "02c_H1_correlation.xlsx",
    2: R02 / "02c_H2_correlation.xlsx",
    3: R02 / "02c_H3_correlation.xlsx",
    4: R02 / "02c_H4_correlation.xlsx",
    5: R02 / "02c_H5_correlation.xlsx",
}


def count_high_corr_from_matrix(df: pd.DataFrame, thr: float = 0.8) -> int:
    # Try to coerce numeric and drop non-numeric cols
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    # If an index column exists (feature names), set it as index
    if numeric_df.isna().all(axis=1).sum() > 0 and not numeric_df.columns.equals(df.columns):
        pass
    # Ensure square
    if numeric_df.shape[0] != numeric_df.shape[1]:
        # Some exports include a first col with feature names
        # Try to set first column as index and drop it
        maybe_index = df.columns[0]
        try:
            m = df.set_index(maybe_index)
            m = m.apply(pd.to_numeric, errors='coerce')
            numeric_df = m
        except Exception:
            pass
    # Count upper triangle excluding diagonal
    n = numeric_df.shape[0]
    if n == 0:
        return 0
    vals = []
    for i in range(n):
        for j in range(i+1, n):
            v = numeric_df.iat[i, j]
            if pd.notna(v) and abs(v) > thr:
                vals.append(v)
    return len(vals)


def audit_corr_counts():
    print("=== AUDIT PHASE 02: Correlation counts ===")
    counts = {}
    for h, path in CORR_FILES.items():
        try:
            xls = pd.ExcelFile(path)
            # heuristic: use the last sheet if multiple; or the largest square-like sheet
            best_df = None
            best_size = -1
            for name in xls.sheet_names:
                df = xls.parse(name)
                # try square-ish
                if abs(df.shape[0] - df.shape[1]) < 5 and df.shape[0]*df.shape[1] > best_size:
                    best_df = df
                    best_size = df.shape[0]*df.shape[1]
            if best_df is None:
                best_df = xls.parse(xls.sheet_names[0])
            cnt = count_high_corr_from_matrix(best_df, thr=0.8)
            counts[h] = cnt
        except Exception as e:
            print(f"H{h}: error reading {path}: {e}")
            counts[h] = None
    print("Counts |r|>0.8 per horizon:")
    for h in sorted(counts):
        print(f"  H{h}: {counts[h]}")
    return counts


def main():
    facts = json.loads(FACTS.read_text())
    print("=== AUDIT PHASE 02: FACTS ===")
    print(json.dumps(facts["samples"], indent=2))
    print("\nSkewness H1:", facts.get("skewness_H1"))
    print("\nUnivariate (sig after FDR):", facts["univariate"]["sig_fdr_q05"])\
    
    counts = audit_corr_counts()
    print("\nphase02_facts.json correlation overview (avg high correlations):", facts.get("correlation",{}).get("avg_high_correlations"))


if __name__ == "__main__":
    main()
