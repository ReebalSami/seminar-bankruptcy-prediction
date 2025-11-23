import json
from pathlib import Path
import pandas as pd

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
R03 = BASE / "results/03_multicollinearity"
FSET = BASE / "data/processed/feature_sets"

HORIZONS = [1, 2, 3, 4, 5]


def read_meta(h):
    xls = pd.ExcelFile(R03 / f"03a_H{h}_vif.xlsx")
    meta = xls.parse("Metadata").iloc[0].to_dict()
    return {
        "H": h,
        "Initial": int(meta.get("Initial_Features")),
        "Final": int(meta.get("Final_Features")),
        "Removed": int(meta.get("Removed_Count")),
        "Iterations": int(meta.get("Iterations")),
        "Max_Final_VIF": float(meta.get("Max_Final_VIF")),
    }


def read_removed_top(h, n=10):
    xls = pd.ExcelFile(R03 / f"03a_H{h}_vif.xlsx")
    df = xls.parse("Removed_Features")
    # sort by VIF_at_Removal descending, using na_position='last'
    df = df.sort_values("VIF_at_Removal", ascending=False, na_position='last')
    return df.head(n)


def check_feature_json(h, expected_final):
    lst = json.loads((FSET / f"H{h}_features.json").read_text())
    return len(lst), (len(lst) == expected_final)


def main():
    print("=== AUDIT PHASE 03: VIF Analysis ===")
    rows = []
    for h in HORIZONS:
        m = read_meta(h)
        rows.append(m)
    dfm = pd.DataFrame(rows)
    print("\nPer-horizon metadata:")
    print(dfm)

    # Cross-check JSON feature list sizes
    print("\nJSON feature list size checks:")
    for h in HORIZONS:
        expected = dfm.loc[dfm.H == h, "Final"].iloc[0]
        size, ok = check_feature_json(h, expected)
        print(f"H{h}: json={size}, expected_final={expected} -> {'OK' if ok else 'MISMATCH'}")

    # Inspect extreme removals in H1
    print("\nTop removed features in H1 (by VIF at removal):")
    top = read_removed_top(1, n=10)
    cols = ["Feature", "VIF_at_Removal", "Iteration_Removed", "Reason"]
    print(top[cols])

    # Load consolidated summary to compare
    all_xls = pd.ExcelFile(R03 / "03a_ALL_vif.xlsx")
    summary = all_xls.parse("Summary")
    print("\nConsolidated summary:")
    print(summary)


if __name__ == "__main__":
    main()
