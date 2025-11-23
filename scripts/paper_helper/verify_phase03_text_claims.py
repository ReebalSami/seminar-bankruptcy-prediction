import json
from pathlib import Path
import pandas as pd

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
R03 = BASE / "results/03_multicollinearity"
R02 = BASE / "results/02_exploratory_analysis"
FEAT_JSON = BASE / "data/polish-companies-bankruptcy/feature_descriptions.json"

with FEAT_JSON.open() as f:
    FEAT = json.load(f)["features"]


def feat_name(code):
    info = FEAT.get(code, {})
    return info.get("short_name") or info.get("name") or code


def feat_cat(code):
    info = FEAT.get(code, {})
    return info.get("category", "?")


def load_removed_h1():
    xls = pd.ExcelFile(R03 / "03a_H1_vif.xlsx")
    df = xls.parse("Removed_Features")
    # Ensure expected columns
    # Columns expected: Feature, VIF_at_Removal, Iteration_Removed, Reason
    return df


def top10_with_meta(df):
    top = df.sort_values("VIF_at_Removal", ascending=False, na_position='last').head(10).copy()
    top["ShortName"] = top["Feature"].map(feat_name)
    top["Category"] = top["Feature"].map(feat_cat)
    return top


def get_corr_h1(pairs):
    # tries common sheet names
    xls_path = R02 / "02c_H1_correlation.xlsx"
    xls = pd.ExcelFile(xls_path)
    # Use first sheet or a named sheet if present
    sheet = xls.sheet_names[0]
    corr = xls.parse(sheet)
    # If file is an index-format matrix, standardize
    # Expect a square matrix with header row/col = feature codes
    corr = corr.set_index(corr.columns[0]) if corr.columns[0] != corr.columns[1] else corr
    res = {}
    for a, b in pairs:
        try:
            val = float(corr.loc[a, b])
        except Exception:
            # try reversed
            try:
                val = float(corr.loc[b, a])
            except Exception:
                val = None
        res[(a, b)] = val
    return res


def main():
    df = load_removed_h1()
    top = top10_with_meta(df)
    print("Top 10 removed features in H1 with names and categories:")
    print(top[["Feature", "ShortName", "Category", "VIF_at_Removal", "Iteration_Removed"]])

    counts = top["Category"].value_counts().to_dict()
    print("\nCategory distribution (Top 10):", counts)

    # Check specific pairs correlations and iterations
    pairs = [("A7", "A14"), ("A14", "A18"), ("A7", "A18"), ("A32", "A52"), ("A16", "A26")]
    corr_vals = get_corr_h1(pairs)
    print("\nSelected correlations (H1):")
    for k, v in corr_vals.items():
        print(f"{k[0]}-{k[1]}: {v}")

    # Iterations for A32, A26 if present
    for code in ["A32", "A26", "A49"]:
        row = df.loc[df["Feature"] == code]
        if not row.empty:
            it = int(row["Iteration_Removed"].iloc[0])
            vif = float(row["VIF_at_Removal"].iloc[0])
            print(f"Removal {code}: Iteration={it}, VIF_at_Removal={vif}")
        else:
            print(f"Removal {code}: not found in H1 removed list")


if __name__ == "__main__":
    main()
