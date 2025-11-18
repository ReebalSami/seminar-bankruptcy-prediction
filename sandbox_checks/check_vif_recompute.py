import json
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < 2:
        return pd.DataFrame({"Feature": df.columns, "VIF": [np.nan] * df.shape[1]})
    X = add_constant(df, has_constant='add')
    records = []
    for i in range(1, X.shape[1]):  # skip constant
        vif = variance_inflation_factor(X.values, i)
        records.append({"Feature": df.columns[i-1], "VIF": float(vif)})
    return pd.DataFrame(records).sort_values("VIF", ascending=False).reset_index(drop=True)


def main():
    # Load imputed dataset
    data_path = PROJECT_ROOT / "data" / "processed" / "poland_imputed.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")
    df = pd.read_parquet(data_path)

    all_reports = {}
    for horizon in range(1, 6):
        features_path = PROJECT_ROOT / "data" / "processed" / "feature_sets" / f"H{horizon}_features.json"
        if not features_path.exists():
            raise FileNotFoundError(f"Missing features JSON: {features_path}")
        retained_features = json.loads(features_path.read_text())

        df_h = df[df["horizon"] == horizon].copy()
        X_h = df_h[retained_features].copy()

        # Defensive clean (mirror script behavior)
        X_h = X_h.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
        std_vals = X_h.std(numeric_only=True)
        X_h = X_h.loc[:, (std_vals >= 1e-12).values]

        vif_df = compute_vif(X_h)

        excel_path = PROJECT_ROOT / "results" / "03_multicollinearity" / f"03a_H{horizon}_vif.xlsx"
        if not excel_path.exists():
            raise FileNotFoundError(f"Missing Phase 03 Excel: {excel_path}")
        final_vif_saved = pd.read_excel(excel_path, sheet_name="Final_VIF")

        merged = pd.merge(vif_df, final_vif_saved, on="Feature", suffixes=("_recomputed", "_saved"))
        merged["abs_diff"] = (merged["VIF_recomputed"] - merged["VIF_saved"]).abs()

        report = {
            "rows": int(len(df_h)),
            "features_checked": int(len(X_h.columns)),
            "max_vif": float(vif_df["VIF"].max()) if not vif_df.empty else np.nan,
            "all_vif_leq_10": bool((vif_df["VIF"] <= 10 + 1e-6).all()),
            "rows_in_comparison": int(len(merged)),
            "max_abs_diff_vs_saved": float(merged["abs_diff"].max()) if not merged.empty else np.nan,
        }
        all_reports[f"H{horizon}"] = report

    out_path = PROJECT_ROOT / "sandbox_checks" / "vif_recompute_ALL_report.json"
    out_path.write_text(json.dumps(all_reports, indent=2))
    print(json.dumps(all_reports, indent=2))


if __name__ == "__main__":
    main()
