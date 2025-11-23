import json
from pathlib import Path
import pandas as pd

BASE = Path("/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction")
R04 = BASE / "results/04_feature_selection"
R03 = BASE / "results/03_multicollinearity"
FSEL = BASE / "data/processed/feature_sets_selected"

HORIZONS = [1,2,3,4,5]


def read_json(p: Path):
    with p.open() as f:
        return json.load(f)


def main():
    print("=== AUDIT PHASE 04: Feature Selection ===")

    # Read consensus workbook overview
    xls = pd.ExcelFile(R04 / "04d_ALL_consensus.xlsx")
    print("Sheets:", xls.sheet_names)
    # Try to find a summary-like sheet
    for s in xls.sheet_names:
        if s.lower().startswith("summary") or s.lower() == "summary":
            summary = xls.parse(s)
            print("\n[Consensus Summary] (first 10 rows):")
            print(summary.head(10))
            break
    else:
        summary = None

    # Per-horizon checks
    print("\nPer-horizon checks:")
    for h in HORIZONS:
        # Method selections
        f_filter = R04 / f"04a_H{h}_filter_selected.json"
        f_wrapper = R04 / f"04b_H{h}_wrapper_selected.json"
        f_emb = R04 / f"04c_H{h}_embedded_selected.json"

        filter_sel = read_json(f_filter) if f_filter.exists() else {}
        wrapper_sel = read_json(f_wrapper) if f_wrapper.exists() else {}
        emb_sel = read_json(f_emb) if f_emb.exists() else {}

        spearman_k = len(filter_sel.get("spearman_selected", []))
        mi_k = len(filter_sel.get("mi_selected", []))
        anova_k = len(filter_sel.get("anova_selected", []))
        wrapper_k = len(wrapper_sel.get("selected_features", []))
        lasso_k = len(emb_sel.get("lasso", {}).get("selected_features", []))
        rf_k = len(emb_sel.get("random_forest", {}).get("selected_features", []))

        # Final consensus JSON
        final_path = FSEL / f"H{h}_features_final.json"
        final_feats = read_json(final_path)
        final_k = len(final_feats.get("features", []))

        # Pull consensus metrics from workbook if present
        cons_rows = None
        mean_agreement = None
        retention = None
        if summary is not None and "Horizon" in summary.columns:
            cons_rows = summary[summary["Horizon"] == h]
            if not cons_rows.empty:
                mean_agreement = float(cons_rows["Mean_Agreement"].iloc[0]) if "Mean_Agreement" in cons_rows.columns else None
                retention = float(cons_rows["Retention_Ratio"].iloc[0]) if "Retention_Ratio" in cons_rows.columns else None

        print(f"H{h}: Spearman={spearman_k}, MI={mi_k}, ANOVA={anova_k}, RFECV={wrapper_k}, Lasso={lasso_k}, RF={rf_k}, Final={final_k}, MeanAgreement={mean_agreement}, Retention={retention}")

    # Check A37 presence in final sets
    print("\nPresence of A37 in final consensus sets:")
    present = {}
    for h in HORIZONS:
        feats = read_json(FSEL / f"H{h}_features_final.json")
        present[h] = ("A37" in feats)
        print(f"H{h}: {'YES' if present[h] else 'NO'}")

    # VIF vs final count comparison
    xls_vif = pd.ExcelFile(R03 / "03a_ALL_vif.xlsx")
    vif_sum = xls_vif.parse("Summary")
    print("\nVIF Final vs Consensus Final:")
    for h in HORIZONS:
        vif_final = int(vif_sum.loc[vif_sum["Horizon"] == h, "Final"].iloc[0])
        final_k = len(read_json(FSEL / f"H{h}_features_final.json"))
        print(f"H{h}: VIF Final={vif_final} -> Consensus Final={final_k}")

    # Nested stability if present
    print("\nNested stability (if available):")
    for h in HORIZONS:
        nested = R04 / "nested" / f"04c_H{h}_embedded_nested.json"
        if nested.exists():
            nd = read_json(nested).get("methods", {})
            stab = {k: nd.get(k, {}).get("stability_nogueira") for k in ["lasso","elastic_net","ridge"]}
            print(f"H{h}: {stab}")
        else:
            print(f"H{h}: nested not found")


if __name__ == "__main__":
    main()
