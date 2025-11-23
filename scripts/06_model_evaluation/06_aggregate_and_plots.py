#!/usr/bin/env python3
"""
Phase 06: Model Evaluation Aggregation
=====================================

Aggregates per-horizon modeling metrics from results/05_modeling and produces:
- results/06_model_evaluation/05_ALL_model_eval.xlsx
- results/06_model_evaluation/05_ALL_model_eval.html

This script is read-only on modeling outputs, safe to run anytime after Phase 05.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOD_RESULTS = PROJECT_ROOT / "results" / "05_modeling"
MOD_RESULTS_EXTRA = MOD_RESULTS / "extra"
EVAL_RESULTS = PROJECT_ROOT / "results" / "06_model_evaluation"
EVAL_RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    # Collect per-horizon metrics (base + extra)
    rows = []
    per_horizon_frames = {}

    def ingest_dir(d: Path):
        for p in sorted(d.glob("H*_metrics.json")):
            blob = json.loads(p.read_text())
            h = blob.get("horizon")
            metrics = blob.get("metrics", {})
            # build per-horizon table
            tbl_rows = []
            for mname, s in metrics.items():
                rows.append({
                    "Horizon": f"H{h}",
                    "Model": mname,
                    "ROC_AUC_Mean": s.get("roc_auc_mean"),
                    "ROC_AUC_Std": s.get("roc_auc_std"),
                    "PR_AUC_Mean": s.get("pr_auc_mean"),
                    "PR_AUC_Std": s.get("pr_auc_std"),
                })
                tbl_rows.append({
                    "Model": mname,
                    "ROC_AUC_Mean": s.get("roc_auc_mean"),
                    "ROC_AUC_Std": s.get("roc_auc_std"),
                    "PR_AUC_Mean": s.get("pr_auc_mean"),
                    "PR_AUC_Std": s.get("pr_auc_std"),
                })
            if tbl_rows:
                key = f"H{h}"
                if key in per_horizon_frames:
                    per_horizon_frames[key] = pd.concat([per_horizon_frames[key], pd.DataFrame(tbl_rows)], ignore_index=True)
                else:
                    per_horizon_frames[key] = pd.DataFrame(tbl_rows)

    ingest_dir(MOD_RESULTS)
    if MOD_RESULTS_EXTRA.exists():
        ingest_dir(MOD_RESULTS_EXTRA)

    df_all = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Horizon", "Model", "ROC_AUC_Mean", "ROC_AUC_Std", "PR_AUC_Mean", "PR_AUC_Std"]
    )

    # Excel
    xlsx_path = EVAL_RESULTS / "05_ALL_model_eval.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="All", index=False)
        # best per horizon
        if not df_all.empty:
            best_rows = []
            for h, g in df_all.groupby("Horizon"):
                best = g.sort_values("ROC_AUC_Mean", ascending=False).iloc[0]
                best_rows.append(best)
            pd.DataFrame(best_rows).to_excel(writer, sheet_name="Best_Per_Horizon", index=False)
        # per-horizon sheets
        for h, frame in per_horizon_frames.items():
            frame.to_excel(writer, sheet_name=f"{h}_Models", index=False)

    # HTML quick view
    html_path = EVAL_RESULTS / "05_ALL_model_eval.html"
    html = [
        "<html><head><meta charset='utf-8'><title>Phase 05: Model Evaluation</title>",
        "<style>body{font-family:Segoe UI,Arial;margin:20px;}table{border-collapse:collapse;}th,td{padding:8px 12px;border:1px solid #ddd;}th{background:#2a9d8f;color:#fff;}</style>",
        "</head><body>",
        "<h1>Phase 05: Model Evaluation (Aggregated)</h1>",
    ]
    html.append("<h2>All Models (Base + Extra)</h2>")
    html.append(df_all.to_html(index=False))
    html.append("</body></html>")
    html_path.write_text("\n".join(html), encoding="utf-8")

    print(f"\n✓ Evaluation written: {xlsx_path}\n✓ HTML: {html_path}")


if __name__ == "__main__":
    main()
