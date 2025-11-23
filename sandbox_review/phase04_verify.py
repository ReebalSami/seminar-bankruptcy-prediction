"""
Phase 04 Verification Script (Read-Only)
========================================

Purpose:
- Verify that Phase 04a selected features exist in the dataset and are subsets of VIF-cleaned features.
- Validate selection counts against configured bounds.
- Summarize any discrepancies for each horizon.

This script performs read-only checks and writes a small report to sandbox_review/phase04_verify_report.json.
It does not modify project code or outputs.

Run:
  python sandbox_review/phase04_verify.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/project_config.yaml"
DATA_PATH = ROOT / "data/processed/poland_imputed.parquet"
VIF_DIR = ROOT / "data/processed/feature_sets"
SEL_DIR = ROOT / "results/04_feature_selection"
REPORT_PATH = ROOT / "sandbox_review/phase04_verify_report.json"


def load_config() -> Dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_dataset_columns() -> List[str]:
    df = pd.read_parquet(DATA_PATH)
    return list(df.columns)


def load_vif_features(h: int) -> List[str]:
    p = VIF_DIR / f"H{h}_features.json"
    with open(p, "r") as f:
        return json.load(f)


def load_filter_selection(h: int) -> Dict[str, List[str]]:
    p = SEL_DIR / f"04a_H{h}_filter_selected.json"
    with open(p, "r") as f:
        return json.load(f)


def main() -> None:
    cfg = load_config()
    horizons: List[int] = cfg["datasets"]["polish"]["horizons"]
    kmin = cfg["feature_selection"]["target_features_min"]
    kmax = cfg["feature_selection"]["target_features_max"]

    cols = load_dataset_columns()

    report = {"checks": []}

    for h in horizons:
        vif_feats = set(load_vif_features(h))
        sel = load_filter_selection(h)

        horizon_result = {
            "horizon": h,
            "vif_feature_count": len(vif_feats),
            "methods": {},
            "issues": []
        }

        for method_key in ("spearman_selected", "mi_selected", "anova_selected"):
            feats = sel.get(method_key, [])
            feats_set = set(feats)

            # Count checks
            count_ok = (kmin <= len(feats) <= kmax) or len(feats) == 0

            # Membership checks
            in_dataset = [f for f in feats if f in cols]
            not_in_dataset = sorted(list(set(feats) - set(in_dataset)))
            in_vif = [f for f in feats if f in vif_feats]
            not_in_vif = sorted(list(set(feats) - set(in_vif)))

            horizon_result["methods"][method_key] = {
                "count": len(feats),
                "count_ok": count_ok,
                "not_in_dataset": not_in_dataset,
                "not_in_vif": not_in_vif,
            }

            if not count_ok and len(feats) > 0:
                horizon_result["issues"].append(
                    f"{method_key}: count {len(feats)} outside [{kmin},{kmax}]"
                )
            if not_in_dataset:
                horizon_result["issues"].append(
                    f"{method_key}: {len(not_in_dataset)} features not in dataset columns"
                )
            if not_in_vif:
                horizon_result["issues"].append(
                    f"{method_key}: {len(not_in_vif)} features not in VIF feature set"
                )

        report["checks"].append(horizon_result)

    # Aggregate summary
    total_issues = sum(len(h["issues"]) for h in report["checks"])
    report["summary"] = {"total_horizons": len(horizons), "total_issues": total_issues}

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["summary"], indent=2))
    if total_issues:
        print("Issues detected. See:", REPORT_PATH)
    else:
        print("No issues detected.")


if __name__ == "__main__":
    main()
