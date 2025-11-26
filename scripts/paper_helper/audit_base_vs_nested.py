#!/usr/bin/env python3
"""
Audit base vs nested outputs.
- Compare final feature sets per horizon: counts, Jaccard, added/removed features
- Compare per-horizon best ROC-AUC winners (base vs nested) using base+extra directories
Outputs:
- results/delta/feature_set_diff.xlsx
- results/delta/metrics_winner_diff.xlsx
- prints concise summary to stdout
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data" / "processed" / "feature_sets_selected"
DELTA = ROOT / "results" / "delta"
DELTA.mkdir(parents=True, exist_ok=True)
BASE_MOD = ROOT / "results" / "05_modeling"
BASE_EXTRA = BASE_MOD / "extra"
NEST_MOD = ROOT / "results" / "05_modeling_nested"
NEST_EXTRA = NEST_MOD / "extra"


def read_feats(h: int, nested: bool) -> List[str]:
    suf = "_nested" if nested else ""
    fp = FEAT / f"H{h}_features_final{suf}.json"
    if not fp.exists():
        return []
    data = json.loads(fp.read_text())
    return list(data.get("features", []))


def feature_set_audit(horizons: List[int]) -> pd.DataFrame:
    rows: List[Dict] = []
    for h in horizons:
        b = set(read_feats(h, nested=False))
        n = set(read_feats(h, nested=True))
        inter = b & n
        union = b | n
        jacc = (len(inter) / len(union)) if union else 1.0
        rows.append({
            "Horizon": f"H{h}",
            "Count_Base": len(b),
            "Count_Nested": len(n),
            "Jaccard": jacc,
            "Removed": ", ".join(sorted(b - n)) if b - n else "",
            "Added": ", ".join(sorted(n - b)) if n - b else "",
        })
    return pd.DataFrame(rows)


def read_metrics_dir(d: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for p in d.glob("H*_metrics.json"):
        blob = json.loads(p.read_text())
        h = int(blob["horizon"])
        for mname, s in blob.get("metrics", {}).items():
            out.setdefault(h, {})[mname] = float(s.get("roc_auc_mean"))
    return out


def winner_table(horizons: List[int]) -> pd.DataFrame:
    base = read_metrics_dir(BASE_MOD)
    base_extra = read_metrics_dir(BASE_EXTRA) if BASE_EXTRA.exists() else {}
    nest = read_metrics_dir(NEST_MOD)
    nest_extra = read_metrics_dir(NEST_EXTRA) if NEST_EXTRA.exists() else {}
    rows: List[Dict] = []
    for h in horizons:
        cand_base = {**base.get(h, {}), **base_extra.get(h, {})}
        cand_nest = {**nest.get(h, {}), **nest_extra.get(h, {})}
        if cand_base:
            wb = max(cand_base.items(), key=lambda kv: kv[1])
        else:
            wb = ("--", float("nan"))
        if cand_nest:
            wn = max(cand_nest.items(), key=lambda kv: kv[1])
        else:
            wn = ("--", float("nan"))
        rows.append({
            "Horizon": f"H{h}",
            "Winner_Base": wb[0],
            "AUC_Base": wb[1],
            "Winner_Nested": wn[0],
            "AUC_Nested": wn[1],
            "Delta_AUC": wn[1] - wb[1] if pd.notnull(wb[1]) and pd.notnull(wn[1]) else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    # Determine horizons by presence of feature files
    horizons = sorted({int(p.name.split("_")[0][1:]) for p in FEAT.glob("H*_features_final.json")})
    feat_df = feature_set_audit(horizons)
    win_df = winner_table(horizons)
    feat_df.to_excel(DELTA / "feature_set_diff.xlsx", index=False)
    win_df.to_excel(DELTA / "metrics_winner_diff.xlsx", index=False)
    print("Feature set diffs:\n", feat_df)
    print("\nWinner diffs:\n", win_df)


if __name__ == "__main__":
    main()
