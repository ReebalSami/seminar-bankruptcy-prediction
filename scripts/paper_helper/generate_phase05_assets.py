#!/usr/bin/env python3
"""
Generate Phase 05 figure: Best ROC-AUC per horizon (overall best model)
- Reads results/05_modeling/H*_metrics.json and results/05_modeling/extra/H*_metrics.json
- Selects overall best model by ROC-AUC per horizon
- Saves bar chart to results/05_modeling/model_best_auc.png
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
MOD_RESULTS = ROOT / "results" / "05_modeling"
EXTRA_RESULTS = MOD_RESULTS / "extra"
OUT_FIG = MOD_RESULTS / "model_best_auc.png"


def _read_dir(dirpath: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    by_h: Dict[str, Dict[str, Dict[str, float]]] = {}
    for p in sorted(dirpath.glob("H*_metrics.json")):
        try:
            blob = json.loads(p.read_text())
        except Exception:
            continue
        h_raw = blob.get("horizon")
        if h_raw is None:
            continue
        h = f"H{int(h_raw)}"
        metrics = blob.get("metrics", {}) or {}
        if not metrics:
            continue
        by_h.setdefault(h, {})
        for name, vals in metrics.items():
            by_h[h][str(name)] = {
                "roc_auc_mean": float(vals.get("roc_auc_mean")) if vals.get("roc_auc_mean") is not None else float("nan"),
                "pr_auc_mean": float(vals.get("pr_auc_mean")) if vals.get("pr_auc_mean") is not None else float("nan"),
            }
    return by_h


def collect_best_overall() -> pd.DataFrame:
    base = _read_dir(MOD_RESULTS)
    extra = _read_dir(EXTRA_RESULTS) if EXTRA_RESULTS.exists() else {}

    horizons = sorted(set(base.keys()) | set(extra.keys()))
    rows: List[Dict] = []
    for h in horizons:
        cand: Dict[str, Dict[str, float]] = {}
        cand.update(base.get(h, {}))
        cand.update(extra.get(h, {}))
        if not cand:
            continue
        best_name, best_vals = max(cand.items(), key=lambda kv: kv[1].get("roc_auc_mean", float("-inf")))
        rows.append({
            "Horizon": h,
            "Best_Model": best_name,
            "ROC_AUC": best_vals.get("roc_auc_mean"),
            "PR_AUC": best_vals.get("pr_auc_mean"),
        })
    return pd.DataFrame(rows)


def plot_best_auc(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print("No modeling results to plot.")
        return
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.5, 3.8))
    order = sorted(df["Horizon"].tolist(), key=lambda x: int(x[1:]))
    ax = sns.barplot(data=df, x="Horizon", y="ROC_AUC", order=order, color="#4C78A8")
    # Annotate model name and value on each bar
    for idx, row in df.set_index("Horizon").loc[order].reset_index().iterrows():
        x = idx
        y = row["ROC_AUC"]
        label = f"{row['Best_Model']}\n{y:.3f}"
        ax.text(x, y + 0.01, label, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.0, max(1.0, df["ROC_AUC"].max() + 0.06))
    ax.set_ylabel("ROC-AUC (Bestes Modell)")
    ax.set_xlabel("Horizont")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Generated figure: {out_path}")


def main() -> None:
    df = collect_best_overall()
    plot_best_auc(df, OUT_FIG)


if __name__ == "__main__":
    main()
