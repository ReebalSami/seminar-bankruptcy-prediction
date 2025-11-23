#!/usr/bin/env python3
"""
Audit Phase 05 (Modeling) numbers used in Chapter 8.
- Compute overall best ROC-AUC per horizon across base + extra results
- Validate 'Kurzfazit' AUCs and PR-AUC paragraph against seminar-paper/kapitel/08_Modellierung.tex
- Print a JSON-like report with OK/FAIL and detected values
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MOD_RESULTS = ROOT / "results" / "05_modeling"
EXTRA_RESULTS = MOD_RESULTS / "extra"
CH8 = ROOT / "seminar-paper" / "kapitel" / "08_Modellierung.tex"


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


def _german_fmt(x: float) -> str:
    s = f"{x:.3f}"
    return s.replace(".", "{,}")


def compute_best() -> Tuple[Dict[str, Tuple[str, float, float]], Dict[str, float]]:
    base = _read_dir(MOD_RESULTS)
    extra = _read_dir(EXTRA_RESULTS) if EXTRA_RESULTS.exists() else {}
    horizons = sorted(set(base.keys()) | set(extra.keys()))
    best_map: Dict[str, Tuple[str, float, float]] = {}
    for h in horizons:
        cand: Dict[str, Dict[str, float]] = {}
        cand.update(base.get(h, {}))
        cand.update(extra.get(h, {}))
        if not cand:
            continue
        mname, vals = max(cand.items(), key=lambda kv: kv[1].get("roc_auc_mean", float("-inf")))
        best_map[h] = (mname, float(vals.get("roc_auc_mean")), float(vals.get("pr_auc_mean")))
    # PR-AUC paragraph uses PR of the best ROC-AUC model per H
    pr_map = {h: t[2] for h, t in best_map.items()}
    return best_map, pr_map


def main() -> None:
    best_map, pr_map = compute_best()
    text = CH8.read_text(encoding="utf-8") if CH8.exists() else ""

    # Kurzfazit line: extract the AUCs listed for H1..H5 in order
    # Expected tokens like 0{,}796 etc.
    expected_auc_tokens = []
    order = ["H1", "H2", "H3", "H4", "H5"]
    for h in order:
        if h not in best_map:
            continue
        auc = best_map[h][1]
        expected_auc_tokens.append(_german_fmt(auc))

    kurz_ok = all(tok in text for tok in expected_auc_tokens)

    # PR-AUC paragraph checks for the same horizons
    expected_pr_tokens = []
    for h in order:
        if h not in pr_map:
            continue
        pr = pr_map[h]
        expected_pr_tokens.append(_german_fmt(pr))
    pr_ok = all(tok in text for tok in expected_pr_tokens)

    report = {
        "best_models": {h: {"model": m, "roc_auc": auc, "pr_auc": pr} for h, (m, auc, pr) in best_map.items()},
        "kurzfazit_tokens": expected_auc_tokens,
        "kurzfazit_present": kurz_ok,
        "pr_tokens": expected_pr_tokens,
        "pr_present": pr_ok,
        "file": str(CH8),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
