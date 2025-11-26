#!/usr/bin/env python3
"""
Audit differences between embedded selections (base vs nested) before consensus.
- Reads base embedded: results/04_feature_selection/04c_H{h}_embedded_selected.json
- Reads nested embedded: results/04_feature_selection/nested/04c_H{h}_embedded_nested.json
- For nested, derives feature set as:
  * if 'selected_features' exists: use it; else
  * if 'fold_selections' exists: majority over folds (> 50% of folds); else empty
- Reports per horizon Jaccard for lasso and random_forest, and set sizes.
Outputs:
- results/delta/embedded_base_vs_nested.xlsx
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "04_feature_selection"
DELTA = ROOT / "results" / "delta"
DELTA.mkdir(parents=True, exist_ok=True)


def _load_base(h: int) -> Dict[str, List[str]]:
    p = RES / f"04c_H{h}_embedded_selected.json"
    if not p.exists():
        return {"lasso": [], "random_forest": []}
    data = json.loads(p.read_text())
    if isinstance(data, dict) and "methods" in data:
        data = data["methods"]
    l1 = list((data.get("lasso") or {}).get("selected_features", []) or [])
    rf = list((data.get("random_forest") or {}).get("selected_features", []) or [])
    return {"lasso": l1, "random_forest": rf}


essential_keys = ("lasso", "random_forest")


def _derive_nested_feats(node) -> List[str]:
    if not node or not isinstance(node, dict):
        return []
    sf = node.get("selected_features")
    if isinstance(sf, list):
        return list(sf)
    folds = node.get("fold_selections")
    if isinstance(folds, list) and folds and all(isinstance(f, list) for f in folds):
        c = Counter()
        for f in folds:
            c.update(f)
        thresh = len(folds) / 2
        return [feat for feat, cnt in c.items() if cnt > thresh]
    # fallback: final_features_majority if present
    ffm = node.get("final_features_majority")
    if isinstance(ffm, list):
        return list(ffm)
    return []


def _load_nested(h: int) -> Dict[str, List[str]]:
    p = RES / "nested" / f"04c_H{h}_embedded_nested.json"
    if not p.exists():
        return {"lasso": [], "random_forest": []}
    data = json.loads(p.read_text())
    methods = (data.get("methods") or {}) if isinstance(data, dict) else {}
    out: Dict[str, List[str]] = {}
    for k in essential_keys:
        out[k] = _derive_nested_feats(methods.get(k) or {})
    return out


def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    U = len(A | B)
    return len(A & B) / U if U else 0.0


def main() -> None:
    # infer horizons by presence of base files using regex
    pat = re.compile(r"04c_H(\d+)_embedded_selected\.json$")
    hset = set()
    for p in RES.glob("04c_H*_embedded_selected.json"):
        m = pat.match(p.name)
        if m:
            hset.add(int(m.group(1)))
    horizons = sorted(hset)
    rows = []
    for h in horizons:
        base = _load_base(h)
        nest = _load_nested(h)
        for m in essential_keys:
            rows.append({
                "Horizon": f"H{h}",
                "Method": m,
                "Base_Count": len(base[m]),
                "Nested_Count": len(nest[m]),
                "Jaccard": jaccard(base[m], nest[m]),
                "Base_only": ", ".join(sorted(set(base[m]) - set(nest[m]))) if set(base[m]) - set(nest[m]) else "",
                "Nested_only": ", ".join(sorted(set(nest[m]) - set(base[m]))) if set(nest[m]) - set(base[m]) else "",
            })
    df = pd.DataFrame(rows)
    df.to_excel(DELTA / "embedded_base_vs_nested.xlsx", index=False)
    print(df)


if __name__ == "__main__":
    main()
