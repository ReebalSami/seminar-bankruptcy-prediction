import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "04_feature_selection"
NESTED_DIR = RESULTS_DIR / "nested"
FINAL_FEATURES_DIR = ROOT / "data" / "processed" / "feature_sets_selected"
TABLES_DIR = ROOT / "seminar-paper" / "tables"

HORIZONS = ["H1", "H2", "H3", "H4", "H5"]
METHODS = ["lasso", "elastic_net", "ridge", "random_forest"]


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def find_retention_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if re.search(r"retention", str(c), re.IGNORECASE)]
    return candidates[0] if candidates else None


def load_consensus_from_xlsx() -> Optional[pd.DataFrame]:
    xlsx_path = RESULTS_DIR / "04d_ALL_consensus.xlsx"
    if not xlsx_path.exists():
        return None
    try:
        # Use explicit sheet that 04d writes the overview to
        return pd.read_excel(xlsx_path, sheet_name="Summary")
    except Exception:
        return None


def summarize_consensus() -> List[Tuple[str, str, Optional[int], Optional[float]]]:
    df = load_consensus_from_xlsx()
    rows: List[Tuple[str, str, Optional[int], Optional[float]]] = []
    if df is not None:
        # 04d summary sheet columns (by generator):
        # 'Horizon', 'Consensus_Features', 'Retention_%' among others
        for h in HORIZONS:
            sub = df[df.get("Horizon", df.iloc[:, 0]).astype(str).str.contains(h, case=False, na=False)]
            if len(sub) > 0:
                r = sub.iloc[0]
                n = None
                ret = None
                if "Consensus_Features" in r:
                    try:
                        n = int(r["Consensus_Features"]) if pd.notna(r["Consensus_Features"]) else None
                    except Exception:
                        n = None
                # retention can be numeric in [0,1] or percentage column 'Retention_%'
                if "Retention_%" in r and pd.notna(r["Retention_%"]):
                    ret = safe_float(r["Retention_%"])  # already percent
                elif "retention_ratio" in r and pd.notna(r["retention_ratio"]):
                    rr = safe_float(r["retention_ratio"])  # ratio 0-1
                    ret = rr * 100 if rr is not None else None
                rows.append((h, "intersection", n, ret))
            else:
                rows.append((h, "intersection", None, None))
    else:
        for h in HORIZONS:
            fpath = FINAL_FEATURES_DIR / f"{h}_features_final.json"
            if fpath.exists():
                try:
                    blob = json.loads(fpath.read_text())
                    feats = blob.get("features", [])
                    n = len(feats) if isinstance(feats, list) else None
                    rr = safe_float(blob.get("retention_ratio"))
                    ret = rr * 100 if rr is not None else None
                except Exception:
                    n = None
                    ret = None
            else:
                n = None
                ret = None
            rows.append((h, "intersection", n, ret))
    return rows


def load_nested_json(h: str) -> Optional[Dict]:
    f = NESTED_DIR / f"04c_{h}_embedded_nested.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def extract_method_stats(h: str) -> Dict[str, Dict[str, Optional[float]]]:
    result: Dict[str, Dict[str, Optional[float]]] = {"stability": {}, "auc": {}}
    nested = load_nested_json(h)
    if nested and isinstance(nested, dict):
        methods = nested.get("methods") or nested
        for m in METHODS:
            md = methods.get(m) if isinstance(methods, dict) else None
            if isinstance(md, dict):
                # Nested JSON uses 'stability_nogueira' and wraps metrics under 'performance'
                stab = safe_float(md.get("stability_nogueira"))
                perf = md.get("performance") if isinstance(md.get("performance"), dict) else {}
                auc = safe_float(perf.get("roc_auc_mean"))
                result["stability"][m] = stab
                result["auc"][m] = auc
            else:
                result["stability"][m] = None
                result["auc"][m] = None
        # Fallback fill for any missing AUCs from non-nested selected JSONs
        candidates = list(RESULTS_DIR.glob(f"04c_{h}*.json"))
        for cand in candidates:
            try:
                data = json.loads(cand.read_text())
            except Exception:
                continue
            methods_nn = data.get("methods") or data
            if not isinstance(methods_nn, dict):
                continue
            for m in METHODS:
                if result["auc"].get(m) is None and isinstance(methods_nn.get(m), dict):
                    perf = methods_nn[m].get("performance") if isinstance(methods_nn[m].get("performance"), dict) else {}
                    auc = safe_float(perf.get("roc_auc_mean") or methods_nn[m].get("roc_auc_mean") or methods_nn[m].get("auc_mean"))
                    if auc is not None:
                        result["auc"][m] = auc
        return result

    # Fallback: try baseline JSONs for AUC
    candidates = list(RESULTS_DIR.glob(f"04c_{h}*.json"))
    for cand in candidates:
        try:
            data = json.loads(cand.read_text())
        except Exception:
            continue
        methods = data.get("methods") or data
        for m in METHODS:
            md = methods.get(m) if isinstance(methods, dict) else None
            if isinstance(md, dict):
                # Old (non-nested) JSON wraps metrics under 'performance'
                perf = md.get("performance") if isinstance(md.get("performance"), dict) else {}
                auc = safe_float(perf.get("roc_auc_mean") or md.get("roc_auc_mean") or md.get("auc_mean"))
                result["auc"][m] = auc
            if m not in result["stability"]:
                result["stability"][m] = None
    return result


def tex_tabular_row(cells: List[str]) -> str:
    return " & ".join(cells) + " " + (chr(92) * 2)


def write_tex_consensus(rows: List[Tuple[str, str, Optional[int], Optional[float]]]):
    out = TABLES_DIR / "phase04_consensus_summary.tex"
    lines = [
        "% Auto-generated: Phase 04 consensus summary (intersection/majority)",
        "% Columns: Horizon, Set, #Features, Retention",
        "\\begin{tabular}{l l r r}",
        "\\toprule",
        tex_tabular_row(["Horizon", "Set", "\\#Features", "Retention (\\%)"]),
        "\\midrule",
    ]
    for h, s, n, ret in rows:
        n_str = str(n) if n is not None else "--"
        r_str = f"{ret:.1f}" if ret is not None else "--"
        lines.append(tex_tabular_row([h, s, n_str, r_str]))
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def write_tex_stability(stab_map_all: Dict[str, Dict[str, Dict[str, Optional[float]]]]):
    out = TABLES_DIR / "phase04_stability_summary.tex"
    lines = [
        "% Auto-generated: Phase 04 nested stability summary",
        "% Columns: Horizon, Method, Stability",
        "\\begin{tabular}{l l r}",
        "\\toprule",
        tex_tabular_row(["Horizon", "Method", "Stability"]),
        "\\midrule",
    ]
    for h in HORIZONS:
        stab_map = stab_map_all.get(h, {}).get("stability", {})
        for m in ["lasso", "elastic_net", "ridge"]:
            mn = m.replace("_", "\\_")
            val = stab_map.get(m)
            s = f"{val:.3f}" if val is not None else "--"
            lines.append(tex_tabular_row([h, mn, s]))
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def write_tex_auc(auc_map_all: Dict[str, Dict[str, Dict[str, Optional[float]]]]):
    out = TABLES_DIR / "phase04_method_auc.tex"
    lines = [
        "% Auto-generated: Phase 04 per-method ROC-AUC summary",
        "% Columns: Horizon, Method, ROC-AUC (mean)",
        "\\begin{tabular}{l l r}",
        "\\toprule",
        tex_tabular_row(["Horizon", "Method", "ROC-AUC"]),
        "\\midrule",
    ]
    for h in HORIZONS:
        auc_map = auc_map_all.get(h, {}).get("auc", {})
        for m in METHODS:
            mn = m.replace("_", "\\_")
            val = auc_map.get(m)
            s = f"{val:.3f}" if val is not None else "--"
            lines.append(tex_tabular_row([h, mn, s]))
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    cons_rows = summarize_consensus()
    write_tex_consensus(cons_rows)

    stats_by_h: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for h in HORIZONS:
        stats_by_h[h] = extract_method_stats(h)

    write_tex_stability(stats_by_h)
    write_tex_auc(stats_by_h)

    print("Generated:")
    print(f" - {TABLES_DIR / 'phase04_consensus_summary.tex'}")
    print(f" - {TABLES_DIR / 'phase04_stability_summary.tex'}")
    print(f" - {TABLES_DIR / 'phase04_method_auc.tex'}")


if __name__ == "__main__":
    main()
