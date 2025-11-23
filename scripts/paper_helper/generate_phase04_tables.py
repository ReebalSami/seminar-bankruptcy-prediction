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
        df = pd.read_excel(xlsx_path)
        return df
    except Exception:
        return None


def summarize_consensus() -> List[Tuple[str, str, Optional[int], Optional[float]]]:
    df = load_consensus_from_xlsx()
    rows: List[Tuple[str, str, Optional[int], Optional[float]]] = []
    if df is not None:
        horizon_col = next((c for c in df.columns if re.search(r"horizon", str(c), re.IGNORECASE)), None)
        set_col = next((c for c in df.columns if re.search(r"set|consensus", str(c), re.IGNORECASE)), None)
        nfeat_col = next((c for c in df.columns if re.search(r"feature.*count|#feat|n_feat", str(c), re.IGNORECASE)), None)
        retention_col = find_retention_column(df)
        if horizon_col:
            for h in HORIZONS:
                sub = df[df[horizon_col].astype(str).str.contains(h, case=False, na=False)]
                pick = sub
                if set_col:
                    inter = sub[sub[set_col].astype(str).str.contains("intersection", case=False, na=False)]
                    if len(inter) > 0:
                        pick = inter
                if len(pick) > 0:
                    r = pick.iloc[0]
                    n = int(r[nfeat_col]) if nfeat_col and pd.notna(r.get(nfeat_col)) else None
                    ret = safe_float(r.get(retention_col)) if retention_col else None
                    rows.append((h, "intersection", n, ret))
                else:
                    rows.append((h, "intersection", None, None))
        else:
            rows = [(h, "intersection", None, None) for h in HORIZONS]
    else:
        for h in HORIZONS:
            fpath = FINAL_FEATURES_DIR / f"{h}_features_final.json"
            if fpath.exists():
                try:
                    feats = json.loads(fpath.read_text())
                    n = len(feats)
                except Exception:
                    n = None
            else:
                n = None
            rows.append((h, "intersection", n, None))
    return rows


def load_nested_json(h: str) -> Optional[Dict]:
    f = NESTED_DIR / f"04c_{h}_embedded_nested.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def extract_method_stats(h: str) -> List[Tuple[str, str, Optional[float]]]:
    nested = load_nested_json(h)
    rows: List[Tuple[str, str, Optional[float]]] = []
    if nested and isinstance(nested, dict):
        methods = nested.get("methods") or nested
        for m in METHODS:
            md = methods.get(m) if isinstance(methods, dict) else None
            if isinstance(md, dict):
                stab = safe_float(md.get("stability") or md.get("nogueira_stability") or md.get("stability_mean"))
                auc = None
                if "fold_metrics" in md and isinstance(md["fold_metrics"], dict):
                    ra = md["fold_metrics"].get("roc_auc")
                    if isinstance(ra, list) and len(ra):
                        auc = safe_float(pd.Series(ra).mean())
                if auc is None:
                    auc = safe_float(md.get("roc_auc_mean") or md.get("auc_mean"))
                rows.append((m, "stability", stab))
                rows.append((m, "auc", auc))
            else:
                rows.append((m, "stability", None))
                rows.append((m, "auc", None))
        return rows

    auc_rows: List[Tuple[str, str, Optional[float]]] = []
    candidates = list(RESULTS_DIR.glob(f"04c_{h}*.json"))
    for cand in candidates:
        try:
            data = json.loads(cand.read_text())
        except Exception:
            continue
        methods = data.get("methods") or data
        for m in METHODS:
            md = methods.get(m) if isinstance(methods, dict) else None
            auc = None
            if isinstance(md, dict):
                auc = safe_float(md.get("roc_auc_mean") or md.get("auc_mean"))
            auc_rows.append((m, "auc", auc))
    for m in METHODS:
        auc_present = next((v for k, t, v in auc_rows if k == m and t == "auc" and v is not None), None)
        rows.append((m, "stability", None))
        rows.append((m, "auc", auc_present))
    return rows


def write_tex_consensus(rows: List[Tuple[str, str, Optional[int], Optional[float]]]):
    out = TABLES_DIR / "phase04_consensus_summary.tex"
    br = chr(92) * 2
    lines = [
        "% Auto-generated: Phase 04 consensus summary (intersection/majority)",
        "% Columns: Horizon, Set, #Features, Retention",
        "\\begin{tabular}{l l r r}",
        "\\toprule",
        "Horizon & Set & \\#Features & Retention (\\%) " + br,
        "\\midrule",
    ]
    for h, s, n, ret in rows:
        n_str = str(n) if n is not None else "--"
        r_str = f"{ret:.1f}" if ret is not None else "--"
        lines.append(f"{h} & {s} & {n_str} & {r_str} " + br)
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def write_tex_stability(stab_map: Dict[str, Dict[str, Optional[float]]]):
    out = TABLES_DIR / "phase04_stability_summary.tex"
    br = chr(92) * 2
    lines = [
        "% Auto-generated: Phase 04 nested stability summary",
        "% Columns: Horizon, Method, Stability",
        "\\begin{tabular}{l l r}",
        "\\toprule",
        "Horizon & Method & Stability " + br,
        "\\midrule",
    ]
    for h in HORIZONS:
        for m in ["lasso", "elastic_net", "ridge"]:
            mn = m.replace("_", "\\_")
            val = stab_map.get(h, {}).get(m)
            s = f"{val:.3f}" if val is not None else "--"
            lines.append(f"{h} & {mn} & {s} " + br)
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def write_tex_auc(auc_map: Dict[str, Dict[str, Optional[float]]]):
    out = TABLES_DIR / "phase04_method_auc.tex"
    br = chr(92) * 2
    lines = [
        "% Auto-generated: Phase 04 per-method ROC-AUC summary",
        "% Columns: Horizon, Method, ROC-AUC (mean)",
        "\\begin{tabular}{l l r}",
        "\\toprule",
        "Horizon & Method & ROC-AUC " + br,
        "\\midrule",
    ]
    for h in HORIZONS:
        for m in METHODS:
            mn = m.replace("_", "\\_")
            val = auc_map.get(h, {}).get(m)
            s = f"{val:.3f}" if val is not None else "--"
            lines.append(f"{h} & {mn} & {s} " + br)
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    cons_rows = summarize_consensus()
    write_tex_consensus(cons_rows)

    stab_map: Dict[str, Dict[str, Optional[float]]] = {}
    auc_map: Dict[str, Dict[str, Optional[float]]] = {}
    for h in HORIZONS:
        rows = extract_method_stats(h)
        stab_map[h] = {}
        auc_map[h] = {}
        for m, t, v in rows:
            if t == "stability":
                stab_map[h][m] = v
            elif t == "auc":
                auc_map[h][m] = v

    write_tex_stability(stab_map)
    write_tex_auc(auc_map)

    print("Generated:")
    print(f" - {TABLES_DIR / 'phase04_consensus_summary.tex'}")
    print(f" - {TABLES_DIR / 'phase04_stability_summary.tex'}")
    print(f" - {TABLES_DIR / 'phase04_method_auc.tex'}")


if __name__ == "__main__":
    main()

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
        df = pd.read_excel(xlsx_path)
        return df
    except Exception:
        return None


def summarize_consensus() -> List[Tuple[str, str, Optional[int], Optional[float]]]:
    df = load_consensus_from_xlsx()
    rows: List[Tuple[str, str, Optional[int], Optional[float]]] = []
    if df is not None:
        horizon_col = next((c for c in df.columns if re.search(r"horizon", str(c), re.IGNORECASE)), None)
        set_col = next((c for c in df.columns if re.search(r"set|consensus", str(c), re.IGNORECASE)), None)
        nfeat_col = next((c for c in df.columns if re.search(r"feature.*count|#feat|n_feat", str(c), re.IGNORECASE)), None)
        retention_col = find_retention_column(df)
        if horizon_col:
            for h in HORIZONS:
                sub = df[df[horizon_col].astype(str).str.contains(h, case=False, na=False)]
                if set_col:
                    inter = sub[sub[set_col].astype(str).str.contains("intersection", case=False, na=False)]
                    pick = inter if len(inter) else sub
                else:
                    pick = sub
                if len(pick) > 0:
                    r = pick.iloc[0]
                    n = int(r[nfeat_col]) if nfeat_col and pd.notna(r.get(nfeat_col)) else None
                    ret = safe_float(r.get(retention_col)) if retention_col else None
                    rows.append((h, "intersection", n, ret))
                else:
                    rows.append((h, "intersection", None, None))
        else:
            rows = [(h, "intersection", None, None) for h in HORIZONS]
    else:
        for h in HORIZONS:
            fpath = FINAL_FEATURES_DIR / f"{h}_features_final.json"
            if fpath.exists():
                try:
                    feats = json.loads(fpath.read_text())
                    n = len(feats)
                except Exception:
                    n = None
            else:
                n = None
            rows.append((h, "intersection", n, None))
    return rows


def load_nested_json(h: str) -> Optional[Dict]:
    f = NESTED_DIR / f"04c_{h}_embedded_nested.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def extract_method_stats(h: str) -> List[Tuple[str, str, Optional[float]]]:
    nested = load_nested_json(h)
    rows: List[Tuple[str, str, Optional[float]]] = []
    if nested and isinstance(nested, dict):
        methods = nested.get("methods") or nested
        for m in METHODS:
            md = methods.get(m) if isinstance(methods, dict) else None
            if isinstance(md, dict):
                stab = safe_float(md.get("stability") or md.get("nogueira_stability") or md.get("stability_mean"))
                auc = None
                if "fold_metrics" in md and isinstance(md["fold_metrics"], dict):
                    ra = md["fold_metrics"].get("roc_auc")
                    if isinstance(ra, list) and len(ra):
                        auc = safe_float(pd.Series(ra).mean())
                if auc is None:
                    auc = safe_float(md.get("roc_auc_mean") or md.get("auc_mean"))
                rows.append((m, "stability", stab))
                rows.append((m, "auc", auc))
            else:
                rows.append((m, "stability", None))
                rows.append((m, "auc", None))
        return rows

    auc_rows: List[Tuple[str, str, Optional[float]]] = []
    candidates = list(RESULTS_DIR.glob(f"04c_{h}*.json"))
    for cand in candidates:
        try:
            data = json.loads(cand.read_text())
        except Exception:
            continue
        methods = data.get("methods") or data
        for m in METHODS:
            md = methods.get(m) if isinstance(methods, dict) else None
            auc = None
            if isinstance(md, dict):
                auc = safe_float(md.get("roc_auc_mean") or md.get("auc_mean"))
            auc_rows.append((m, "auc", auc))
    for m in METHODS:
        auc_present = next((v for k, t, v in auc_rows if k == m and t == "auc" and v is not None), None)
        rows.append((m, "stability", None))
        rows.append((m, "auc", auc_present))
    return rows


def write_tex_consensus(rows: List[Tuple[str, str, Optional[int], Optional[float]]]):
    out = TABLES_DIR / "phase04_consensus_summary.tex"
    lines = [
        "% Auto-generated: Phase 04 consensus summary (intersection/majority)",
        "% Columns: Horizon, Set, #Features, Retention",
        "\\begin{tabular}{l l r r}",
        "\\toprule",
            "Horizon & Set & \\#Features & Retention (\\%) " + chr(92)*2,
        "\\midrule",
    ]
    for h, s, n, ret in rows:
        n_str = str(n) if n is not None else "--"
        r_str = f"{ret:.1f}" if ret is not None else "--"
        lines.append(f"{h} & {s} & {n_str} & {r_str} \\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def write_tex_stability(stab_map: Dict[str, Dict[str, Optional[float]]]):
    out = TABLES_DIR / "phase04_stability_summary.tex"
    lines = [
        "% Auto-generated: Phase 04 nested stability summary",
        "% Columns: Horizon, Method, Stability",
        "\\begin{tabular}{l l r}",
        "\\toprule",
            "Horizon & Method & Stability " + chr(92)*2,
        "\\midrule",
    ]
    for h in HORIZONS:
        for m in ["lasso", "elastic_net", "ridge"]:
            mn = m.replace("_", "\\_")
            val = stab_map.get(h, {}).get(m)
            s = f"{val:.3f}" if val is not None else "--"
            lines.append(f"{h} & {mn} & {s} \\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def write_tex_auc(auc_map: Dict[str, Dict[str, Optional[float]]]):
    out = TABLES_DIR / "phase04_method_auc.tex"
    lines = [
        "% Auto-generated: Phase 04 per-method ROC-AUC summary",
        "% Columns: Horizon, Method, ROC-AUC (mean)",
        "\\begin{tabular}{l l r}",
        "\\toprule",
            "Horizon & Method & ROC-AUC " + chr(92)*2,
        "\\midrule",
    ]
    for h in HORIZONS:
        for m in METHODS:
            mn = m.replace("_", "\\_")
            val = auc_map.get(h, {}).get(m)
            s = f"{val:.3f}" if val is not None else "--"
            lines.append(f"{h} & {mn} & {s} \\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    cons_rows = summarize_consensus()
    write_tex_consensus(cons_rows)

    stab_map: Dict[str, Dict[str, Optional[float]]] = {}
    auc_map: Dict[str, Dict[str, Optional[float]]] = {}
    for h in HORIZONS:
        rows = extract_method_stats(h)
        stab_map[h] = {}
        auc_map[h] = {}
        for m, t, v in rows:
            if t == "stability":
                stab_map[h][m] = v
            elif t == "auc":
                auc_map[h][m] = v

    write_tex_stability(stab_map)
    write_tex_auc(auc_map)

    print("Generated:")
    print(f" - {TABLES_DIR / 'phase04_consensus_summary.tex'}")
    print(f" - {TABLES_DIR / 'phase04_stability_summary.tex'}")
    print(f" - {TABLES_DIR / 'phase04_method_auc.tex'}")


if __name__ == "__main__":
    main()

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
        df = pd.read_excel(xlsx_path)
        return df
    except Exception:
        return None


def summarize_consensus() -> List[Tuple[str, str, Optional[int], Optional[float]]]:
    # Prefer the consolidated XLSX if present
    df = load_consensus_from_xlsx()
    rows: List[Tuple[str, str, Optional[int], Optional[float]]] = []
    if df is not None:
        # Try to filter for intersection rows if multi-set present
        horizon_col = next((c for c in df.columns if re.search(r"horizon", str(c), re.IGNORECASE)), None)
        set_col = next((c for c in df.columns if re.search(r"set|consensus", str(c), re.IGNORECASE)), None)
        nfeat_col = next((c for c in df.columns if re.search(r"feature.*count|#feat|n_feat", str(c), re.IGNORECASE)), None)
        retention_col = find_retention_column(df)
        if horizon_col:
            for h in HORIZONS:
                sub = df[df[horizon_col].astype(str).str.contains(h, case=False, na=False)]
                if set_col:
                    # Prefer intersection if available
                    inter = sub[sub[set_col].astype(str).str.contains("intersection", case=False, na=False)]
                    pick = inter if len(inter) else sub
                else:
                    pick = sub
                if len(pick) > 0:
                    r = pick.iloc[0]
                    n = int(r[nfeat_col]) if nfeat_col and pd.notna(r.get(nfeat_col)) else None
                    ret = safe_float(r.get(retention_col)) if retention_col else None
                    rows.append((h, "intersection", n, ret))
                else:
                    rows.append((h, "intersection", None, None))
        else:
            # Fallback: nothing parsable
            rows = [(h, "intersection", None, None) for h in HORIZONS]
    else:
        # Fallback to counting final JSON feature sets
        for h in HORIZONS:
            fpath = FINAL_FEATURES_DIR / f"{h}_features_final.json"
            if fpath.exists():
                try:
                    feats = json.loads(fpath.read_text())
                    n = len(feats)
                except Exception:
                    n = None
            else:
                n = None
            rows.append((h, "intersection", n, None))
    return rows


def load_nested_json(h: str) -> Optional[Dict]:
    f = NESTED_DIR / f"04c_{h}_embedded_nested.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def extract_method_stats(h: str) -> List[Tuple[str, str, Optional[float]]]:
    # Try nested first (preferred)
    nested = load_nested_json(h)
    rows: List[Tuple[str, str, Optional[float]]] = []
    if nested and isinstance(nested, dict):
        methods = nested.get("methods") or nested
        for m in METHODS:
            md = methods.get(m) if isinstance(methods, dict) else None
            if isinstance(md, dict):
                # stability
                stab = safe_float(md.get("stability") or md.get("nogueira_stability") or md.get("stability_mean"))
                # auc (mean across folds or precomputed)
                auc = None
                if "fold_metrics" in md and isinstance(md["fold_metrics"], dict):
                    ra = md["fold_metrics"].get("roc_auc")
                    if isinstance(ra, list) and len(ra):
                        auc = safe_float(pd.Series(ra).mean())
                if auc is None:
                    auc = safe_float(md.get("roc_auc_mean") or md.get("auc_mean"))
                rows.append((m, "stability", stab))
                rows.append((m, "auc", auc))
            else:
                rows.append((m, "stability", None))
                rows.append((m, "auc", None))
        return rows

    # Fallback: try baseline 04c JSON per horizon (unknown exact filename schema)
    auc_rows: List[Tuple[str, str, Optional[float]]] = []
    candidates = list(RESULTS_DIR.glob(f"04c_{h}*.json"))
    for cand in candidates:
        try:
            data = json.loads(cand.read_text())
        except Exception:
            continue
        methods = data.get("methods") or data
        for m in METHODS:
            md = methods.get(m) if isinstance(methods, dict) else None
            auc = None
            if isinstance(md, dict):
                auc = safe_float(md.get("roc_auc_mean") or md.get("auc_mean"))
            auc_rows.append((m, "auc", auc))
    # No stability in baseline
    for m in METHODS:
        auc_present = next((v for k, t, v in auc_rows if k == m and t == "auc" and v is not None), None)
        rows.append((m, "stability", None))
        rows.append((m, "auc", auc_present))
    return rows


def write_tex_consensus(rows: List[Tuple[str, str, Optional[int], Optional[float]]]):
    out = TABLES_DIR / "phase04_consensus_summary.tex"
    lines = [
        "% Auto-generated: Phase 04 consensus summary (intersection/majority)",
        "% Columns: Horizon, Set, #Features, Retention",
        "\\begin{tabular}{l l r r}",
        "\\toprule",
        "Horizon & Set & \\#Features & Retention (\\%) \\\",
        "\\midrule",
    ]
    for h, s, n, ret in rows:
        n_str = str(n) if n is not None else "--"
        r_str = f"{ret:.1f}" if ret is not None else "--"
        lines.append(f"{h} & {s} & {n_str} & {r_str} \\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def write_tex_stability(stab_map: Dict[str, Dict[str, Optional[float]]]):
    out = TABLES_DIR / "phase04_stability_summary.tex"
    lines = [
        "% Auto-generated: Phase 04 nested stability summary",
        "% Columns: Horizon, Method, Stability",
        "\\begin{tabular}{l l r}",
        "\\toprule",
        "Horizon & Method & Stability \\\",
        "\\midrule",
    ]
    for h in HORIZONS:
        for m in ["lasso", "elastic_net", "ridge"]:
            mn = m.replace("_", "\\_")
            val = stab_map.get(h, {}).get(m)
            s = f"{val:.3f}" if val is not None else "--"
            lines.append(f"{h} & {mn} & {s} \\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def write_tex_auc(auc_map: Dict[str, Dict[str, Optional[float]]]):
    out = TABLES_DIR / "phase04_method_auc.tex"
    lines = [
        "% Auto-generated: Phase 04 per-method ROC-AUC summary",
        "% Columns: Horizon, Method, ROC-AUC (mean)",
        "\\begin{tabular}{l l r}",
        "\\toprule",
        "Horizon & Method & ROC-AUC \\\",
        "\\midrule",
    ]
    for h in HORIZONS:
        for m in METHODS:
            mn = m.replace("_", "\\_")
            val = auc_map.get(h, {}).get(m)
            s = f"{val:.3f}" if val is not None else "--"
            lines.append(f"{h} & {mn} & {s} \\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Consensus
    cons_rows = summarize_consensus()
    write_tex_consensus(cons_rows)

    # Method stats (stability + AUC)
    stab_map: Dict[str, Dict[str, Optional[float]]] = {}
    auc_map: Dict[str, Dict[str, Optional[float]]] = {}
    for h in HORIZONS:
        rows = extract_method_stats(h)
        stab_map[h] = {}
        auc_map[h] = {}
        for m, t, v in rows:
            if t == "stability":
                stab_map[h][m] = v
            elif t == "auc":
                auc_map[h][m] = v

    write_tex_stability(stab_map)
    write_tex_auc(auc_map)

    print("Generated:")
    print(f" - {TABLES_DIR / 'phase04_consensus_summary.tex'}")
    print(f" - {TABLES_DIR / 'phase04_stability_summary.tex'}")
    print(f" - {TABLES_DIR / 'phase04_method_auc.tex'}")


if __name__ == "__main__":
    main()
