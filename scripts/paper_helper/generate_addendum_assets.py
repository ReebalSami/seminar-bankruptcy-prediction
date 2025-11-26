#!/usr/bin/env python3
"""
Generate Post-Submission Addendum assets (v1.0 -> v1.1 delta).
- Reads base results:   results/05_modeling[/extra], data/processed/feature_sets_selected/H*_features_final.json
- Reads nested results: results/05_modeling_nested[/extra], data/processed/feature_sets_selected/H*_features_final_nested.json
- Computes per-horizon winners (by ROC-AUC), PR-AUC, feature counts, and deltas
- Writes:
  * results/delta/v1_to_v1_1_delta.xlsx
  * seminar-paper/tables/addendum_delta_table.tex
  * seminar-paper/figures/addendum/delta_auc.png
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
BASE_MOD = ROOT / "results" / "05_modeling"
BASE_EXTRA = BASE_MOD / "extra"
NEST_MOD = ROOT / "results" / "05_modeling_nested"
NEST_EXTRA = NEST_MOD / "extra"
FEAT_DIR = ROOT / "data" / "processed" / "feature_sets_selected"
DELTA_DIR = ROOT / "results" / "delta"
DELTA_DIR.mkdir(parents=True, exist_ok=True)
TBL_OUT = ROOT / "seminar-paper" / "tables" / "addendum_delta_table.tex"
FIG_DIR = ROOT / "seminar-paper" / "figures" / "addendum"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUT = FIG_DIR / "delta_auc.png"
XLSX_OUT = DELTA_DIR / "v1_to_v1_1_delta.xlsx"


def _read_metrics_dir(d: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    by_h: Dict[str, Dict[str, Dict[str, float]]] = {}
    for p in sorted(d.glob("H*_metrics.json")):
        try:
            blob = json.loads(p.read_text())
        except Exception:
            continue
        h = blob.get("horizon")
        if h is None:
            continue
        key = f"H{int(h)}"
        by_h.setdefault(key, {})
        for mname, vals in (blob.get("metrics", {}) or {}).items():
            by_h[key][str(mname)] = {
                "roc_auc_mean": float(vals.get("roc_auc_mean")) if vals.get("roc_auc_mean") is not None else float("nan"),
                "pr_auc_mean": float(vals.get("pr_auc_mean")) if vals.get("pr_auc_mean") is not None else float("nan"),
            }
    return by_h


def _best_overall(base: Path, extra: Path) -> Dict[str, Tuple[str, float, float]]:
    agg = {}
    a = _read_metrics_dir(base)
    b = _read_metrics_dir(extra) if extra.exists() else {}
    horizons = sorted(set(a.keys()) | set(b.keys()), key=lambda x: int(x[1:]))
    best: Dict[str, Tuple[str, float, float]] = {}
    for h in horizons:
        cand: Dict[str, Dict[str, float]] = {}
        cand.update(a.get(h, {}))
        cand.update(b.get(h, {}))
        if not cand:
            continue
        m, s = max(cand.items(), key=lambda kv: kv[1].get("roc_auc_mean", float("-inf")))
        best[h] = (m, float(s.get("roc_auc_mean")), float(s.get("pr_auc_mean")))
    return best


def _feat_count(h: str, nested: bool) -> int:
    suffix = "_nested" if nested else ""
    fp = FEAT_DIR / f"{h}_features_final{suffix}.json"
    if not fp.exists():
        return 0
    try:
        data = json.loads(fp.read_text())
        return int(len(data.get("features", [])))
    except Exception:
        return 0


def build_delta() -> pd.DataFrame:
    base_best = _best_overall(BASE_MOD, BASE_EXTRA)
    nest_best = _best_overall(NEST_MOD, NEST_EXTRA)
    horizons = sorted(set(base_best.keys()) | set(nest_best.keys()), key=lambda x: int(x[1:]))
    rows: List[Dict] = []
    for h in horizons:
        wb, ab, pb = base_best.get(h, ("-", float("nan"), float("nan")))
        wn, an, pn = nest_best.get(h, ("-", float("nan"), float("nan")))
        fb = _feat_count(h, nested=False)
        fn = _feat_count(h, nested=True)
        rows.append({
            "Horizon": h,
            "Winner_Base": wb,
            "AUC_Base": ab,
            "PR_Base": pb,
            "Feat_Base": fb,
            "Winner_Nested": wn,
            "AUC_Nested": an,
            "PR_Nested": pn,
            "Feat_Nested": fn,
            "Delta_AUC": an - ab if pd.notnull(an) and pd.notnull(ab) else float("nan"),
            "Delta_PR": pn - pb if pd.notnull(pn) and pd.notnull(pb) else float("nan"),
            "Delta_Feat": fn - fb if pd.notnull(fn) and pd.notnull(fb) else float("nan"),
        })
    return pd.DataFrame(rows)


def _latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = ""
    for ch in s:
        out += repl.get(ch, ch)
    return out


def _fmt_f(x: float) -> str:
    try:
        if pd.isna(x):
            return "--"
        return f"{float(x):.3f}"
    except Exception:
        return "--"


def _fmt_i(x) -> str:
    try:
        if pd.isna(x):
            return "--"
        return str(int(x))
    except Exception:
        return "--"


def write_table(df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("% Auto-generated delta table (v1.0 -> v1.1)")
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Delta (v1.0 $\rightarrow$ v1.1): Gewinner, ROC-AUC, PR-AUC und Feature-Anzahl je Horizont}")
    lines.append(r"  \label{tab:addendum_delta}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \begin{tabular}{l l r r r l r r r r r r}")
    lines.append(r"    \toprule")
    lines.append(r"    Horizont & Gewinner (v1.0) & AUC (v1.0) & PR (v1.0) & Feats (v1.0) & Gewinner (v1.1) & AUC (v1.1) & PR (v1.1) & Feats (v1.1) & $\Delta$ AUC & $\Delta$ PR & $\Delta$ Feats \\")
    lines.append(r"    \midrule")
    for _, r in df.iterrows():
        wb = _latex_escape(str(r.get('Winner_Base', '')) or "--")
        wn = _latex_escape(str(r.get('Winner_Nested', '')) or "--")
        line = (
            f"    {r['Horizon']} & {wb} & "
            f"{_fmt_f(r['AUC_Base'])} & {_fmt_f(r['PR_Base'])} & {_fmt_i(r['Feat_Base'])} & "
            f"{wn} & {_fmt_f(r['AUC_Nested'])} & {_fmt_f(r['PR_Nested'])} & {_fmt_i(r['Feat_Nested'])} & "
            f"{_fmt_f(r['Delta_AUC'])} & {_fmt_f(r['Delta_PR'])} & {_fmt_i(r['Delta_Feat'])}"
        )
        lines.append(line + " \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print("No data for delta figure.")
        return
    long = pd.melt(
        df[["Horizon", "AUC_Base", "AUC_Nested"]].rename(columns={"AUC_Base": "v1.0", "AUC_Nested": "v1.1"}),
        id_vars="Horizon",
        var_name="Version",
        value_name="ROC_AUC",
    )
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.2, 3.6))
    order = sorted(df["Horizon"].tolist(), key=lambda x: int(x[1:]))
    ax = sns.barplot(data=long, x="Horizon", y="ROC_AUC", hue="Version", order=order, palette=["#9ecae1", "#3182bd"])
    ax.set_ylim(0.0, max(1.0, long["ROC_AUC"].max() + 0.06))
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Horizont")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    df = build_delta()
    # Persist
    df.to_excel(XLSX_OUT, index=False)
    write_table(df, TBL_OUT)
    write_figure(df, FIG_OUT)
    print(f"Generated delta assets:\n- {XLSX_OUT}\n- {TBL_OUT}\n- {FIG_OUT}")


if __name__ == "__main__":
    main()
