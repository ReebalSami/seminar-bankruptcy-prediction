#!/usr/bin/env python3
"""
Phase 05c: CatBoost Cross-Validated Evaluation
==============================================

Adds CatBoost CV metrics per Horizont to existing Phase 05 modeling outputs
without destroying prior baseline (LR, RF) or extra model results.

Behavior:
 - Loads feature sets (EPV-konform) from Phase 04 (old approach)
 - For jede Horizon: 5-fach stratifizierte CV mit CatBoostClassifier
 - Ergänzt / erstellt JSON: results/05_modeling/H{h}_metrics.json (fügt Key 'catboost')
 - Aktualisiert Summary-Datei: results/05_modeling/ALL_summary.xlsx (Bestes Modell inkl. CatBoost)

Rationale:
 Repliziert archivierte Nutzung von CatBoost, aber unter identischem, rigorosem
 Cross-Validation-Schema wie übrige Modelle. Liefert konservativere Schätzung
 der generalisierbaren Güte.

Notes:
 - Keine Hyperparameter-Tuning-Schleife (Zeitersparnis); Parameter moderat.
 - Klassengewichte werden automatisch approximiert über inverse Häufigkeiten.
 - Sicher bei mehrfacher Ausführung (idempotent Update der JSONs + Summary).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "poland_imputed.parquet"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "feature_sets_selected"
RESULTS_DIR = PROJECT_ROOT / "results" / "05_modeling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PROJECT_ROOT / "logs" / "05_modeling"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "05c_catboost.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def cv_catboost(model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    roc_list: List[float] = []
    pr_list: List[float] = []
    for tr, va in cv.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        model.fit(Xtr, ytr, verbose=0)
        proba = model.predict_proba(Xva)[:, 1]
        roc_list.append(roc_auc_score(yva, proba))
        pr_list.append(average_precision_score(yva, proba))
    return {
        "roc_auc_mean": float(np.mean(roc_list)),
        "roc_auc_std": float(np.std(roc_list)),
        "pr_auc_mean": float(np.mean(pr_list)),
        "pr_auc_std": float(np.std(pr_list)),
    }


def _update_summary():
    """Recompute best model per horizon including newly added CatBoost metrics."""
    json_paths = sorted(RESULTS_DIR.glob("H*_metrics.json"))
    rows = []
    for p in json_paths:
        try:
            blob = json.loads(p.read_text())
        except Exception:
            continue
        h = blob.get("horizon")
        metrics = blob.get("metrics", {}) or {}
        if not metrics:
            continue
        best_model, best_auc = max(metrics.items(), key=lambda kv: kv[1].get("roc_auc_mean", -np.inf))
        rows.append({
            "Horizon": f"H{h}",
            "Best_Model": best_model,
            "Best_ROC_AUC": metrics[best_model]["roc_auc_mean"],
            # Features_Used: we retain existing value if present in previous summary
        })

    # Preserve existing Features_Used column if present
    summary_path = RESULTS_DIR / "ALL_summary.xlsx"
    existing_features_map: Dict[str, int] = {}
    if summary_path.exists():
        try:
            df_old = pd.read_excel(summary_path, sheet_name="Summary")
            for _, r in df_old.iterrows():
                h = str(r.get("Horizon"))
                if h and pd.notnull(r.get("Features_Used")):
                    existing_features_map[h] = int(r["Features_Used"])
        except Exception:
            pass

    for r in rows:
        h = r["Horizon"]
        if h in existing_features_map:
            r["Features_Used"] = existing_features_map[h]

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        if rows:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)
        # Also include all per-horizon model tables again (refresh)
        for p in json_paths:
            try:
                blob = json.loads(p.read_text())
            except Exception:
                continue
            h = blob.get("horizon")
            metrics = blob.get("metrics", {}) or {}
            rows_h = []
            for mname, vals in metrics.items():
                rows_h.append({
                    "Model": mname,
                    "ROC_AUC_Mean": vals.get("roc_auc_mean"),
                    "ROC_AUC_Std": vals.get("roc_auc_std"),
                    "PR_AUC_Mean": vals.get("pr_auc_mean"),
                    "PR_AUC_Std": vals.get("pr_auc_std"),
                })
            if rows_h:
                pd.DataFrame(rows_h).to_excel(writer, sheet_name=f"H{h}_Models", index=False)
    logger.info(f"✓ Summary updated (incl. CatBoost): {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 05c: CatBoost CV evaluation per horizon")
    parser.add_argument("--horizons", "--h", nargs="*", type=int, help="Subset of horizons (e.g., --h 1 2)")
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config" / "project_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    horizons: List[int] = cfg["datasets"]["polish"]["horizons"]
    if args.horizons:
        horizons = [h for h in horizons if h in set(args.horizons)]

    logger.info("=" * 80)
    logger.info("PHASE 05c: CATBOOST CV")
    logger.info("=" * 80)
    logger.info(f"Loading dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df):,} rows")

    for h in horizons:
        logger.info("-" * 80)
        logger.info(f"H{h}: feature set laden")
        fpath = FEATURES_DIR / f"H{h}_features_final.json"
        if not fpath.exists():
            logger.warning(f"Feature-Set fehlt: {fpath} -> skip H{h}")
            continue
        feats_blob = json.loads(fpath.read_text())
        feats: List[str] = feats_blob.get("features", [])
        if not feats:
            logger.warning(f"Keine Features in {fpath} -> skip H{h}")
            continue

        df_h = df[df["horizon"] == h].copy()
        X = df_h[feats]
        y = df_h["bankrupt"]
        pos_rate = y.mean()
        logger.info(f"H{h}: {len(df_h)} Beobachtungen, {len(feats)} Features, Insolvenzrate={pos_rate*100:.2f}%")

        # Class weights (inverse frequency) for mild imbalance handling
        w0 = 1.0
        w1 = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        class_weights = [w0, w1]

        cat_model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights=class_weights,
            random_seed=42,
            verbose=0,
        )

        logger.info(f"H{h}: CatBoost CV start")
        metrics = cv_catboost(cat_model, X, y, folds=5)
        logger.info(f"H{h}: CatBoost ROC-AUC(mean)={metrics['roc_auc_mean']:.4f} PR-AUC(mean)={metrics['pr_auc_mean']:.4f}")

        # Update / create JSON metrics file
        json_path = RESULTS_DIR / f"H{h}_metrics.json"
        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text())
            except Exception:
                existing = {"horizon": h, "metrics": {}}
        else:
            existing = {"horizon": h, "metrics": {}}
        existing.setdefault("metrics", {})["catboost"] = metrics
        json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        logger.info(f"✓ JSON aktualisiert: {json_path}")

    _update_summary()
    logger.info("=" * 80)
    logger.info("PHASE 05c CATBOOST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
