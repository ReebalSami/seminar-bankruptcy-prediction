#!/usr/bin/env python3
"""
Phase 04e: Modeling (Train + Cross-Validation Evaluation)
=========================================================

Trains baseline models per horizon using the final consensus features from 04d:
- Logistic Regression (L2, lbfgs) with StandardScaler
- Random Forest (params sourced from config)

Outputs:
- results/04_modeling/H{h}_metrics.json
- results/04_modeling/ALL_summary.xlsx
- logs/04_modeling/04e_modeling.log

Notes:
- Read-only on feature sets; safe to run any time after 04d.
- Uses 5-fold Stratified CV on the whole horizon subset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import argparse


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
        logging.FileHandler(LOGS_DIR / "04e_modeling.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def cv_scores(pipeline, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    roc_list: List[float] = []
    pr_list: List[float] = []
    for tr, va in cv.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        pipeline.fit(Xtr, ytr)
        proba = pipeline.predict_proba(Xva)[:, 1]
        roc_list.append(roc_auc_score(yva, proba))
        pr_list.append(average_precision_score(yva, proba))
    return {
        "roc_auc_mean": float(np.mean(roc_list)),
        "roc_auc_std": float(np.std(roc_list)),
        "pr_auc_mean": float(np.mean(pr_list)),
        "pr_auc_std": float(np.std(pr_list)),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 04e Modeling: train + CV evaluate per horizon.")
    parser.add_argument("--horizons", "--h", nargs="*", type=int, help="Subset of horizons to run (e.g., --h 1 2)")
    args = parser.parse_args()
    with open(PROJECT_ROOT / "config" / "project_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    horizons: List[int] = cfg["datasets"]["polish"]["horizons"]
    if args.horizons:
        horizons = [h for h in horizons if h in set(args.horizons)]
    rf_cfg = cfg["feature_selection"]["random_forest"]
    random_state = cfg["analysis"]["random_state"]

    logger.info("=" * 80)
    logger.info("PHASE 04e: MODELING (Baseline LR + RF)")
    logger.info("=" * 80)

    logger.info(f"Loading dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df):,} observations")

    summary_rows = []

    for h in horizons:
        logger.info("-" * 80)
        logger.info(f"H{h}: preparing data and features")
        fpath = FEATURES_DIR / f"H{h}_features_final.json"
        if not fpath.exists():
            logger.warning(f"Final feature set not found: {fpath} — skipping H{h}")
            continue
        features_blob = json.loads(fpath.read_text())
        feats: List[str] = features_blob.get("features", [])
        if not feats:
            logger.warning(f"No features listed in {fpath} — skipping H{h}")
            continue

        df_h = df[df["horizon"] == h].copy()
        X = df_h[feats]
        y = df_h["bankrupt"]
        logger.info(f"H{h}: {len(df_h)} rows, {len(feats)} features; bankruptcy rate={y.mean()*100:.2f}%")

        models = {
            "logreg_l2": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=100000,
                    tol=1e-3,
                    class_weight="balanced",
                    C=1.0,
                    random_state=random_state,
                    n_jobs=1,
                )),
            ]),
            "random_forest": RandomForestClassifier(
                n_estimators=rf_cfg["n_estimators"],
                max_depth=rf_cfg["max_depth"],
                min_samples_split=rf_cfg["min_samples_split"],
                min_samples_leaf=rf_cfg["min_samples_leaf"],
                max_features=rf_cfg["max_features"],
                class_weight=rf_cfg["class_weight"],
                n_jobs=rf_cfg["n_jobs"],
                random_state=rf_cfg["random_state"],
                verbose=rf_cfg["verbose"],
            ),
        }

        horizon_rows = []
        metrics_per_model: Dict[str, Dict[str, float]] = {}

        for mname, model in models.items():
            logger.info(f"H{h}: CV evaluating model={mname}")
            if mname == "random_forest":
                # RF does not need scaling
                s = cv_scores(model, X, y, folds=5)
            else:
                s = cv_scores(model, X, y, folds=5)
            metrics_per_model[mname] = s
            horizon_rows.append({
                "Horizon": f"H{h}",
                "Model": mname,
                "ROC_AUC_Mean": s["roc_auc_mean"],
                "ROC_AUC_Std": s["roc_auc_std"],
                "PR_AUC_Mean": s["pr_auc_mean"],
                "PR_AUC_Std": s["pr_auc_std"],
            })

        # Persist per-horizon metrics JSON
        out_json = RESULTS_DIR / f"H{h}_metrics.json"
        with open(out_json, "w") as f:
            json.dump({"horizon": h, "metrics": metrics_per_model}, f, indent=2)
        logger.info(f"  ✓ Saved: {out_json}")

        # Append best model row marker
        best_model = max(metrics_per_model.items(), key=lambda kv: kv[1]["roc_auc_mean"])[0]
        best_auc = metrics_per_model[best_model]["roc_auc_mean"]
        summary_rows.append({
            "Horizon": f"H{h}",
            "Best_Model": best_model,
            "Best_ROC_AUC": best_auc,
            "Features_Used": len(feats),
        })

        # Write per-horizon Excel sheet as part of ALL_summary
        # (We aggregate after the loop.)

    # Aggregate summary to Excel
    all_summary_xlsx = RESULTS_DIR / "ALL_summary.xlsx"
    with pd.ExcelWriter(all_summary_xlsx, engine="openpyxl") as writer:
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        # Also include all per-horizon model tables if present
        for p in sorted(RESULTS_DIR.glob("H*_metrics.json")):
            blob = json.loads(p.read_text())
            h = blob.get("horizon")
            rows = []
            for mname, s in blob.get("metrics", {}).items():
                rows.append({
                    "Model": mname,
                    "ROC_AUC_Mean": s["roc_auc_mean"],
                    "ROC_AUC_Std": s["roc_auc_std"],
                    "PR_AUC_Mean": s["pr_auc_mean"],
                    "PR_AUC_Std": s["pr_auc_std"],
                })
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name=f"H{h}_Models", index=False)
    logger.info(f"✓ ALL summary written: {all_summary_xlsx}")

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 04e MODELING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
