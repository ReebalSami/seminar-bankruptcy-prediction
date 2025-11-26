#!/usr/bin/env python3
"""
Phase 05b: Additional Models + Soft-Voting Ensemble
===================================================

Trains additional models per horizon using the final feature sets from Phase 04 (old approach):
- GradientBoostingClassifier
- ExtraTreesClassifier
- SVC (probability=True, class_weight=balanced) within a StandardScaler pipeline
- Soft-Voting ensemble combining LR(L2, scaled) + RF + GB

Outputs:
- results/05_modeling/extra/H{h}_metrics.json
- results/05_modeling/extra/ALL_summary_extra.xlsx
- logs/05_modeling/05b_extra_models.log

All evaluations use 5-fold Stratified CV on the horizon subset. Metrics: ROC-AUC, PR-AUC.
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
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "poland_imputed.parquet"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "feature_sets_selected"
RESULTS_DIR_BASE = PROJECT_ROOT / "results"

LOGS_DIR = PROJECT_ROOT / "logs" / "05_modeling"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "05b_extra_models.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def cv_scores(model, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    roc_list: List[float] = []
    pr_list: List[float] = []
    for tr, va in cv.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xva)[:, 1]
        roc_list.append(roc_auc_score(yva, proba))
        pr_list.append(average_precision_score(yva, proba))
    return {
        "roc_auc_mean": float(np.mean(roc_list)),
        "roc_auc_std": float(np.std(roc_list)),
        "pr_auc_mean": float(np.mean(pr_list)),
        "pr_auc_std": float(np.std(pr_list)),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 05b: Additional models + soft-voting ensemble")
    parser.add_argument("--horizons", "--h", nargs="*", type=int, help="Subset of horizons to run (e.g., --h 1 2)")
    parser.add_argument("--variant", choices=["base", "nested"], default="base", help="Use nested feature files and write to nested results dir")
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config" / "project_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    horizons: List[int] = cfg["datasets"]["polish"]["horizons"]
    if args.horizons:
        horizons = [h for h in horizons if h in set(args.horizons)]

    rf_cfg = cfg["feature_selection"]["random_forest"]
    random_state = cfg["analysis"]["random_state"]

    # Variant handling
    features_suffix = "_nested" if args.variant == "nested" else ""
    if args.variant == "nested":
        results_dir = RESULTS_DIR_BASE / "05_modeling_nested" / "extra"
    else:
        results_dir = RESULTS_DIR_BASE / "05_modeling" / "extra"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PHASE 05b: EXTRA MODELS + ENSEMBLE")
    logger.info("=" * 80)

    logger.info(f"Loading dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df):,} observations")

    summary_rows = []

    for h in horizons:
        logger.info("-" * 80)
        logger.info(f"H{h}: preparing data and features")
        fpath = FEATURES_DIR / f"H{h}_features_final{features_suffix}.json"
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

        # Base learners
        lr_scaled = Pipeline([
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
        ])

        gb = GradientBoostingClassifier(
            random_state=random_state,
        )

        et = ExtraTreesClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_split=rf_cfg["min_samples_split"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            max_features=rf_cfg["max_features"],
            class_weight=rf_cfg["class_weight"],
            n_jobs=rf_cfg["n_jobs"],
            random_state=rf_cfg["random_state"],
        )

        svc_scaled = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, class_weight="balanced", random_state=random_state))
        ])

        ensemble_soft = VotingClassifier(
            estimators=[("lr", lr_scaled), ("gb", gb), ("et", et)],
            voting="soft",
            n_jobs=None,
            flatten_transform=True,
        )

        models = {
            "logreg_l2_scaled": lr_scaled,
            "gradboost": gb,
            "extratrees": et,
            "svc_scaled": svc_scaled,
            "softvote_lr_gb_et": ensemble_soft,
        }

        metrics_per_model: Dict[str, Dict[str, float]] = {}

        for mname, model in models.items():
            logger.info(f"H{h}: CV evaluating model={mname}")
            s = cv_scores(model, X, y, folds=5)
            metrics_per_model[mname] = s

        # Persist per-horizon metrics JSON
        out_json = results_dir / f"H{h}_metrics.json"
        with open(out_json, "w") as f:
            json.dump({"horizon": h, "metrics": metrics_per_model}, f, indent=2)
        logger.info(f"  ✓ Saved: {out_json}")

        best_model = max(metrics_per_model.items(), key=lambda kv: kv[1]["roc_auc_mean"])[0]
        best_auc = metrics_per_model[best_model]["roc_auc_mean"]
        summary_rows.append({
            "Horizon": f"H{h}",
            "Best_Model": best_model,
            "Best_ROC_AUC": best_auc,
            "Features_Used": len(feats),
        })

    # Aggregate summary to Excel
    all_summary_xlsx = results_dir / "ALL_summary_extra.xlsx"
    with pd.ExcelWriter(all_summary_xlsx, engine="openpyxl") as writer:
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        for p in sorted(results_dir.glob("H*_metrics.json")):
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
    logger.info(f"✓ ALL extra summary written: {all_summary_xlsx}")


if __name__ == "__main__":
    main()
