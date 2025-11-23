#!/usr/bin/env python3
"""
Phase 04c (Nested): Embedded Methods with Nested CV
===================================================

Goals:
- Perform feature selection inside training folds (nested selection) to avoid leakage
- Record per-fold selected features for each method
- Compute fold-level stability (Nogueira) per method
- Evaluate performance across outer folds using fold-specific features
- Save detailed JSON/Excel to results/04_feature_selection/nested

Methods:
- Lasso (L1, liblinear)
- Elastic Net (elasticnet, saga)
- Ridge (L2, lbfgs; take top-k by |coef|)

Notes:
- Uses StandardScaler where appropriate
- Adds tqdm progress bars
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings


# =============================================================================
# Setup
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_SETS_DIR = DATA_DIR / "feature_sets"
RESULTS_BASE = PROJECT_ROOT / "results" / "04_feature_selection"
RESULTS_DIR = RESULTS_BASE / "nested"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = PROJECT_ROOT / "logs" / "04_feature_selection"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "04c_embedded_methods_nested.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

with open(PROJECT_ROOT / "config" / "project_config.yaml", "r") as f:
    config = yaml.safe_load(f)

fs_config = config["feature_selection"]
HORIZONS = config["datasets"]["polish"]["horizons"]
RANDOM_STATE = config["analysis"]["random_state"]
OUTER_FOLDS = fs_config["outer_folds"]
INNER_FOLDS = fs_config["embedded_cv_folds"]


# =============================================================================
# Helpers
# =============================================================================

def compute_nogueira_stability(fold_selections: List[Set[str]], n_features_total: int) -> float:
    """Nogueira stability metric across folds for one method."""
    from itertools import combinations
    k = len(fold_selections)
    if k < 2:
        return 1.0
    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 1.0
        u = len(a | b)
        return len(a & b) / u if u else 0.0
    pairs = list(combinations(fold_selections, 2))
    avg_jac = float(np.mean([jaccard(a, b) for a, b in pairs])) if pairs else 0.0
    k_avg = float(np.mean([len(s) for s in fold_selections]))
    p_rand = (k_avg / n_features_total) ** 2 if n_features_total > 0 else 0.0
    if p_rand >= 1:
        return avg_jac
    stab = (avg_jac - p_rand) / (1 - p_rand)
    return max(0.0, min(1.0, float(stab)))


def evaluate_fold(X_tr, y_tr, X_va, y_va) -> Dict[str, float]:
    """Evaluate with a standard L2 logistic pipeline on provided features."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            tol=1e-4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ])
    pipeline.fit(X_tr, y_tr)
    yhat = pipeline.predict_proba(X_va)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_va, yhat)),
        "pr_auc": float(average_precision_score(y_va, yhat)),
    }


# =============================================================================
# Method Selection (train-only)
# =============================================================================

def select_lasso(X: pd.DataFrame, y: pd.Series) -> List[str]:
    cfg = fs_config["lasso"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = LogisticRegressionCV(
        Cs=cfg["c_values"],
        cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        penalty=cfg["penalty"],
        solver=cfg["solver"],
        dual=cfg.get("dual", False),
        max_iter=cfg["max_iter"],
        tol=cfg["tol"],
        class_weight=cfg["class_weight"],
        scoring=fs_config["embedded_scoring"],
        n_jobs=cfg.get("n_jobs", 1),
        random_state=RANDOM_STATE,
        verbose=0,
    )
    cv.fit(Xs, y)
    coefs = cv.coef_[0]
    return X.columns[coefs != 0].tolist()


def _fit_lr_cv_with_retry(cv_factory, Xs, y, max_retries: int = 2):
    """Fit LogisticRegressionCV with retries on ConvergenceWarning."""
    last_model = None
    for attempt in range(max_retries + 1):
        model = cv_factory()
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(Xs, y)
            conv_warnings = [w for w in wlist if issubclass(w.category, ConvergenceWarning)]
        if not conv_warnings:
            return model
        last_model = model
        # Increase max_iter and relax tol for next attempt
        if hasattr(cv_factory, "max_iter"):
            cv_factory.max_iter = int(getattr(cv_factory, "max_iter", 50000) * 2)
        # Rebuild factory with updated params by closure mutation
        if hasattr(cv_factory, "_mutate"):
            cv_factory._mutate()
    return last_model


def select_elastic_net(X: pd.DataFrame, y: pd.Series) -> List[str]:
    cfg = fs_config["elastic_net"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Constrain grid to well-behaved values
    cs = [c for c in cfg["c_values"] if c <= 10]
    if not cs:
        cs = [0.1, 1, 10]
    l1rs = cfg["l1_ratio"]
    if len(l1rs) > 4:
        l1rs = l1rs[:4]

    # Build a small factory that we can mutate between retries
    state = {"max_iter": cfg["max_iter"], "tol": cfg["tol"]}

    def make_cv():
        return LogisticRegressionCV(
            Cs=cs,
            cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            penalty=cfg["penalty"],
            solver=cfg["solver"],
            l1_ratios=l1rs,
            max_iter=state["max_iter"],
            tol=state["tol"],
            class_weight=cfg["class_weight"],
            scoring=fs_config["embedded_scoring"],
            n_jobs=cfg["n_jobs"],
            random_state=RANDOM_STATE,
            verbose=0,
        )

    # Attach helpers for mutation
    def mutate():
        state["max_iter"] = int(state["max_iter"] * 2)
        state["tol"] = float(state["tol"] * 2)  # relax tolerance to reach stopping criterion

    make_cv.max_iter = state["max_iter"]  # type: ignore[attr-defined]
    make_cv._mutate = mutate  # type: ignore[attr-defined]

    cv = _fit_lr_cv_with_retry(make_cv, Xs, y, max_retries=2)
    coefs = cv.coef_[0]
    return X.columns[coefs != 0].tolist()


def select_ridge_topk(X: pd.DataFrame, y: pd.Series, top_k: int = 30) -> List[str]:
    cfg = fs_config["ridge"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = LogisticRegressionCV(
        Cs=cfg["c_values"],
        cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        penalty=cfg["penalty"],
        solver=cfg["solver"],
        max_iter=cfg["max_iter"],
        tol=cfg["tol"],
        class_weight=cfg["class_weight"],
        scoring=fs_config["embedded_scoring"],
        n_jobs=cfg["n_jobs"],
        random_state=RANDOM_STATE,
        verbose=0,
    )
    cv.fit(Xs, y)
    coefs = cv.coef_[0]
    idx = np.argsort(np.abs(coefs))[-top_k:]
    return X.columns[idx].tolist()


# =============================================================================
# Main per-horizon nested process
# =============================================================================

def process_horizon_nested(h: int, df: pd.DataFrame) -> Dict:
    # Load VIF features for horizon
    with open(FEATURE_SETS_DIR / f"H{h}_features.json", "r") as f:
        vif_feats: List[str] = json.load(f)

    df_h = df[df["horizon"] == h].copy()
    X = df_h[vif_feats]
    y = df_h["bankrupt"]

    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    methods = {
        "lasso": select_lasso,
        "elastic_net": select_elastic_net,
        "ridge": lambda Xtr, ytr: select_ridge_topk(Xtr, ytr, top_k=30),
    }

    fold_selections: Dict[str, List[List[str]]] = {m: [] for m in methods}
    fold_metrics: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}

    logger.info(f"H{h}: {len(df_h)} observations, {len(vif_feats)} VIF features; bankruptcy rate={y.mean()*100:.2f}%")

    for fold_idx, (tr, va) in enumerate(tqdm(list(outer.split(X, y)), desc=f"H{h} outer CV", unit="fold")):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        for mname, selector in methods.items():
            feats = selector(Xtr, ytr)
            fold_selections[mname].append(feats)

            # Evaluate using features selected on training fold
            if len(feats) == 0:
                fold_metrics[mname].append({"roc_auc": 0.0, "pr_auc": 0.0})
            else:
                em = evaluate_fold(Xtr[feats], ytr, Xva[feats], yva)
                fold_metrics[mname].append(em)

    # Aggregate
    results: Dict[str, Dict] = {}
    n_total = X.shape[1]
    for mname in methods.keys():
        sel_sets = [set(s) for s in fold_selections[mname]]
        stability = compute_nogueira_stability(sel_sets, n_total)
        # Majority across folds for a final consensus selection per method
        all_feats = [f for s in fold_selections[mname] for f in s]
        counts = pd.Series(all_feats).value_counts() if len(all_feats) else pd.Series(dtype=int)
        majority_threshold = max(1, int(np.ceil(OUTER_FOLDS / 2)))
        final_feats = counts[counts >= majority_threshold].index.tolist()

        roc_aucs = [m["roc_auc"] for m in fold_metrics[mname]]
        pr_aucs = [m["pr_auc"] for m in fold_metrics[mname]]

        results[mname] = {
            "fold_selections": fold_selections[mname],
            "stability_nogueira": float(stability),
            "final_features_majority": final_feats,
            "performance": {
                "roc_auc_mean": float(np.mean(roc_aucs)),
                "roc_auc_std": float(np.std(roc_aucs)),
                "pr_auc_mean": float(np.mean(pr_aucs)),
                "pr_auc_std": float(np.std(pr_aucs)),
            },
        }

    # Save outputs
    out_json = RESULTS_DIR / f"04c_H{h}_embedded_nested.json"
    with open(out_json, "w") as f:
        json.dump({"horizon": h, "methods": results}, f, indent=2)
    logger.info(f"  ✓ Nested JSON: {out_json}")

    # Excel summary
    with pd.ExcelWriter(RESULTS_DIR / f"04c_H{h}_embedded_nested.xlsx", engine="openpyxl") as writer:
        rows = []
        for mname, data in results.items():
            perf = data["performance"]
            rows.append({
                "Method": mname,
                "Stability_Nogueira": data["stability_nogueira"],
                "Final_Features_Count": len(data["final_features_majority"]),
                "ROC_AUC_Mean": perf["roc_auc_mean"],
                "ROC_AUC_Std": perf["roc_auc_std"],
                "PR_AUC_Mean": perf["pr_auc_mean"],
                "PR_AUC_Std": perf["pr_auc_std"],
            })
        pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)

        for mname, data in results.items():
            # Fold selections sheet
            df_fs = pd.DataFrame({
                f"Fold_{i+1}": pd.Series(feats) for i, feats in enumerate(data["fold_selections"])
            })
            df_fs.to_excel(writer, sheet_name=f"{mname[:28]}_folds", index=False)

            # Final features
            pd.DataFrame({"Feature": data["final_features_majority"]}).to_excel(
                writer, sheet_name=f"{mname[:28]}_final", index=False
            )

    logger.info(f"  ✓ Nested Excel: {RESULTS_DIR / f'04c_H{h}_embedded_nested.xlsx'}")

    return results


def main():
    logger.info("=" * 80)
    logger.info("PHASE 04c (Nested): Embedded Methods with Nested CV")
    logger.info("=" * 80)

    df = pd.read_parquet(DATA_DIR / "poland_imputed.parquet")
    logger.info(f"Loaded {len(df)} observations")

    for h in HORIZONS:
        process_horizon_nested(h, df)

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 04c (Nested) COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
