#!/usr/bin/env python3
"""
Phase 04c: EMBEDDED METHODS - PERFECT IMPLEMENTATION
=====================================================

RESEARCH-BACKED OPTIMAL IMPLEMENTATION:
- Lasso (L1) - As per prompt
- Elastic Net (L1+L2) - Research-backed optimal for bankruptcy prediction
- Ridge (L2) - Baseline comparison

OPTIMIZED FOR: M1 Pro 8-core (6 perf + 2 efficiency), 16 GB RAM

References:
- GeeksforGeeks: "Elastic Net works well with many correlated features"
- scikit-learn 1.7.2 documentation
- Bankruptcy prediction literature (2024)
"""

import logging
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_SETS_DIR = DATA_DIR / "feature_sets"
RESULTS_DIR = PROJECT_ROOT / "results" / "04_feature_selection"
LOGS_DIR = PROJECT_ROOT / "logs" / "04_feature_selection"
CONFIG_PATH = PROJECT_ROOT / "config" / "project_config.yaml"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "04c_embedded_methods_PERFECT.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

fs_config = config["feature_selection"]
HORIZONS = [1, 2, 3, 4, 5]
RANDOM_STATE = config["analysis"]["random_state"]
OUTER_FOLDS = fs_config["outer_folds"]


# =============================================================================
# REGULARIZATION METHODS
# =============================================================================

def perform_lasso_selection(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Lasso (L1) Regularization - LIBLINEAR solver (optimal for small datasets).
    
    Logistic Regression with L1 penalty, cross-validated C selection.
    Uses LIBLINEAR solver which converges in ~200-500 iterations for small data.
    """
    logger.info("  [1/4] Lasso (L1 Regularization) - LIBLINEAR solver...")
    
    lasso_cfg = fs_config["lasso"]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Lasso with CV
    logger.info(f"    Solver: {lasso_cfg['solver']} (optimal for small datasets <10k)")
    logger.info(f"    Testing {len(lasso_cfg['c_values'])} C values with {fs_config['embedded_cv_folds']}-fold CV...")
    logger.info(f"    Max iterations: {lasso_cfg['max_iter']} (LIBLINEAR converges fast)")
    
    lasso_cv = LogisticRegressionCV(
        Cs=lasso_cfg["c_values"],
        cv=StratifiedKFold(n_splits=fs_config["embedded_cv_folds"], shuffle=True, random_state=RANDOM_STATE),
        penalty=lasso_cfg["penalty"],
        solver=lasso_cfg["solver"],
        dual=lasso_cfg.get("dual", False),
        max_iter=lasso_cfg["max_iter"],
        tol=lasso_cfg["tol"],
        class_weight=lasso_cfg["class_weight"],
        scoring=fs_config["embedded_scoring"],
        n_jobs=lasso_cfg.get("n_jobs", 1),
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    logger.info("    Fitting... (progress below)")
    lasso_cv.fit(X_scaled, y)
    
    # Extract non-zero features
    coefs = lasso_cv.coef_[0]
    non_zero_mask = coefs != 0
    selected_features = X.columns[non_zero_mask].tolist()
    
    logger.info(f"    ✓ Lasso: {len(selected_features)} features selected (C={lasso_cv.C_[0]:.4f})")
    
    return {
        "method": "Lasso_L1",
        "selected_features": selected_features,
        "n_features": len(selected_features),
        "optimal_C": float(lasso_cv.C_[0]),
        "coefficients": {feat: float(coef) for feat, coef in zip(X.columns, coefs)}
    }


def perform_elastic_net_selection(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Elastic Net (L1+L2) Regularization - RESEARCH-BACKED OPTIMAL.
    
    Combines L1 (sparsity) and L2 (stability for correlated features).
    Superior for bankruptcy prediction with correlated financial ratios.
    Uses SAGA solver (only option for elastic net).
    """
    logger.info("  [2/4] Elastic Net (L1+L2 Regularization) - SAGA solver...")
    
    en_cfg = fs_config["elastic_net"]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Elastic Net with CV (grid search over C and l1_ratio)
    logger.info(f"    Solver: {en_cfg['solver']} (only option for elastic net)")
    logger.info(f"    Testing {len(en_cfg['c_values'])} C × {len(en_cfg['l1_ratio'])} l1_ratio = {len(en_cfg['c_values']) * len(en_cfg['l1_ratio'])} combinations...")
    logger.info(f"    Max iterations: {en_cfg['max_iter']} (may have some warnings on imbalanced data - OK)")
    
    en_cv = LogisticRegressionCV(
        Cs=en_cfg["c_values"],
        cv=StratifiedKFold(n_splits=fs_config["embedded_cv_folds"], shuffle=True, random_state=RANDOM_STATE),
        penalty=en_cfg["penalty"],
        solver=en_cfg["solver"],
        l1_ratios=en_cfg["l1_ratio"],  # Grid search L1/L2 balance
        max_iter=en_cfg["max_iter"],
        tol=en_cfg["tol"],
        class_weight=en_cfg["class_weight"],
        scoring=fs_config["embedded_scoring"],
        n_jobs=en_cfg["n_jobs"],
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    logger.info("    Fitting... (progress below, may take a few minutes)")
    en_cv.fit(X_scaled, y)
    
    # Extract selected features
    coefs = en_cv.coef_[0]
    non_zero_mask = coefs != 0
    selected_features = X.columns[non_zero_mask].tolist()
    
    logger.info(f"    ✓ Elastic Net: {len(selected_features)} features (C={en_cv.C_[0]:.4f}, l1_ratio={en_cv.l1_ratio_[0]:.2f})")
    
    return {
        "method": "Elastic_Net",
        "selected_features": selected_features,
        "n_features": len(selected_features),
        "optimal_C": float(en_cv.C_[0]),
        "optimal_l1_ratio": float(en_cv.l1_ratio_[0]),
        "coefficients": {feat: float(coef) for feat, coef in zip(X.columns, coefs)}
    }


def perform_ridge_selection(X: pd.DataFrame, y: pd.Series, top_k: int = 30) -> Dict:
    """
    Ridge (L2) Regularization - Baseline comparison.
    
    L2 doesn't set coefficients to zero, so we select top-k by absolute magnitude.
    Uses LBFGS solver (optimal for L2, fast convergence).
    """
    logger.info("  [3/4] Ridge (L2 Regularization) - LBFGS solver...")
    
    ridge_cfg = fs_config["ridge"]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Ridge with CV
    logger.info(f"    Solver: {ridge_cfg['solver']} (optimal for L2)")
    logger.info(f"    Testing {len(ridge_cfg['c_values'])} C values...")
    logger.info(f"    Max iterations: {ridge_cfg['max_iter']} (fast convergence expected)")
    
    ridge_cv = LogisticRegressionCV(
        Cs=ridge_cfg["c_values"],
        cv=StratifiedKFold(n_splits=fs_config["embedded_cv_folds"], shuffle=True, random_state=RANDOM_STATE),
        penalty=ridge_cfg["penalty"],
        solver=ridge_cfg["solver"],
        max_iter=ridge_cfg["max_iter"],
        tol=ridge_cfg["tol"],
        class_weight=ridge_cfg["class_weight"],
        scoring=fs_config["embedded_scoring"],
        n_jobs=ridge_cfg["n_jobs"],
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    logger.info("    Fitting...")
    ridge_cv.fit(X_scaled, y)
    
    # Ridge doesn't zero out coefficients, so select top-k by magnitude
    coefs = ridge_cv.coef_[0]
    abs_coefs = np.abs(coefs)
    top_k_indices = np.argsort(abs_coefs)[-top_k:]
    selected_features = X.columns[top_k_indices].tolist()
    
    logger.info(f"    ✓ Ridge: Top {len(selected_features)} features by |coefficient| (C={ridge_cv.C_[0]:.4f})")
    
    return {
        "method": "Ridge_L2",
        "selected_features": selected_features,
        "n_features": len(selected_features),
        "optimal_C": float(ridge_cv.C_[0]),
        "coefficients": {feat: float(coef) for feat, coef in zip(X.columns, coefs)}
    }


def perform_random_forest_selection(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Random Forest with Permutation Importance.
    
    Uses permutation importance (more reliable than impurity-based).
    """
    logger.info("  [4/4] Random Forest + Permutation Importance...")
    
    rf_cfg = fs_config["random_forest"]
    
    # Random Forest
    logger.info(f"    Training RF with {rf_cfg['n_estimators']} trees, max_depth={rf_cfg['max_depth']}...")
    
    rf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        max_features=rf_cfg["max_features"],
        class_weight=rf_cfg["class_weight"],
        n_jobs=rf_cfg["n_jobs"],
        random_state=rf_cfg["random_state"],
        verbose=rf_cfg["verbose"]
    )
    
    rf.fit(X, y)
    
    # Permutation importance
    logger.info("    Computing permutation importance...")
    perm_importance = permutation_importance(
        rf, X, y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=rf_cfg["n_jobs"],
        scoring="roc_auc"
    )
    
    # Select features with positive importance
    positive_mask = perm_importance.importances_mean > 0
    selected_features = X.columns[positive_mask].tolist()
    
    logger.info(f"    ✓ Random Forest: {len(selected_features)} features with positive importance")
    
    # Feature importance ranking
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    }).sort_values("importance_mean", ascending=False)
    
    return {
        "method": "Random_Forest",
        "selected_features": selected_features,
        "n_features": len(selected_features),
        "importance_ranking": importance_df.to_dict("records")
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_features(X: pd.DataFrame, y: pd.Series, features: List[str]) -> Dict:
    """Evaluate feature set with outer CV."""
    if len(features) == 0:
        return {"roc_auc_mean": 0, "roc_auc_std": 0, "pr_auc_mean": 0, "pr_auc_std": 0}
    
    X_selected = X[features]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=50000,
            tol=0.0001,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])
    
    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    roc_aucs, pr_aucs = [], []
    
    for train_idx, val_idx in cv.split(X_selected, y):
        X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        
        roc_aucs.append(roc_auc_score(y_val, y_pred_proba))
        pr_aucs.append(average_precision_score(y_val, y_pred_proba))
    
    return {
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs))
    }


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_horizon(horizon: int, df: pd.DataFrame) -> Dict:
    """Process all embedded methods for a single horizon."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing Horizon {horizon}")
    logger.info(f"{'='*80}")
    
    # Load VIF features
    feature_file = FEATURE_SETS_DIR / f"H{horizon}_features.json"
    with open(feature_file, "r") as f:
        vif_features = json.load(f)
    
    df_h = df[df["horizon"] == horizon].copy()
    X = df_h[vif_features]
    y = df_h["bankrupt"]
    
    logger.info(f"H{horizon}: {len(df_h)} observations, {len(vif_features)} VIF features")
    logger.info(f"H{horizon}: Bankruptcy rate: {y.mean()*100:.2f}%")
    
    # Run all methods
    results = {}
    
    # 1. Lasso
    results["lasso"] = perform_lasso_selection(X, y)
    results["lasso"]["performance"] = evaluate_features(X, y, results["lasso"]["selected_features"])
    
    # 2. Elastic Net
    results["elastic_net"] = perform_elastic_net_selection(X, y)
    results["elastic_net"]["performance"] = evaluate_features(X, y, results["elastic_net"]["selected_features"])
    
    # 3. Ridge
    results["ridge"] = perform_ridge_selection(X, y, top_k=30)
    results["ridge"]["performance"] = evaluate_features(X, y, results["ridge"]["selected_features"])
    
    # 4. Random Forest
    results["random_forest"] = perform_random_forest_selection(X, y)
    results["random_forest"]["performance"] = evaluate_features(X, y, results["random_forest"]["selected_features"])
    
    # Summary
    logger.info(f"\nH{horizon} Summary:")
    for method_name, method_data in results.items():
        perf = method_data["performance"]
        logger.info(f"  {method_name}: {method_data['n_features']} features, ROC-AUC={perf['roc_auc_mean']:.4f}")
    
    # Save results
    save_horizon_results(horizon, results)
    
    return results


def save_horizon_results(horizon: int, results: Dict):
    """Save horizon results to Excel, JSON, and HTML."""
    logger.info(f"\nSaving H{horizon} results...")
    
    # JSON output
    json_data = {
        "horizon": int(horizon),
        "methods": {
            method: {
                "selected_features": data["selected_features"],
                "n_features": int(data["n_features"]),
                "performance": data["performance"]
            }
            for method, data in results.items()
        }
    }
    
    json_path = RESULTS_DIR / f"04c_H{horizon}_embedded_selected.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"  ✓ JSON: {json_path}")
    
    # Excel with comparison
    with pd.ExcelWriter(RESULTS_DIR / f"04c_H{horizon}_embedded.xlsx", engine="openpyxl") as writer:
        # Summary sheet
        summary_rows = []
        for method, data in results.items():
            perf = data["performance"]
            summary_rows.append({
                "Method": method,
                "N_Features": data["n_features"],
                "ROC_AUC_Mean": f"{perf['roc_auc_mean']:.4f}",
                "ROC_AUC_Std": f"{perf['roc_auc_std']:.4f}",
                "PR_AUC_Mean": f"{perf['pr_auc_mean']:.4f}",
                "PR_AUC_Std": f"{perf['pr_auc_std']:.4f}"
            })
        
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        
        # Per-method sheets
        for method, data in results.items():
            features_df = pd.DataFrame({
                "Feature": data["selected_features"],
                "Selected": "Yes"
            })
            features_df.to_excel(writer, sheet_name=method[:30], index=False)
    
    logger.info(f"  ✓ Excel: {RESULTS_DIR / f'04c_H{horizon}_embedded.xlsx'}")


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("PHASE 04c: EMBEDDED METHODS - PERFECT IMPLEMENTATION")
    logger.info("="*80)
    logger.info("Methods: Lasso (L1) + Elastic Net (L1+L2) + Ridge (L2) + Random Forest")
    logger.info(f"Optimized for: M1 Pro 8-core, 16 GB RAM")
    logger.info("")
    
    # Load data
    df = pd.read_parquet(DATA_DIR / "poland_imputed.parquet")
    logger.info(f"Loaded {len(df)} observations\n")
    
    # Process all horizons
    all_results = {}
    for horizon in HORIZONS:
        all_results[f"H{horizon}"] = process_horizon(horizon, df)
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 04c COMPLETE - PERFECT IMPLEMENTATION")
    logger.info("="*80)
    logger.info("All methods completed successfully!")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("\nNext: Run 04d_stability_consensus.py for final feature selection")


if __name__ == "__main__":
    main()
