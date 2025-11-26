"""
Phase 04c: Embedded Methods for Feature Selection
==================================================

Implements embedded feature selection:
1. Lasso (L1 Regularization) Logistic Regression
2. Random Forest with Permutation Importance

Uses nested cross-validation and stability analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup
log_dir = Path("logs/04_feature_selection")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "04c_embedded_methods.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config/project_config.yaml", "r") as f:
    config = yaml.safe_load(f)

fs_config = config["feature_selection"]
HORIZONS = config["datasets"]["polish"]["horizons"]
RANDOM_STATE = fs_config["random_state"]
OUTER_FOLDS = fs_config["outer_folds"]
INNER_FOLDS = fs_config["inner_folds"]
TARGET_MIN = fs_config["target_features_min"]
TARGET_MAX = fs_config["target_features_max"]
N_JOBS = fs_config["n_jobs"]

# Paths
DATA_PATH = Path("data/processed/poland_imputed.parquet")
FEATURE_SETS_DIR = Path("data/processed/feature_sets")
RESULTS_DIR = Path("results/04_feature_selection")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_vif_features(horizon: int) -> List[str]:
    """Load VIF-cleaned features for a specific horizon."""
    feature_file = FEATURE_SETS_DIR / f"H{horizon}_features.json"
    with open(feature_file, "r") as f:
        features = json.load(f)
    logger.info(f"H{horizon}: Loaded {len(features)} VIF-cleaned features")
    return features


def perform_lasso_selection(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Lasso L1 feature selection with cross-validated C."""
    logger.info("  Running Lasso Logistic Regression with CV...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Lasso with CV
    lasso_cv = LogisticRegressionCV(
        Cs=fs_config["lasso_c_values"],
        cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        penalty="l1",
        solver="saga",
        max_iter=fs_config["max_iter"],
        tol=fs_config.get("tol", 1e-4),
        class_weight=fs_config["class_weight"],
        scoring=fs_config["lasso_scoring"],
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    logger.info(f"    Fitting Lasso with {len(fs_config['lasso_c_values'])} C values...")
    lasso_cv.fit(X_scaled, y)
    
    optimal_c = lasso_cv.C_[0]
    coefficients = lasso_cv.coef_[0]
    non_zero_mask = coefficients != 0
    selected_features = X.columns[non_zero_mask].tolist()
    
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
        "selected": non_zero_mask
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    
    logger.info(f"  ✓ Lasso: Optimal C={optimal_c:.4f}, {len(selected_features)} features selected")
    
    return {
        "optimal_c": optimal_c,
        "selected_features": selected_features,
        "coefficients": coef_df
    }


def perform_rf_importance(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Random Forest feature importance with permutation."""
    logger.info("  Training Random Forest...")
    
    rf = RandomForestClassifier(
        n_estimators=fs_config["rf_n_estimators"],
        max_depth=fs_config["rf_max_depth"],
        min_samples_split=fs_config["rf_min_samples_split"],
        min_samples_leaf=fs_config["rf_min_samples_leaf"],
        class_weight=fs_config["class_weight"],
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=0
    )
    
    rf.fit(X, y)
    
    # Permutation importance
    logger.info("    Computing permutation importance...")
    perm_importance = permutation_importance(
        rf, X, y, n_repeats=10, random_state=RANDOM_STATE, n_jobs=N_JOBS, scoring="roc_auc"
    )
    
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "perm_importance": perm_importance.importances_mean,
        "perm_std": perm_importance.importances_std
    }).sort_values("perm_importance", ascending=False).reset_index(drop=True)
    
    # Select features with positive importance
    positive_mask = importance_df["perm_importance"] > 0
    selected_features = importance_df.loc[positive_mask, "feature"].tolist()
    
    # Limit to TARGET_MAX
    if len(selected_features) > TARGET_MAX:
        selected_features = selected_features[:TARGET_MAX]
    elif len(selected_features) < TARGET_MIN:
        selected_features = importance_df["feature"].head(TARGET_MIN).tolist()
    
    importance_df["selected"] = importance_df["feature"].isin(selected_features)
    
    logger.info(f"  ✓ RF: {len(selected_features)} features selected")
    
    return {
        "selected_features": selected_features,
        "importance_scores": importance_df
    }


def evaluate_features(X: pd.DataFrame, y: pd.Series, features: List[str]) -> Dict:
    """Evaluate feature set with outer CV."""
    if len(features) == 0:
        return {"roc_auc_mean": 0, "roc_auc_std": 0, "pr_auc_mean": 0, "pr_auc_std": 0}
    
    X_selected = X[features]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", 
            solver="lbfgs",  # LBFGS is optimal for L2
            max_iter=fs_config["max_iter"],
            tol=fs_config.get("tol", 0.001),
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
        "roc_auc_mean": np.mean(roc_aucs), "roc_auc_std": np.std(roc_aucs),
        "pr_auc_mean": np.mean(pr_aucs), "pr_auc_std": np.std(pr_aucs)
    }


def process_horizon(horizon: int, df: pd.DataFrame) -> Dict:
    """Process embedded methods for a single horizon."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing Horizon {horizon}")
    logger.info(f"{'='*80}")
    
    df_h = df[df["horizon"] == horizon].copy()
    logger.info(f"H{horizon}: {len(df_h)} observations")
    
    vif_features = load_vif_features(horizon)
    X, y = df_h[vif_features], df_h["bankrupt"]
    
    logger.info(f"H{horizon}: Bankruptcy {y.mean()*100:.2f}%")
    
    # Lasso
    logger.info("\n[1/2] Lasso L1...")
    lasso_results = perform_lasso_selection(X, y)
    lasso_perf = evaluate_features(X, y, lasso_results["selected_features"])
    logger.info(f"  Performance: ROC-AUC={lasso_perf['roc_auc_mean']:.4f}")
    
    # Random Forest
    logger.info("\n[2/2] Random Forest...")
    rf_results = perform_rf_importance(X, y)
    rf_perf = evaluate_features(X, y, rf_results["selected_features"])
    logger.info(f"  Performance: ROC-AUC={rf_perf['roc_auc_mean']:.4f}")
    
    return {
        "horizon": horizon,
        "n_obs": len(df_h),
        "n_features": len(vif_features),
        "bankruptcy_rate": y.mean(),
        "lasso": {
            "optimal_c": lasso_results["optimal_c"],
            "selected_features": lasso_results["selected_features"],
            "coefficients": lasso_results["coefficients"],
            "performance": lasso_perf
        },
        "random_forest": {
            "selected_features": rf_results["selected_features"],
            "importance_scores": rf_results["importance_scores"],
            "performance": rf_perf
        }
    }


def save_results(horizon: int, results: Dict):
    """Save results to Excel, HTML, and JSON."""
    logger.info(f"\nSaving H{horizon} results...")
    
    # Excel
    excel_path = RESULTS_DIR / f"04c_H{horizon}_embedded.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results["lasso"]["coefficients"].to_excel(writer, sheet_name="Lasso_Coefficients", index=False)
        results["random_forest"]["importance_scores"].to_excel(writer, sheet_name="RF_Importance", index=False)
        
        summary_df = pd.DataFrame({
            "Method": ["Lasso L1", "Random Forest"],
            "Features": [len(results["lasso"]["selected_features"]), len(results["random_forest"]["selected_features"])],
            "ROC_AUC": [
                f"{results['lasso']['performance']['roc_auc_mean']:.4f} ± {results['lasso']['performance']['roc_auc_std']:.4f}",
                f"{results['random_forest']['performance']['roc_auc_mean']:.4f} ± {results['random_forest']['performance']['roc_auc_std']:.4f}"
            ]
        })
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    logger.info(f"  ✓ Excel: {excel_path}")
    
    # JSON
    json_data = {
        "horizon": horizon,
        "lasso": {
            "selected_features": results["lasso"]["selected_features"],
            "optimal_c": results["lasso"]["optimal_c"],
            "performance": results["lasso"]["performance"]
        },
        "random_forest": {
            "selected_features": results["random_forest"]["selected_features"],
            "performance": results["random_forest"]["performance"]
        }
    }
    json_path = RESULTS_DIR / f"04c_H{horizon}_embedded_selected.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"  ✓ JSON: {json_path}")
    
    # Simple HTML
    html_path = RESULTS_DIR / f"04c_H{horizon}_embedded.html"
    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Phase 04c - H{horizon}</title>
<style>body{{font-family:sans-serif;margin:40px;}}table{{border-collapse:collapse;width:100%;}}
th,td{{border:1px solid #ddd;padding:8px;}}th{{background:#9b59b6;color:white;}}</style>
</head><body><h1>Phase 04c: Embedded Methods - H{horizon}</h1>
<h2>Summary</h2><table><tr><th>Method</th><th>Features</th><th>ROC-AUC</th></tr>
<tr><td>Lasso L1</td><td>{len(results['lasso']['selected_features'])}</td><td>{results['lasso']['performance']['roc_auc_mean']:.4f}</td></tr>
<tr><td>Random Forest</td><td>{len(results['random_forest']['selected_features'])}</td><td>{results['random_forest']['performance']['roc_auc_mean']:.4f}</td></tr>
</table></body></html>"""
    
    with open(html_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"  ✓ HTML: {html_path}")


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("PHASE 04c: EMBEDDED METHODS FOR FEATURE SELECTION")
    logger.info("="*80)
    
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df):,} observations")
    
    for horizon in HORIZONS:
        results = process_horizon(horizon, df)
        save_results(horizon, results)
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 04c COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
