"""
Phase 04a: Filter Methods for Feature Selection
================================================

Implements statistical filter methods for univariate feature selection:
1. Spearman rank correlation (non-parametric)
2. Mutual Information (non-linear dependencies)
3. ANOVA F-test (parametric baseline)

Uses nested cross-validation to determine optimal k features.

Input:
- VIF-cleaned feature sets: data/processed/feature_sets/H{1-5}_features.json
- Imputed dataset: data/processed/poland_imputed.parquet

Output:
- Excel: results/04_feature_selection/04a_H{1-5}_filter.xlsx
- HTML: results/04_feature_selection/04a_H{1-5}_filter.html
- JSON: Feature rankings per method
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =============================================================================
# Setup
# =============================================================================

# Configure logging
log_dir = Path("logs/04_feature_selection")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "04a_filter_methods.log"),
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

# =============================================================================
# Core Functions
# =============================================================================

def load_vif_features(horizon: int) -> List[str]:
    """Load VIF-cleaned features for a specific horizon."""
    feature_file = FEATURE_SETS_DIR / f"H{horizon}_features.json"
    with open(feature_file, "r") as f:
        features = json.load(f)
    logger.info(f"H{horizon}: Loaded {len(features)} VIF-cleaned features")
    return features


def compute_spearman_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute Spearman rank correlation for each feature.
    
    Returns DataFrame with columns: [feature, correlation, abs_correlation, p_value]
    """
    results = []
    for col in X.columns:
        rho, pval = spearmanr(X[col], y)
        results.append({
            "feature": col,
            "spearman_rho": rho,
            "spearman_abs": abs(rho),
            "spearman_pvalue": pval
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("spearman_abs", ascending=False).reset_index(drop=True)
    df["spearman_rank"] = np.arange(1, len(df) + 1)
    return df


def compute_mutual_info_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute Mutual Information scores for each feature.
    
    Returns DataFrame with columns: [feature, mi_score]
    """
    mi_scores = mutual_info_classif(
        X, y,
        n_neighbors=fs_config["mutual_info_n_neighbors"],
        random_state=fs_config["mutual_info_random_state"]
    )
    
    df = pd.DataFrame({
        "feature": X.columns,
        "mi_score": mi_scores
    })
    df = df.sort_values("mi_score", ascending=False).reset_index(drop=True)
    df["mi_rank"] = np.arange(1, len(df) + 1)
    return df


def compute_anova_f_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute ANOVA F-statistic for each feature.
    
    Returns DataFrame with columns: [feature, f_statistic, p_value]
    """
    f_stats, p_values = f_classif(X, y)
    
    df = pd.DataFrame({
        "feature": X.columns,
        "f_statistic": f_stats,
        "anova_pvalue": p_values
    })
    df = df.sort_values("f_statistic", ascending=False).reset_index(drop=True)
    df["anova_rank"] = np.arange(1, len(df) + 1)
    return df


def evaluate_top_k_features(
    X: pd.DataFrame,
    y: pd.Series,
    features_ranked: List[str],
    k: int,
    cv_folds: int = 5
) -> Dict[str, float]:
    """
    Evaluate top-k features using nested CV with logistic regression.
    
    Returns dict with mean ROC-AUC and PR-AUC.
    """
    X_k = X[features_ranked[:k]]
    
    # Pipeline: scaler + logistic regression  
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",  # LBFGS is optimal for L2
            max_iter=fs_config["max_iter"],
            tol=fs_config.get("tol", 0.001),
            class_weight=fs_config["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=1  # Pipeline will be parallelized at CV level
        ))
    ])
    
    # Stratified cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    roc_aucs = []
    pr_aucs = []
    
    for train_idx, val_idx in cv.split(X_k, y):
        X_train, X_val = X_k.iloc[train_idx], X_k.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Compute metrics
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)
    
    return {
        "k": k,
        "roc_auc_mean": np.mean(roc_aucs),
        "roc_auc_std": np.std(roc_aucs),
        "pr_auc_mean": np.mean(pr_aucs),
        "pr_auc_std": np.std(pr_aucs)
    }


def select_optimal_k_nested(
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    k_range: Tuple[int, int]
) -> Tuple[int, pd.DataFrame, List[str]]:
    """
    Select optimal k with PROPER nested CV (no leakage).
    
    For each k:
        For each CV fold:
            Compute rankings on TRAIN data only
            Select top-k from TRAIN rankings
            Evaluate on VAL data
        Average performance across folds
    Choose k with best average performance.
    
    Args:
        method: "spearman", "mutual_info", or "anova_f"
    
    Returns (optimal_k, cv_results_df, final_selected_features)
    """
    k_min, k_max = k_range
    k_values = list(range(k_min, min(k_max + 1, len(X.columns) + 1), 3))  # Step=3 for speed
    
    logger.info(f"  Evaluating k in range [{k_min}, {k_max}]: {k_values}")
    
    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    results = []
    for k in k_values:
        logger.info(f"    Testing k={k}...")
        
        fold_roc_aucs = []
        fold_pr_aucs = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # CRITICAL: Compute rankings on TRAIN data only (no leakage)
            if method == "spearman":
                rankings_train = compute_spearman_scores(X_train, y_train)
            elif method == "mutual_info":
                rankings_train = compute_mutual_info_scores(X_train, y_train)
            elif method == "anova_f":
                rankings_train = compute_anova_f_scores(X_train, y_train)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Select top-k from TRAIN rankings
            top_k_features = rankings_train["feature"].head(k).tolist()
            
            # Evaluate on VAL data
            X_train_k = X_train[top_k_features]
            X_val_k = X_val[top_k_features]
            
            # Pipeline with scaling
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2", 
                    solver="lbfgs",  # LBFGS is optimal for L2
                    max_iter=fs_config["max_iter"],
                    tol=fs_config.get("tol", 0.001),
                    class_weight=fs_config["class_weight"], 
                    random_state=RANDOM_STATE, 
                    n_jobs=1
                ))
            ])
            
            pipeline.fit(X_train_k, y_train)
            y_pred_proba = pipeline.predict_proba(X_val_k)[:, 1]
            
            fold_roc_aucs.append(roc_auc_score(y_val, y_pred_proba))
            fold_pr_aucs.append(average_precision_score(y_val, y_pred_proba))
        
        results.append({
            "k": k,
            "roc_auc_mean": np.mean(fold_roc_aucs),
            "roc_auc_std": np.std(fold_roc_aucs),
            "pr_auc_mean": np.mean(fold_pr_aucs),
            "pr_auc_std": np.std(fold_pr_aucs)
        })
    
    cv_results = pd.DataFrame(results)
    
    # Select k with highest ROC-AUC
    optimal_idx = cv_results["roc_auc_mean"].idxmax()
    optimal_k = int(cv_results.loc[optimal_idx, "k"])
    
    # Now compute final rankings on FULL data and select top-k
    if method == "spearman":
        final_rankings = compute_spearman_scores(X, y)
    elif method == "mutual_info":
        final_rankings = compute_mutual_info_scores(X, y)
    elif method == "anova_f":
        final_rankings = compute_anova_f_scores(X, y)
    
    final_selected = final_rankings["feature"].head(optimal_k).tolist()
    
    logger.info(f"  Optimal k={optimal_k} (ROC-AUC: {cv_results.loc[optimal_idx, 'roc_auc_mean']:.4f})")
    
    return optimal_k, cv_results, final_selected


def process_horizon_filter_methods(horizon: int, df: pd.DataFrame) -> Dict:
    """
    Process filter methods for a single horizon.
    
    Returns dict with all results for this horizon.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing Horizon {horizon}")
    logger.info(f"{'='*80}")
    
    # Filter data by horizon
    df_h = df[df["horizon"] == horizon].copy()
    logger.info(f"H{horizon}: {len(df_h)} observations")
    
    # Load VIF features
    vif_features = load_vif_features(horizon)
    
    # Prepare X, y
    X = df_h[vif_features]
    y = df_h["bankrupt"]
    
    logger.info(f"H{horizon}: Class distribution - 0: {(y==0).sum()}, 1: {(y==1).sum()} ({y.mean()*100:.2f}% bankruptcy)")
    
    # ==========================================================================
    # 1. Spearman Rank Correlation (FIXED: No leakage)
    # ==========================================================================
    logger.info("\n[1/3] Spearman - NESTED CV (rankings per fold)...")
    spearman_k, spearman_cv, spearman_selected = select_optimal_k_nested(
        X, y, "spearman", (TARGET_MIN, TARGET_MAX)
    )
    # Also compute full rankings for reporting
    spearman_df = compute_spearman_scores(X, y)
    
    # ==========================================================================
    # 2. Mutual Information (FIXED: No leakage)
    # ==========================================================================
    logger.info("\n[2/3] Mutual Info - NESTED CV (rankings per fold)...")
    mi_k, mi_cv, mi_selected = select_optimal_k_nested(
        X, y, "mutual_info", (TARGET_MIN, TARGET_MAX)
    )
    # Also compute full rankings for reporting
    mi_df = compute_mutual_info_scores(X, y)
    
    # ==========================================================================
    # 3. ANOVA F-Test (FIXED: No leakage)
    # ==========================================================================
    if fs_config["use_anova_f"]:
        logger.info("\n[3/3] ANOVA F - NESTED CV (rankings per fold)...")
        anova_k, anova_cv, anova_selected = select_optimal_k_nested(
            X, y, "anova_f", (TARGET_MIN, TARGET_MAX)
        )
        # Also compute full rankings for reporting
        anova_df = compute_anova_f_scores(X, y)
    else:
        anova_df = None
        anova_cv = None
        anova_k = None
        anova_selected = []
    
    # ==========================================================================
    # Merge Rankings
    # ==========================================================================
    logger.info("\nMerging rankings from all methods...")
    merged = spearman_df[["feature", "spearman_rho", "spearman_abs", "spearman_rank"]]
    merged = merged.merge(mi_df[["feature", "mi_score", "mi_rank"]], on="feature")
    
    if anova_df is not None:
        merged = merged.merge(anova_df[["feature", "f_statistic", "anova_rank"]], on="feature")
    
    # Compute average rank
    rank_cols = ["spearman_rank", "mi_rank"]
    if anova_df is not None:
        rank_cols.append("anova_rank")
    
    merged["avg_rank"] = merged[rank_cols].mean(axis=1)
    merged = merged.sort_values("avg_rank").reset_index(drop=True)
    
    # ==========================================================================
    # Package Results
    # ==========================================================================
    results = {
        "horizon": horizon,
        "n_obs": len(df_h),
        "n_features": len(vif_features),
        "bankruptcy_rate": y.mean(),
        "spearman": {
            "rankings": spearman_df,
            "cv_results": spearman_cv,
            "optimal_k": spearman_k,
            "selected_features": spearman_selected
        },
        "mutual_info": {
            "rankings": mi_df,
            "cv_results": mi_cv,
            "optimal_k": mi_k,
            "selected_features": mi_selected
        },
        "anova": {
            "rankings": anova_df,
            "cv_results": anova_cv,
            "optimal_k": anova_k,
            "selected_features": anova_selected
        } if anova_df is not None else None,
        "merged_rankings": merged
    }
    
    logger.info(f"\nH{horizon} Summary:")
    logger.info(f"  - Spearman: {spearman_k} features selected")
    logger.info(f"  - Mutual Info: {mi_k} features selected")
    if anova_k:
        logger.info(f"  - ANOVA F: {anova_k} features selected")
    
    return results


def save_horizon_results(horizon: int, results: Dict):
    """Save per-horizon results to Excel, HTML, and JSON."""
    logger.info(f"\nSaving H{horizon} results...")
    
    # Excel output
    excel_path = RESULTS_DIR / f"04a_H{horizon}_filter.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Sheet 1: Merged rankings
        results["merged_rankings"].to_excel(writer, sheet_name="Merged_Rankings", index=False)
        
        # Sheet 2: Spearman
        results["spearman"]["rankings"].to_excel(writer, sheet_name="Spearman_Rankings", index=False)
        results["spearman"]["cv_results"].to_excel(writer, sheet_name="Spearman_CV", index=False)
        
        # Sheet 3: Mutual Info
        results["mutual_info"]["rankings"].to_excel(writer, sheet_name="MI_Rankings", index=False)
        results["mutual_info"]["cv_results"].to_excel(writer, sheet_name="MI_CV", index=False)
        
        # Sheet 4: ANOVA (if available)
        if results["anova"] is not None:
            results["anova"]["rankings"].to_excel(writer, sheet_name="ANOVA_Rankings", index=False)
            results["anova"]["cv_results"].to_excel(writer, sheet_name="ANOVA_CV", index=False)
        
        # Sheet 5: Selected Features Summary
        summary = pd.DataFrame({
            "Method": ["Spearman", "Mutual_Info", "ANOVA_F"] if results["anova"] else ["Spearman", "Mutual_Info"],
            "Optimal_K": [
                results["spearman"]["optimal_k"],
                results["mutual_info"]["optimal_k"],
                results["anova"]["optimal_k"] if results["anova"] else None
            ],
            "Features": [
                ", ".join(results["spearman"]["selected_features"][:10]) + "...",
                ", ".join(results["mutual_info"]["selected_features"][:10]) + "...",
                ", ".join(results["anova"]["selected_features"][:10]) + "..." if results["anova"] else None
            ]
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)
    
    logger.info(f"  ✓ Excel: {excel_path}")
    
    # JSON output (for downstream scripts)
    json_data = {
        "horizon": horizon,
        "spearman_selected": results["spearman"]["selected_features"],
        "mi_selected": results["mutual_info"]["selected_features"],
        "anova_selected": results["anova"]["selected_features"] if results["anova"] else []
    }
    json_path = RESULTS_DIR / f"04a_H{horizon}_filter_selected.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"  ✓ JSON: {json_path}")
    
    # HTML report
    html_path = RESULTS_DIR / f"04a_H{horizon}_filter.html"
    generate_html_report(horizon, results, html_path)
    logger.info(f"  ✓ HTML: {html_path}")


def generate_html_report(horizon: int, results: Dict, output_path: Path):
    """Generate professional HTML report for a horizon."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Phase 04a: Filter Methods - Horizon {horizon}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .metric {{
                display: inline-block;
                margin: 10px 20px 10px 0;
                padding: 10px 15px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }}
            .metric-label {{
                font-weight: bold;
                color: #7f8c8d;
            }}
            .metric-value {{
                font-size: 1.2em;
                color: #2c3e50;
            }}
            .section {{
                margin-top: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Phase 04a: Filter Methods - Horizon {horizon}</h1>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="metric">
                    <span class="metric-label">Observations:</span>
                    <span class="metric-value">{results['n_obs']:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">VIF Features:</span>
                    <span class="metric-value">{results['n_features']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Bankruptcy Rate:</span>
                    <span class="metric-value">{results['bankruptcy_rate']*100:.2f}%</span>
                </div>
            </div>
            
            <div class="section">
                <h2>Method Comparison</h2>
                <table>
                    <tr>
                        <th>Method</th>
                        <th>Optimal K</th>
                        <th>ROC-AUC (mean ± std)</th>
                        <th>PR-AUC (mean ± std)</th>
                    </tr>
    """
    
    # Add Spearman row
    sp_cv = results["spearman"]["cv_results"]
    sp_best = sp_cv.loc[sp_cv["roc_auc_mean"].idxmax()]
    html_content += f"""
                    <tr>
                        <td><strong>Spearman Rank Correlation</strong></td>
                        <td>{results['spearman']['optimal_k']}</td>
                        <td>{sp_best['roc_auc_mean']:.4f} ± {sp_best['roc_auc_std']:.4f}</td>
                        <td>{sp_best['pr_auc_mean']:.4f} ± {sp_best['pr_auc_std']:.4f}</td>
                    </tr>
    """
    
    # Add MI row
    mi_cv = results["mutual_info"]["cv_results"]
    mi_best = mi_cv.loc[mi_cv["roc_auc_mean"].idxmax()]
    html_content += f"""
                    <tr>
                        <td><strong>Mutual Information</strong></td>
                        <td>{results['mutual_info']['optimal_k']}</td>
                        <td>{mi_best['roc_auc_mean']:.4f} ± {mi_best['roc_auc_std']:.4f}</td>
                        <td>{mi_best['pr_auc_mean']:.4f} ± {mi_best['pr_auc_std']:.4f}</td>
                    </tr>
    """
    
    # Add ANOVA row if available
    if results["anova"] is not None:
        an_cv = results["anova"]["cv_results"]
        an_best = an_cv.loc[an_cv["roc_auc_mean"].idxmax()]
        html_content += f"""
                    <tr>
                        <td><strong>ANOVA F-Test</strong></td>
                        <td>{results['anova']['optimal_k']}</td>
                        <td>{an_best['roc_auc_mean']:.4f} ± {an_best['roc_auc_std']:.4f}</td>
                        <td>{an_best['pr_auc_mean']:.4f} ± {an_best['pr_auc_std']:.4f}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Top 10 Features (by Average Rank)</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Spearman |ρ|</th>
                        <th>MI Score</th>
    """
    
    if results["anova"] is not None:
        html_content += "<th>F-Statistic</th>"
    
    html_content += """
                        <th>Avg Rank</th>
                    </tr>
    """
    
    # Add top 10 features
    top10 = results["merged_rankings"].head(10)
    for idx, row in top10.iterrows():
        html_content += f"""
                    <tr>
                        <td>{idx+1}</td>
                        <td><strong>{row['feature']}</strong></td>
                        <td>{row['spearman_abs']:.4f}</td>
                        <td>{row['mi_score']:.4f}</td>
        """
        if results["anova"] is not None:
            html_content += f"<td>{row['f_statistic']:.2f}</td>"
        html_content += f"""
                        <td>{row['avg_rank']:.1f}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Interpretation</h2>
                <p><strong>Spearman Rank Correlation:</strong> Measures monotonic relationship between feature and target. Non-parametric, robust to outliers.</p>
                <p><strong>Mutual Information:</strong> Captures non-linear dependencies. Higher MI = more predictive power.</p>
    """
    
    if results["anova"] is not None:
        html_content += "<p><strong>ANOVA F-Test:</strong> Parametric test assuming normality. Serves as baseline comparison.</p>"
    
    html_content += f"""
                <p><strong>Optimal k selection:</strong> Determined via {INNER_FOLDS}-fold stratified CV using ROC-AUC as criterion.</p>
            </div>
            
            <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 0.9em;">
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Project: Bankruptcy Prediction Seminar (FH Wedel) - Phase 04a: Filter Methods</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("PHASE 04a: FILTER METHODS FOR FEATURE SELECTION")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  - Target features: {TARGET_MIN}-{TARGET_MAX}")
    logger.info(f"  - Outer folds: {OUTER_FOLDS}")
    logger.info(f"  - Inner folds: {INNER_FOLDS}")
    logger.info(f"  - Random state: {RANDOM_STATE}")
    logger.info(f"  - Methods: Spearman, MI" + (", ANOVA F" if fs_config["use_anova_f"] else ""))
    
    # Load imputed dataset
    logger.info(f"\nLoading dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"  ✓ Loaded {len(df):,} observations")
    
    # Process each horizon
    all_results = {}
    for horizon in HORIZONS:
        results = process_horizon_filter_methods(horizon, df)
        save_horizon_results(horizon, results)
        all_results[f"H{horizon}"] = results
    
    # Generate consolidated summary
    logger.info("\n" + "="*80)
    logger.info("GENERATING CONSOLIDATED SUMMARY")
    logger.info("="*80)
    
    summary_rows = []
    for horizon in HORIZONS:
        res = all_results[f"H{horizon}"]
        summary_rows.append({
            "Horizon": f"H{horizon}",
            "N_Obs": res["n_obs"],
            "VIF_Features": res["n_features"],
            "Bankruptcy_Rate": f"{res['bankruptcy_rate']*100:.2f}%",
            "Spearman_K": res["spearman"]["optimal_k"],
            "MI_K": res["mutual_info"]["optimal_k"],
            "ANOVA_K": res["anova"]["optimal_k"] if res["anova"] else None
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save consolidated summary
    summary_path = RESULTS_DIR / "04a_ALL_filter_summary.xlsx"
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    logger.info(f"\n✓ Consolidated summary: {summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 04a COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info(f"Total horizons processed: {len(HORIZONS)}")
    logger.info("\nNext step: Run 04b_wrapper_methods.py")


if __name__ == "__main__":
    main()
