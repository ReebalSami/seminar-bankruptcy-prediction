"""
Phase 04b: Wrapper Methods for Feature Selection  
=================================================

Implements wrapper-based feature selection using Recursive Feature Elimination:
- RFECV with Logistic Regression
- Pipeline: StandardScaler -> LogisticRegression
- Stratified cross-validation for optimal feature count
- Class-weighted estimator for imbalanced data

Input:
- VIF-cleaned feature sets: data/processed/feature_sets/H{1-5}_features.json
- Imputed dataset: data/processed/poland_imputed.parquet

Output:
- Excel: results/04_feature_selection/04b_H{1-5}_wrapper.xlsx
- HTML: results/04_feature_selection/04b_H{1-5}_wrapper.html
- JSON: Selected features per horizon

Author: Bankruptcy Prediction Project
Date: 2024-11-18
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
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
        logging.FileHandler(log_dir / "04b_wrapper_methods.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config/project_config.yaml", "r") as f:
    config = yaml.safe_load(f)

fs_config = config["feature_selection"]
HORIZONS = config["datasets"]["polish"]["horizons"]
RANDOM_STATE = config["analysis"]["random_state"]
OUTER_FOLDS = fs_config["outer_folds"]
INNER_FOLDS = fs_config["inner_folds"]
RFE_MIN_FEATURES = fs_config["rfe_min_features"]
RFE_STEP = fs_config["rfe_step"]
RFE_SCORING = fs_config["rfe_scoring"]
# Use a safe default for parallelism; fall back to ridge settings if present
N_JOBS = fs_config.get("n_jobs", fs_config.get("ridge", {}).get("n_jobs", 1))

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


def perform_rfecv(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Perform Recursive Feature Elimination with Cross-Validation.
    
    FIXED: Wraps estimator in Pipeline with StandardScaler to avoid scale bias.
    Returns dict with selected features, rankings, and CV scores.
    """
    logger.info("  Initializing RFECV with Pipeline (Scaler + LR)...")
    
    # CRITICAL FIX: Wrap estimator in Pipeline with scaling
    # This ensures scaling happens inside each CV fold (no leakage)
    estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            penalty="l2",
            solver="lbfgs",  # LBFGS is optimal for L2
            max_iter=fs_config.get("ridge", {}).get("max_iter", 100000),
            tol=fs_config.get("ridge", {}).get("tol", 0.001),
            class_weight=fs_config.get("ridge", {}).get("class_weight", "balanced"),
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])
    
    # RFECV with inner stratified CV
    rfecv = RFECV(
        estimator=estimator,  # Now includes scaling!
        step=RFE_STEP,
        min_features_to_select=RFE_MIN_FEATURES,
        cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring=RFE_SCORING,
        importance_getter='named_steps.clf.coef_',  # Access coef_ from Pipeline
        n_jobs=N_JOBS,
        verbose=1
    )
    
    logger.info(f"  Running RFECV with {INNER_FOLDS}-fold CV (this may take several minutes)...")
    rfecv.fit(X, y)
    
    # Extract results
    optimal_k = rfecv.n_features_
    selected_features = X.columns[rfecv.support_].tolist()
    feature_rankings = pd.DataFrame({
        "feature": X.columns,
        "ranking": rfecv.ranking_,
        "selected": rfecv.support_
    }).sort_values("ranking")
    
    # CV scores per number of features
    cv_scores_df = pd.DataFrame({
        "n_features": range(RFE_MIN_FEATURES, len(X.columns) + 1),
        "cv_score_mean": rfecv.cv_results_["mean_test_score"],
        "cv_score_std": rfecv.cv_results_["std_test_score"]
    })
    
    logger.info(f"  ✓ RFECV complete: {optimal_k} features selected")
    logger.info(f"    Best CV score: {rfecv.cv_results_['mean_test_score'].max():.4f}")
    
    return {
        "optimal_k": optimal_k,
        "selected_features": selected_features,
        "feature_rankings": feature_rankings,
        "cv_scores": cv_scores_df,
        "rfecv_object": rfecv
    }


def evaluate_selected_features_outer_cv(
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: List[str]
) -> Dict:
    """
    Evaluate selected features using outer CV (independent evaluation).
    
    Returns dict with performance metrics.
    """
    logger.info(f"  Evaluating {len(selected_features)} selected features with outer CV...")
    
    X_selected = X[selected_features]
    
    # Pipeline with scaling + logistic regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",  # LBFGS is optimal for L2
            max_iter=fs_config.get("ridge", {}).get("max_iter", 100000),
            tol=fs_config.get("ridge", {}).get("tol", 0.001),
            class_weight=fs_config.get("ridge", {}).get("class_weight", "balanced"),
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])
    
    # Outer stratified CV
    cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    roc_aucs = []
    pr_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y), 1):
        X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)
        
        logger.info(f"    Fold {fold}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    
    return {
        "roc_auc_mean": np.mean(roc_aucs),
        "roc_auc_std": np.std(roc_aucs),
        "pr_auc_mean": np.mean(pr_aucs),
        "pr_auc_std": np.std(pr_aucs),
        "roc_aucs_folds": roc_aucs,
        "pr_aucs_folds": pr_aucs
    }


def process_horizon_wrapper(horizon: int, df: pd.DataFrame) -> Dict:
    """
    Process wrapper methods (RFECV) for a single horizon.
    
    Returns dict with all results.
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
    # Perform RFECV
    # ==========================================================================
    logger.info("\n[1/2] Running RFECV...")
    rfecv_results = perform_rfecv(X, y)
    
    # ==========================================================================
    # Outer CV Evaluation
    # ==========================================================================
    logger.info("\n[2/2] Outer CV evaluation...")
    outer_cv_results = evaluate_selected_features_outer_cv(
        X, y, rfecv_results["selected_features"]
    )
    
    logger.info(f"\n  Final Performance (Outer CV):")
    logger.info(f"    ROC-AUC: {outer_cv_results['roc_auc_mean']:.4f} ± {outer_cv_results['roc_auc_std']:.4f}")
    logger.info(f"    PR-AUC:  {outer_cv_results['pr_auc_mean']:.4f} ± {outer_cv_results['pr_auc_std']:.4f}")
    
    # ==========================================================================
    # Package Results
    # ==========================================================================
    results = {
        "horizon": horizon,
        "n_obs": len(df_h),
        "n_features": len(vif_features),
        "bankruptcy_rate": y.mean(),
        "rfecv": rfecv_results,
        "outer_cv": outer_cv_results
    }
    
    logger.info(f"\nH{horizon} Summary:")
    logger.info(f"  - Optimal features: {rfecv_results['optimal_k']}")
    logger.info(f"  - ROC-AUC: {outer_cv_results['roc_auc_mean']:.4f}")
    
    return results


def save_horizon_results(horizon: int, results: Dict):
    """Save per-horizon results to Excel, HTML, and JSON."""
    logger.info(f"\nSaving H{horizon} results...")
    
    # Excel output
    excel_path = RESULTS_DIR / f"04b_H{horizon}_wrapper.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Sheet 1: Feature rankings
        results["rfecv"]["feature_rankings"].to_excel(
            writer, sheet_name="Feature_Rankings", index=False
        )
        
        # Sheet 2: CV scores per n_features
        results["rfecv"]["cv_scores"].to_excel(
            writer, sheet_name="RFECV_Scores", index=False
        )
        
        # Sheet 3: Selected features
        selected_df = pd.DataFrame({
            "feature": results["rfecv"]["selected_features"]
        })
        selected_df.to_excel(writer, sheet_name="Selected_Features", index=False)
        
        # Sheet 4: Outer CV performance
        outer_cv_df = pd.DataFrame({
            "Fold": range(1, OUTER_FOLDS + 1),
            "ROC_AUC": results["outer_cv"]["roc_aucs_folds"],
            "PR_AUC": results["outer_cv"]["pr_aucs_folds"]
        })
        outer_cv_df.loc[len(outer_cv_df)] = [
            "Mean ± Std",
            f"{results['outer_cv']['roc_auc_mean']:.4f} ± {results['outer_cv']['roc_auc_std']:.4f}",
            f"{results['outer_cv']['pr_auc_mean']:.4f} ± {results['outer_cv']['pr_auc_std']:.4f}"
        ]
        outer_cv_df.to_excel(writer, sheet_name="Outer_CV_Performance", index=False)
        
        # Sheet 5: Summary
        summary_df = pd.DataFrame({
            "Metric": [
                "Horizon",
                "Observations",
                "VIF Features",
                "Bankruptcy Rate",
                "RFECV Optimal K",
                "ROC-AUC (mean)",
                "ROC-AUC (std)",
                "PR-AUC (mean)",
                "PR-AUC (std)"
            ],
            "Value": [
                f"H{horizon}",
                results["n_obs"],
                results["n_features"],
                f"{results['bankruptcy_rate']*100:.2f}%",
                results["rfecv"]["optimal_k"],
                f"{results['outer_cv']['roc_auc_mean']:.4f}",
                f"{results['outer_cv']['roc_auc_std']:.4f}",
                f"{results['outer_cv']['pr_auc_mean']:.4f}",
                f"{results['outer_cv']['pr_auc_std']:.4f}"
            ]
        })
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    logger.info(f"  ✓ Excel: {excel_path}")
    
    # JSON output
    json_data = {
        "horizon": int(horizon),
        "optimal_k": int(results["rfecv"]["optimal_k"]),
        "selected_features": results["rfecv"]["selected_features"],
        "performance": {
            "roc_auc_mean": float(results["outer_cv"]["roc_auc_mean"]),
            "roc_auc_std": float(results["outer_cv"]["roc_auc_std"]),
            "pr_auc_mean": float(results["outer_cv"]["pr_auc_mean"]),
            "pr_auc_std": float(results["outer_cv"]["pr_auc_std"])
        }
    }
    json_path = RESULTS_DIR / f"04b_H{horizon}_wrapper_selected.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"  ✓ JSON: {json_path}")
    
    # HTML report
    html_path = RESULTS_DIR / f"04b_H{horizon}_wrapper.html"
    generate_html_report(horizon, results, html_path)
    logger.info(f"  ✓ HTML: {html_path}")


def generate_html_report(horizon: int, results: Dict, output_path: Path):
    """Generate professional HTML report."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Phase 04b: Wrapper Methods (RFECV) - Horizon {horizon}</title>
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
                border-bottom: 3px solid #e74c3c;
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
                background-color: #e74c3c;
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
                background-color: #fff5f5;
                border-left: 4px solid #e74c3c;
            }}
            .highlight {{
                background-color: #ffffcc;
                padding: 2px 5px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Phase 04b: Wrapper Methods (RFECV) - Horizon {horizon}</h1>
            
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
                <div class="metric">
                    <span class="metric-label">Selected Features:</span>
                    <span class="metric-value" class="highlight">{results['rfecv']['optimal_k']}</span>
                </div>
            </div>
            
            <div class="section">
                <h2>RFECV Performance</h2>
                <p><strong>Method:</strong> Recursive Feature Elimination with Cross-Validation</p>
                <p><strong>Base Estimator:</strong> Logistic Regression (L2 penalty, class_weight='balanced')</p>
                <p><strong>Inner CV:</strong> {INNER_FOLDS}-fold Stratified K-Fold</p>
                <p><strong>Scoring:</strong> {RFE_SCORING.upper()}</p>
                
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td><strong>Optimal Number of Features</strong></td>
                        <td class="highlight"><strong>{results['rfecv']['optimal_k']}</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Best Inner CV Score</strong></td>
                        <td>{results['rfecv']['cv_scores']['cv_score_mean'].max():.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Outer CV ROC-AUC (mean ± std)</strong></td>
                        <td>{results['outer_cv']['roc_auc_mean']:.4f} ± {results['outer_cv']['roc_auc_std']:.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>Outer CV PR-AUC (mean ± std)</strong></td>
                        <td>{results['outer_cv']['pr_auc_mean']:.4f} ± {results['outer_cv']['pr_auc_std']:.4f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Selected Features (Top 15)</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Selection Status</th>
                    </tr>
    """
    
    # Add top 15 features
    top_features = results["rfecv"]["feature_rankings"].head(15)
    for idx, row in top_features.iterrows():
        status = "✓ Selected" if row["selected"] else "✗ Eliminated"
        style = "color: green;" if row["selected"] else "color: #999;"
        html_content += f"""
                    <tr style="{style}">
                        <td>{row['ranking']}</td>
                        <td><strong>{row['feature']}</strong></td>
                        <td>{status}</td>
                    </tr>
        """
    
    html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>Outer CV Performance (Per Fold)</h2>
                <table>
                    <tr>
                        <th>Fold</th>
                        <th>ROC-AUC</th>
                        <th>PR-AUC</th>
                    </tr>
    """
    
    for fold, (roc, pr) in enumerate(zip(results["outer_cv"]["roc_aucs_folds"], 
                                           results["outer_cv"]["pr_aucs_folds"]), 1):
        html_content += f"""
                    <tr>
                        <td>Fold {fold}</td>
                        <td>{roc:.4f}</td>
                        <td>{pr:.4f}</td>
                    </tr>
        """
    
    html_content += f"""
                    <tr style="font-weight: bold; background-color: #f0f0f0;">
                        <td>Mean ± Std</td>
                        <td>{results['outer_cv']['roc_auc_mean']:.4f} ± {results['outer_cv']['roc_auc_std']:.4f}</td>
                        <td>{results['outer_cv']['pr_auc_mean']:.4f} ± {results['outer_cv']['pr_auc_std']:.4f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Interpretation</h2>
                <p><strong>RFECV (Recursive Feature Elimination with Cross-Validation):</strong></p>
                <ul>
                    <li><strong>Wrapper method:</strong> Uses a model (Logistic Regression) to evaluate feature subsets</li>
                    <li><strong>Recursive elimination:</strong> Iteratively removes least important features</li>
                    <li><strong>Cross-validation:</strong> Determines optimal number of features via {INNER_FOLDS}-fold CV</li>
                    <li><strong>Advantage:</strong> Considers feature interactions (unlike filter methods)</li>
                    <li><strong>Trade-off:</strong> Computationally expensive but more accurate</li>
                </ul>
                
                <p><strong>Performance Evaluation:</strong></p>
                <ul>
                    <li><strong>Inner CV:</strong> Used for feature selection (avoid overfitting)</li>
                    <li><strong>Outer CV:</strong> Independent evaluation of selected features</li>
                    <li><strong>Nested CV:</strong> Prevents data leakage and provides unbiased performance estimates</li>
                </ul>
            </div>
            
            <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 0.9em;">
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Project: Bankruptcy Prediction Seminar (FH Wedel) - Phase 04b: Wrapper Methods</p>
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
    logger.info("PHASE 04b: WRAPPER METHODS (RFECV) FOR FEATURE SELECTION")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  - Method: RFECV (Recursive Feature Elimination CV)")
    logger.info(f"  - Base estimator: LogisticRegression (L2, class_weight='balanced')")
    logger.info(f"  - Inner folds: {INNER_FOLDS}")
    logger.info(f"  - Outer folds: {OUTER_FOLDS}")
    logger.info(f"  - Min features: {RFE_MIN_FEATURES}")
    logger.info(f"  - Step: {RFE_STEP}")
    logger.info(f"  - Scoring: {RFE_SCORING}")
    
    # Load imputed dataset
    logger.info(f"\nLoading dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"  ✓ Loaded {len(df):,} observations")
    
    # Process each horizon
    all_results = {}
    for horizon in HORIZONS:
        results = process_horizon_wrapper(horizon, df)
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
            "RFECV_K": res["rfecv"]["optimal_k"],
            "ROC_AUC": f"{res['outer_cv']['roc_auc_mean']:.4f} ± {res['outer_cv']['roc_auc_std']:.4f}",
            "PR_AUC": f"{res['outer_cv']['pr_auc_mean']:.4f} ± {res['outer_cv']['pr_auc_std']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save consolidated summary
    summary_path = RESULTS_DIR / "04b_ALL_wrapper_summary.xlsx"
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    logger.info(f"\n✓ Consolidated summary: {summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 04b COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info(f"Total horizons processed: {len(HORIZONS)}")
    logger.info("\nNext step: Run 04c_embedded_methods.py")


if __name__ == "__main__":
    main()
