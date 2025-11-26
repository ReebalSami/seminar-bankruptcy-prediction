"""
Phase 04d: Stability Analysis & Consensus Feature Selection
============================================================

Integrates results from all feature selection methods:
- Filter methods (Spearman, MI, ANOVA F)
- Wrapper methods (RFECV)
- Embedded methods (Lasso, Random Forest)

Performs:
1. Cross-method agreement analysis (Jaccard similarity)
2. Stability metrics (Nogueira stability)
3. Consensus feature set generation
4. Baseline performance validation

Input:
- Filter results: results/04_feature_selection/04a_H{1-5}_filter_selected.json
- Wrapper results: results/04_feature_selection/04b_H{1-5}_wrapper_selected.json
- Embedded results: results/04_feature_selection/04c_H{1-5}_embedded_selected.json

Output:
- Final feature sets: data/processed/feature_sets_selected/H{1-5}_features_final.json
- Consolidated reports: results/04_feature_selection/04d_ALL_consensus.xlsx/.html
"""

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set
import argparse

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =============================================================================
# Setup
# =============================================================================

log_dir = Path("logs/04_feature_selection")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "04d_stability_consensus.log"),
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
CONSENSUS_METHOD = fs_config["consensus_method"]
MIN_METHOD_AGREEMENT = fs_config["min_method_agreement"]
BASELINE_THRESHOLD = fs_config["baseline_performance_threshold"]
MIN_EPV = config["analysis"].get("min_epv", 10)

# Paths
DATA_PATH = Path("data/processed/poland_imputed.parquet")
RESULTS_DIR = Path("results/04_feature_selection")
OUTPUT_DIR = Path("data/processed/feature_sets_selected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Runtime-variant globals (set in main)
FILE_SUFFIX = ""
OUTPUT_DIR_FINAL = OUTPUT_DIR
RESULTS_DIR_FINAL = RESULTS_DIR

# =============================================================================
# Helper Functions
# =============================================================================

def compute_jaccard_similarity(set1: Set, set2: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_nogueira_stability(fold_selections: List[Set], n_features: int) -> float:
    """
    Compute Nogueira stability metric CORRECTLY.
    
    Nogueira et al. (2018): "On the Stability of Feature Selection Algorithms"
    Measures stability of ONE method across MULTIPLE folds/resamples.
    
    Args:
        fold_selections: List of feature sets from CV folds of SAME method
        n_features: Total number of features available
    
    Range: [0, 1], where 1 = perfect stability
    """
    k = len(fold_selections)
    if k < 2:
        return 1.0
    
    # Average pairwise Jaccard similarity across folds
    jaccard_sum = 0
    count = 0
    for s1, s2 in combinations(fold_selections, 2):
        jaccard_sum += compute_jaccard_similarity(s1, s2)
        count += 1
    
    avg_jaccard = jaccard_sum / count if count > 0 else 0
    
    # Nogueira stability corrects for random agreement
    k_avg = np.mean([len(s) for s in fold_selections])
    p_random = (k_avg / n_features) ** 2
    
    if p_random >= 1:
        return avg_jaccard
    
    nogueira_stability = (avg_jaccard - p_random) / (1 - p_random)
    return max(0, min(1, nogueira_stability))


def load_method_results(horizon: int) -> Dict:
    """Load results from all feature selection methods."""
    logger.info(f"  Loading results for H{horizon}...")
    
    # Filter methods
    filter_path = RESULTS_DIR / f"04a_H{horizon}_filter_selected.json"
    if filter_path.exists():
        with open(filter_path) as f:
            filter_data = json.load(f)
    else:
        logger.warning(f"    Filter results not found: {filter_path}")
        filter_data = {"spearman_selected": [], "mi_selected": [], "anova_selected": []}
    
    # Wrapper methods
    wrapper_path = RESULTS_DIR / f"04b_H{horizon}_wrapper_selected.json"
    if wrapper_path.exists():
        with open(wrapper_path) as f:
            wrapper_data = json.load(f)
    else:
        logger.warning(f"    Wrapper results not found: {wrapper_path}")
        wrapper_data = {"selected_features": []}
    
    # Embedded methods (non-nested, summary)
    embedded_path = RESULTS_DIR / f"04c_H{horizon}_embedded_selected.json"
    if embedded_path.exists():
        with open(embedded_path) as f:
            embedded_data = json.load(f)
        # Normalize structure if wrapped under 'methods'
        if isinstance(embedded_data, dict) and "methods" in embedded_data:
            embedded_data = embedded_data["methods"]
    else:
        logger.warning(f"    Embedded results not found: {embedded_path}")
        embedded_data = {"lasso": {"selected_features": []}, "random_forest": {"selected_features": []}}

    # Optional: nested outputs for stability per method
    nested_path = RESULTS_DIR / "nested" / f"04c_H{horizon}_embedded_nested.json"
    nested_data = None
    if nested_path.exists():
        with open(nested_path) as f:
            tmp = json.load(f)
        nested_data = tmp.get("methods", {})
    
    return {
        "filter": filter_data,
        "wrapper": wrapper_data,
        "embedded": embedded_data,
        "nested": nested_data,
    }


def create_consensus_features(method_selections: Dict[str, List[str]], method: str) -> List[str]:
    """
    Create consensus feature set from multiple methods.
    
    Args:
        method_selections: Dict mapping method name to selected features
        method: "intersection" or "majority_vote"
    """
    all_methods = list(method_selections.keys())
    all_features = set()
    for features in method_selections.values():
        all_features.update(features)
    
    if method == "intersection":
        # Features selected by ALL methods
        consensus = set(method_selections[all_methods[0]])
        for method_features in method_selections.values():
            consensus &= set(method_features)
        return sorted(list(consensus))
    
    elif method == "majority_vote":
        # Features selected by majority of methods
        feature_votes = {feat: 0 for feat in all_features}
        for features in method_selections.values():
            for feat in features:
                feature_votes[feat] += 1
        
        threshold = len(all_methods) / 2
        consensus = [feat for feat, votes in feature_votes.items() if votes > threshold]
        
        # Sort by number of votes (descending)
        consensus_sorted = sorted(consensus, key=lambda f: feature_votes[f], reverse=True)
        return consensus_sorted
    
    else:
        raise ValueError(f"Unknown consensus method: {method}")


def evaluate_feature_set_performance(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    cv_folds: int = 5
) -> Dict:
    """Evaluate feature set performance using cross-validation."""
    if len(features) == 0:
        return {"roc_auc_mean": 0, "roc_auc_std": 0, "pr_auc_mean": 0, "pr_auc_std": 0}
    
    X_selected = X[features]
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",  # LBFGS is optimal for L2
            max_iter=100000,  # M1 Pro can handle this
            tol=0.001,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    roc_aucs = []
    pr_aucs = []
    
    for train_idx, val_idx in cv.split(X_selected, y):
        X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        
        roc_aucs.append(roc_auc_score(y_val, y_pred_proba))
        pr_aucs.append(average_precision_score(y_val, y_pred_proba))
    
    return {
        "roc_auc_mean": np.mean(roc_aucs),
        "roc_auc_std": np.std(roc_aucs),
        "pr_auc_mean": np.mean(pr_aucs),
        "pr_auc_std": np.std(pr_aucs)
    }


def analyze_horizon_consensus(horizon: int, df: pd.DataFrame, vif_features: List[str]) -> Dict:
    """Perform comprehensive consensus analysis for a single horizon."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing Consensus: Horizon {horizon}")
    logger.info(f"{'='*80}")
    
    # Load method results
    method_results = load_method_results(horizon)
    
    # Extract feature selections — prefer nested embedded selections when available
    def _extract_nested_feats(node) -> List[str]:
        if not node or not isinstance(node, dict):
            return []
        # common keys
        if isinstance(node.get("selected_features"), list):
            return list(node["selected_features"])  # type: ignore
        folds = node.get("fold_selections")
        if isinstance(folds, list) and folds and all(isinstance(f, list) for f in folds):
            # majority over folds
            from collections import Counter
            c = Counter()
            for f in folds:
                c.update(f)
            # keep those appearing in >50% of folds
            thresh = len(folds) / 2
            return [feat for feat, cnt in c.items() if cnt > thresh]
        return []

    embedded = method_results["embedded"]
    nested = method_results.get("nested") or {}
    l1 = embedded.get("lasso", {}).get("selected_features", [])
    rf_emb = embedded.get("random_forest", {}).get("selected_features", [])
    # Prefer nested if available
    if isinstance(nested, dict):
        if "lasso" in nested:
            cand = _extract_nested_feats(nested["lasso"])
            if cand:
                l1 = cand
        if "random_forest" in nested:
            cand = _extract_nested_feats(nested["random_forest"])
            if cand:
                rf_emb = cand

    selections = {
        "Spearman": method_results["filter"]["spearman_selected"],
        "Mutual_Info": method_results["filter"]["mi_selected"],
        "ANOVA_F": method_results["filter"]["anova_selected"],
        "RFECV": method_results["wrapper"]["selected_features"],
        "Lasso_L1": l1,
        "Random_Forest": rf_emb,
    }
    
    # Filter out empty methods
    selections = {k: v for k, v in selections.items() if len(v) > 0}
    
    logger.info(f"\n  Methods available: {list(selections.keys())}")
    for method, features in selections.items():
        logger.info(f"    {method}: {len(features)} features")
    
    # ==========================================================================
    # 1. Cross-Method Agreement (NOT stability - just agreement!)
    # ==========================================================================
    logger.info("\n  [1/3] Computing cross-method AGREEMENT (pairwise Jaccard)...")
    
    method_pairs = list(combinations(selections.keys(), 2))
    jaccard_matrix = []
    
    for m1, m2 in method_pairs:
        jacc = compute_jaccard_similarity(set(selections[m1]), set(selections[m2]))
        jaccard_matrix.append({
            "Method_1": m1,
            "Method_2": m2,
            "Jaccard": jacc,
            "Overlap_Count": len(set(selections[m1]) & set(selections[m2]))
        })
    
    jaccard_df = pd.DataFrame(jaccard_matrix).sort_values("Jaccard", ascending=False)
    mean_agreement = jaccard_df["Jaccard"].mean()
    
    logger.info(f"    Mean cross-method agreement: {mean_agreement:.3f}")
    
    # NOTE: Removed incorrect "Nogueira stability" computation across methods
    # True Nogueira stability requires per-method fold selections (not available from filter/wrapper/embedded outputs)
    
    # ==========================================================================
    # 2. Consensus Feature Sets
    # ==========================================================================
    logger.info("\n  [2/3] Creating consensus feature sets...")
    
    # Intersection (features selected by ALL methods)
    consensus_intersection = create_consensus_features(selections, "intersection")
    logger.info(f"    Intersection: {len(consensus_intersection)} features")
    
    # Majority vote (features selected by >50% of methods)
    consensus_majority = create_consensus_features(selections, "majority_vote")
    logger.info(f"    Majority vote: {len(consensus_majority)} features")
    
    # Union (features selected by ANY method) - for reference
    all_selected = set()
    for feats in selections.values():
        all_selected.update(feats)
    consensus_union = sorted(list(all_selected))
    logger.info(f"    Union: {len(consensus_union)} features")
    
    # ==========================================================================
    # 3. Performance Evaluation WITH GUARDRAILS
    # ==========================================================================
    logger.info("\n  [3/3] Evaluating performance with guardrails...")
    
    # Prepare data
    df_h = df[df["horizon"] == horizon].copy()
    X = df_h[vif_features]
    y = df_h["bankrupt"]
    
    # Baseline: all VIF features
    baseline_perf = evaluate_feature_set_performance(X, y, vif_features)
    logger.info(f"    Baseline ({len(vif_features)} VIF features): ROC-AUC={baseline_perf['roc_auc_mean']:.4f}")
    
    # Try consensus methods with performance threshold
    final_consensus = None
    final_method = None
    consensus_perf = None
    
    # Try primary consensus method first
    if CONSENSUS_METHOD == "intersection":
        candidate = consensus_intersection
    else:
        candidate = consensus_majority
    
    # EPV guardrail: limit features based on events per variable
    events = int(y.sum())
    if events > 0 and len(candidate) > int(events // MIN_EPV):
        allowed = int(events // MIN_EPV)
        logger.warning(f"    EPV guardrail: {len(candidate)} features > allowed {allowed}; truncating candidate")
        candidate = candidate[:allowed]

    if len(candidate) > 0:
        perf = evaluate_feature_set_performance(X, y, candidate)
        retention = perf["roc_auc_mean"] / baseline_perf["roc_auc_mean"]
        
        logger.info(f"    Testing {CONSENSUS_METHOD}: {len(candidate)} features, ROC-AUC={perf['roc_auc_mean']:.4f}, retention={retention*100:.1f}%")
        
        # CRITICAL: Enforce baseline_performance_threshold guardrail
        if retention >= BASELINE_THRESHOLD:
            final_consensus = candidate
            final_method = CONSENSUS_METHOD
            consensus_perf = perf
            logger.info(f"    ✓ {CONSENSUS_METHOD} meets threshold (≥{BASELINE_THRESHOLD*100:.0f}%)")
        else:
            logger.warning(f"    ✗ {CONSENSUS_METHOD} fails threshold ({retention*100:.1f}% < {BASELINE_THRESHOLD*100:.0f}%)")
    
    # Fallback: Try alternative if primary failed
    if final_consensus is None:
        logger.info("    Trying fallback: majority_vote...")
        fallback = consensus_majority if CONSENSUS_METHOD != "majority_vote" else consensus_intersection

        # EPV guardrail for fallback
        if events > 0 and len(fallback) > int(events // MIN_EPV):
            allowed = int(events // MIN_EPV)
            logger.warning(f"    EPV guardrail: fallback {len(fallback)} > allowed {allowed}; truncating")
            fallback = fallback[:allowed]
        
        if len(fallback) > 0:
            perf = evaluate_feature_set_performance(X, y, fallback)
            retention = perf["roc_auc_mean"] / baseline_perf["roc_auc_mean"]
            
            logger.info(f"    Fallback: {len(fallback)} features, ROC-AUC={perf['roc_auc_mean']:.4f}, retention={retention*100:.1f}%")
            
            if retention >= BASELINE_THRESHOLD:
                final_consensus = fallback
                final_method = "majority_vote" if CONSENSUS_METHOD != "majority_vote" else "intersection"
                consensus_perf = perf
                logger.info(f"    ✓ Fallback meets threshold")
            else:
                logger.warning(f"    ✗ Fallback also fails threshold")
    
    # Last resort: Use union if all else fails
    if final_consensus is None or len(final_consensus) == 0:
        logger.warning("    All consensus methods failed - using union as last resort")
        final_consensus = consensus_union
        final_method = "union"
        consensus_perf = evaluate_feature_set_performance(X, y, final_consensus)
    
    retention_ratio = consensus_perf["roc_auc_mean"] / baseline_perf["roc_auc_mean"]
    logger.info(f"\n    FINAL: {final_method} with {len(final_consensus)} features")
    logger.info(f"    ROC-AUC: {consensus_perf['roc_auc_mean']:.4f} (retention: {retention_ratio*100:.1f}%)")
    
    # ==========================================================================
    # Package Results
    # ==========================================================================
    # Optional stability from nested outputs
    stability_info = {}
    if method_results.get("nested"):
        for key in ["lasso", "elastic_net", "ridge"]:
            if key in method_results["nested"]:
                stability_info[key] = method_results["nested"][key].get("stability_nogueira")

    return {
        "horizon": horizon,
        "n_vif_features": len(vif_features),
        "method_selections": selections,
        "jaccard_matrix": jaccard_df,
        "mean_agreement": mean_agreement,  # Renamed from mean_jaccard for clarity
        "consensus": {
            "intersection": consensus_intersection,
            "majority_vote": consensus_majority,
            "union": consensus_union,
            "final": final_consensus,
            "method_used": final_method  # Actual method used (may differ from config)
        },
        "performance": {
            "baseline": baseline_perf,
            "consensus": consensus_perf,
            "retention_ratio": retention_ratio,
            "threshold_met": retention_ratio >= BASELINE_THRESHOLD
        },
        "stability": stability_info,
    }


def save_final_feature_sets(all_results: Dict):
    """Save final consensus feature sets for downstream modeling."""
    logger.info("\nSaving final feature sets...")
    
    for horizon in HORIZONS:
        res = all_results[f"H{horizon}"]
        
        # Save as JSON
        # Ensure all numpy types are converted to native Python types
        baseline_perf = res["performance"]["baseline"]
        consensus_perf = res["performance"]["consensus"]
        baseline_perf_py = {
            "roc_auc_mean": float(baseline_perf["roc_auc_mean"]),
            "roc_auc_std": float(baseline_perf["roc_auc_std"]),
            "pr_auc_mean": float(baseline_perf["pr_auc_mean"]),
            "pr_auc_std": float(baseline_perf["pr_auc_std"]),
        }
        consensus_perf_py = {
            "roc_auc_mean": float(consensus_perf["roc_auc_mean"]),
            "roc_auc_std": float(consensus_perf["roc_auc_std"]),
            "pr_auc_mean": float(consensus_perf["pr_auc_mean"]),
            "pr_auc_std": float(consensus_perf["pr_auc_std"]),
        }
        retention_ratio_py = float(res["performance"]["retention_ratio"]) 
        threshold_met_py = bool(res["performance"]["threshold_met"])
        mean_agreement_py = float(res["mean_agreement"])

        output_data = {
            "horizon": horizon,
            "features": res["consensus"]["final"],
            "count": len(res["consensus"]["final"]),
            "method_used": res["consensus"]["method_used"],
            "mean_cross_method_agreement": mean_agreement_py,
            "performance": consensus_perf_py,
            "baseline_performance": baseline_perf_py,
            "retention_ratio": retention_ratio_py,
            "threshold_met": threshold_met_py
        }
        
        json_path = OUTPUT_DIR_FINAL / f"H{horizon}_features_final{FILE_SUFFIX}.json"
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"  ✓ H{horizon}: {json_path} ({len(res['consensus']['final'])} features)")


def generate_consolidated_report(all_results: Dict):
    """Generate consolidated Excel and HTML reports."""
    logger.info("\nGenerating consolidated reports...")
    
    # ==========================================================================
    # Excel Report
    # ==========================================================================
    excel_path = RESULTS_DIR_FINAL / f"04d_ALL_consensus{FILE_SUFFIX}.xlsx"
    
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Sheet 1: Summary
        summary_rows = []
        for horizon in HORIZONS:
            res = all_results[f"H{horizon}"]
            stab_vals = list(res.get("stability", {}).values()) if res.get("stability") else []
            stab_avg = float(np.mean(stab_vals)) if stab_vals else None
            summary_rows.append({
                "Horizon": f"H{horizon}",
                "VIF_Features": res["n_vif_features"],
                "Consensus_Features": len(res["consensus"]["final"]),
                "Method_Used": res["consensus"]["method_used"],
                "Reduction_%": (1 - len(res["consensus"]["final"]) / res["n_vif_features"]) * 100,
                "Cross_Method_Agreement": res["mean_agreement"],
                "Nested_Stability_Avg": stab_avg,
                "Baseline_ROC_AUC": res["performance"]["baseline"]["roc_auc_mean"],
                "Consensus_ROC_AUC": res["performance"]["consensus"]["roc_auc_mean"],
                "Retention_%": res["performance"]["retention_ratio"] * 100,
                "Threshold_Met": "✓" if res["performance"]["threshold_met"] else "✗"
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        # Sheet 2: Method Comparison (per horizon)
        for horizon in HORIZONS:
            res = all_results[f"H{horizon}"]
            method_comp = pd.DataFrame({
                "Method": list(res["method_selections"].keys()),
                "Features_Count": [len(v) for v in res["method_selections"].values()],
                "Features": [", ".join(v[:10]) + "..." for v in res["method_selections"].values()]
            })
            method_comp.to_excel(writer, sheet_name=f"H{horizon}_Methods", index=False)
        
        # Sheet 3: Jaccard similarity matrices (all horizons)
        for horizon in HORIZONS:
            res = all_results[f"H{horizon}"]
            res["jaccard_matrix"].to_excel(writer, sheet_name=f"H{horizon}_Jaccard", index=False)
    
    logger.info(f"  ✓ Excel: {excel_path}")
    
    # ==========================================================================
    # HTML Report
    # ==========================================================================
    html_path = RESULTS_DIR_FINAL / f"04d_ALL_consensus{FILE_SUFFIX}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Phase 04d: Consensus Feature Selection</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #27ae60; color: white; font-weight: bold; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .metric {{ display: inline-block; margin: 10px 20px 10px 0; padding: 10px 15px; background-color: #ecf0f1; border-radius: 5px; }}
            .metric-label {{ font-weight: bold; color: #7f8c8d; }}
            .metric-value {{ font-size: 1.2em; color: #2c3e50; }}
            .section {{ margin-top: 30px; padding: 20px; background-color: #f0fff4; border-left: 4px solid #27ae60; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Phase 04d: Consensus Feature Selection - All Horizons</h1>
            
            <div class="section">
                <h2>Overview</h2>
                <p><strong>Consensus Method:</strong> {CONSENSUS_METHOD.replace('_', ' ').title()}</p>
                <p><strong>Methods Integrated:</strong> 6 (Spearman, Mutual Info, ANOVA F, RFECV, Lasso L1, Random Forest)</p>
            </div>
            
            <div class="section">
                <h2>Summary Across All Horizons</h2>
                <table>
                    <tr>
                        <th>Horizon</th>
                        <th>VIF Features</th>
                        <th>Consensus Features</th>
                        <th>Method Used</th>
                        <th>Reduction %</th>
                        <th>Agreement</th>
                        <th>Retention %</th>
                        <th>Threshold</th>
                    </tr>
    """
    
    for horizon in HORIZONS:
        res = all_results[f"H{horizon}"]
        reduction = (1 - len(res["consensus"]["final"]) / res["n_vif_features"]) * 100
        threshold_icon = "✓" if res["performance"]["threshold_met"] else "✗"
        html_content += f"""
                    <tr>
                        <td><strong>H{horizon}</strong></td>
                        <td>{res['n_vif_features']}</td>
                        <td>{len(res['consensus']['final'])}</td>
                        <td>{res['consensus']['method_used']}</td>
                        <td>{reduction:.1f}%</td>
                        <td>{res['mean_agreement']:.3f}</td>
                        <td>{res['performance']['retention_ratio']*100:.1f}%</td>
                        <td>{threshold_icon}</td>
                    </tr>
        """
    
    html_content += f"""
                </table>
            </div>
            
            <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 0.9em;">
                <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Project: Bankruptcy Prediction Seminar - Phase 04d: Consensus Analysis</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"  ✓ HTML: {html_path}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Phase 04d: Stability & Consensus")
    parser.add_argument("--variant", choices=["base", "nested"], default="base", help="Write outputs with suffix when 'nested'")
    args = parser.parse_args()

    global FILE_SUFFIX, OUTPUT_DIR_FINAL, RESULTS_DIR_FINAL
    if args.variant == "nested":
        FILE_SUFFIX = "_nested"
    else:
        FILE_SUFFIX = ""
    OUTPUT_DIR_FINAL = OUTPUT_DIR
    RESULTS_DIR_FINAL = RESULTS_DIR

    logger.info("="*80)
    logger.info("PHASE 04d: STABILITY ANALYSIS & CONSENSUS FEATURE SELECTION")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  - Consensus method: {CONSENSUS_METHOD}")
    logger.info(f"  - Min method agreement: {MIN_METHOD_AGREEMENT}")
    logger.info(f"  - Baseline threshold: {BASELINE_THRESHOLD}")
    
    # Load dataset
    logger.info(f"\nLoading dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"  ✓ Loaded {len(df):,} observations")
    
    # Load VIF features per horizon
    vif_features_per_horizon = {}
    for horizon in HORIZONS:
        vif_file = Path(f"data/processed/feature_sets/H{horizon}_features.json")
        with open(vif_file) as f:
            vif_features_per_horizon[horizon] = json.load(f)
    
    # Process each horizon
    all_results = {}
    for horizon in HORIZONS:
        results = analyze_horizon_consensus(
            horizon, df, vif_features_per_horizon[horizon]
        )
        all_results[f"H{horizon}"] = results
    
    # Save final feature sets
    save_final_feature_sets(all_results)
    
    # Generate consolidated reports
    generate_consolidated_report(all_results)
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 04d COMPLETE")
    logger.info("="*80)
    logger.info(f"Final feature sets saved to: {OUTPUT_DIR}")
    logger.info(f"Consolidated reports saved to: {RESULTS_DIR}")
    logger.info("\nNext step: Phase 05 - Modeling")


if __name__ == "__main__":
    main()
