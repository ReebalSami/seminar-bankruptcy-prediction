"""
Phase 03a: VIF-Based Multicollinearity Control

Iterative VIF pruning to remove multicollinear features.
Threshold: VIF > 10 (standard econometric practice, Penn State STAT 462, O'Brien 2007)

OUTPUT: Per-horizon Excel/HTML/JSON + Consolidated reports
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import json

# Import project utilities
from src.bankruptcy_prediction.utils.target_utils import get_canonical_target
from src.bankruptcy_prediction.utils.logging_setup import setup_logging, print_header, print_section
from src.bankruptcy_prediction.utils.config_loader import get_config


def load_data(config, logger):
    """Load data with canonical target."""
    data_path = PROJECT_ROOT / config.get('paths', 'data') / 'processed' / 'poland_imputed.parquet'
    logger.info(f"Loading data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    logger.info(f"✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Canonicalize target
    df = get_canonical_target(df, drop_duplicates=True)
    
    logger.info(f"✓ Data ready: {df['y'].sum()} bankruptcies / {len(df)} total")
    return df


def get_feature_columns(df, logger):
    """Extract feature columns A1-A64."""
    features = [c for c in df.columns if c.startswith('A') and c[1:].isdigit()]
    features = sorted(features, key=lambda x: int(x[1:]))
    logger.info(f"✓ Feature columns: {len(features)} features")
    return features


def preprocess_features(X, logger):
    """Pre-processing checks before VIF computation."""
    removed = []
    X_clean = X.copy()
    
    # Check for zero variance
    std_vals = X_clean.std()
    zero_var = std_vals[std_vals < 1e-12].index.tolist()
    if zero_var:
        logger.warning(f"Removing {len(zero_var)} zero-variance features")
        removed.extend([{'Feature': f, 'Reason': 'zero_variance', 'VIF': np.nan} for f in zero_var])
        X_clean = X_clean.drop(columns=zero_var)
    
    # Check for NaN/Inf - FAIL FAST
    has_nan = X_clean.columns[X_clean.isna().any()].tolist()
    has_inf = X_clean.columns[np.isinf(X_clean).any()].tolist()
    problematic = list(set(has_nan + has_inf))
    
    if problematic:
        logger.error(f"CRITICAL: Found NaN/Inf in features: {problematic}")
        logger.error("This indicates upstream imputation failure!")
        raise ValueError(
            f"Cannot compute VIF with NaN/Inf values. "
            f"Fix imputation for {len(problematic)} features: {', '.join(problematic)}"
        )
    
    logger.info(f"✓ Pre-processing: {len(X_clean.columns)} features retained, {len(removed)} removed")
    return X_clean, removed


def compute_vif(X, logger):
    """Compute VIF for all features."""
    if len(X.columns) < 2:
        logger.warning("Cannot compute VIF with < 2 features")
        return pd.DataFrame({'Feature': X.columns, 'VIF': [np.nan] * len(X.columns)})
    
    # Add constant for intercept
    X_with_const = add_constant(X, has_constant='add')
    
    vif_data = []
    for i in range(1, X_with_const.shape[1]):  # Skip constant (index 0)
        try:
            vif = variance_inflation_factor(X_with_const.values, i)
            feature = X.columns[i - 1]
            vif_data.append({'Feature': feature, 'VIF': vif})
        except Exception as e:
            feature = X.columns[i - 1]
            logger.warning(f"VIF computation failed for {feature}: {e}")
            vif_data.append({'Feature': feature, 'VIF': np.inf})
    
    return pd.DataFrame(vif_data)


def iterative_vif_pruning(X, threshold, max_iterations, logger=None):
    """Iteratively remove features with VIF > threshold.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    threshold : float
        VIF threshold from config (typically 10)
    max_iterations : int
        Maximum iterations from config
    logger : logging.Logger
        Logger instance
    """
    X_current = X.copy()
    iterations_log = []
    removed_features = []
    
    for iteration in range(1, max_iterations + 1):
        vif_df = compute_vif(X_current, logger)
        
        if len(X_current.columns) <= 2:
            logger.warning(f"Only {len(X_current.columns)} features remain. Stopping.")
            iterations_log.append({
                'Iteration': iteration,
                'Features_Remaining': len(X_current.columns),
                'Max_VIF': vif_df['VIF'].max() if not vif_df.empty else np.nan,
                'Removed_Feature': None,
                'Removed_VIF': None,
                'Reason': 'min_features_reached'
            })
            break
        
        max_vif = vif_df['VIF'].max()
        
        if max_vif <= threshold:
            logger.info(f"Iteration {iteration}: max(VIF) = {max_vif:.2f} ≤ {threshold}. Converged!")
            iterations_log.append({
                'Iteration': iteration,
                'Features_Remaining': len(X_current.columns),
                'Max_VIF': max_vif,
                'Removed_Feature': None,
                'Removed_VIF': None,
                'Reason': 'converged'
            })
            break
        
        # Remove feature with highest VIF
        worst_feature = vif_df.loc[vif_df['VIF'].idxmax()]
        feature_name = worst_feature['Feature']
        feature_vif = worst_feature['VIF']
        
        logger.info(f"Iteration {iteration}: Removing {feature_name} (VIF={feature_vif:.2f})")
        
        iterations_log.append({
            'Iteration': iteration,
            'Features_Remaining': len(X_current.columns),
            'Max_VIF': max_vif,
            'Removed_Feature': feature_name,
            'Removed_VIF': feature_vif,
            'Reason': 'max_vif'
        })
        
        removed_features.append({
            'Feature': feature_name,
            'VIF_at_Removal': feature_vif,
            'Iteration_Removed': iteration,
            'Reason': 'max_vif'
        })
        
        X_current = X_current.drop(columns=[feature_name])
    
    logger.info(f"✓ VIF pruning complete: {len(X.columns)} → {len(X_current.columns)} features")
    return X_current, iterations_log, removed_features


def save_horizon_outputs(horizon, X_initial, X_final, iterations_log, removed_pre, 
                         removed_vif, output_dir, logger, vif_threshold):
    """Save per-horizon outputs: Excel, HTML, JSON.
    
    Parameters
    ----------
    vif_threshold : float
        VIF threshold from config
    """
    # Compute final VIF
    final_vif_df = compute_vif(X_final, logger)
    final_vif_df = final_vif_df.sort_values('VIF', ascending=False)
    
    # Combine removed features
    all_removed = removed_pre + removed_vif
    removed_df = pd.DataFrame(all_removed) if all_removed else pd.DataFrame(
        columns=['Feature', 'VIF_at_Removal', 'Iteration_Removed', 'Reason'])
    
    # Metadata
    metadata = {
        'Horizon': horizon,
        'Initial_Features': len(X_initial.columns),
        'Final_Features': len(X_final.columns),
        'Removed_Count': len(all_removed),
        'Iterations': len(iterations_log),
        'Max_Final_VIF': final_vif_df['VIF'].max() if not final_vif_df.empty else np.nan,
        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save Excel
    excel_path = output_dir / f'03a_H{horizon}_vif.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        pd.DataFrame(iterations_log).to_excel(writer, sheet_name='Iterations', index=False)
        final_vif_df.to_excel(writer, sheet_name='Final_VIF', index=False)
        removed_df.to_excel(writer, sheet_name='Removed_Features', index=False)
        pd.DataFrame([metadata]).to_excel(writer, sheet_name='Metadata', index=False)
    
    logger.info(f"✓ Saved: {excel_path.name}")
    
    # Save JSON (feature list)
    json_dir = PROJECT_ROOT / 'data' / 'processed' / 'feature_sets'
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f'H{horizon}_features.json'
    
    with open(json_path, 'w') as f:
        json.dump(X_final.columns.tolist(), f, indent=2)
    
    logger.info(f"✓ Saved: {json_path.name}")
    
    # Create HTML
    import importlib.util
    spec = importlib.util.spec_from_file_location("html_creator", PROJECT_ROOT / "scripts" / "03_multicollinearity" / "html_creator.py")
    html_creator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(html_creator)
    html_creator.create_html_report(horizon, metadata, iterations_log, final_vif_df, removed_df, output_dir, logger, vif_threshold=vif_threshold)
    
    return metadata


def create_consolidated_reports(all_metadata, output_dir, logger, vif_threshold):
    """Create consolidated Excel and HTML for all horizons.
    
    Parameters
    ----------
    vif_threshold : float
        VIF threshold from config
    """
    print_section(logger, "CREATING CONSOLIDATED REPORTS", width=60)
    
    summary_data = []
    all_removed = []
    
    for meta in all_metadata:
        summary_data.append({
            'Horizon': meta['Horizon'],
            'Initial': meta['Initial_Features'],
            'Final': meta['Final_Features'],
            'Removed': meta['Removed_Count'],
            'Iterations': meta['Iterations'],
            'Max_Final_VIF': meta['Max_Final_VIF']
        })
        
        # Load removed features
        h = meta['Horizon']
        h_excel = output_dir / f'03a_H{h}_vif.xlsx'
        removed_h = pd.read_excel(h_excel, sheet_name='Removed_Features')
        removed_h['Horizon'] = h
        all_removed.append(removed_h)
    
    summary_df = pd.DataFrame(summary_data)
    all_removed_df = pd.concat(all_removed, ignore_index=True) if all_removed else pd.DataFrame()
    
    with pd.ExcelWriter(output_dir / '03a_ALL_vif.xlsx', engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        if not all_removed_df.empty:
            all_removed_df.to_excel(writer, sheet_name='All_Removed', index=False)
    
    logger.info(f"✓ Saved: 03a_ALL_vif.xlsx")
    
    # Create HTML
    import importlib.util
    spec = importlib.util.spec_from_file_location("html_creator", PROJECT_ROOT / "scripts" / "03_multicollinearity" / "html_creator.py")
    html_creator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(html_creator)
    html_creator.create_consolidated_html(summary_df, output_dir, logger, vif_threshold=vif_threshold)


def main():
    """Main execution function."""
    logger = setup_logging('03a_vif_analysis')
    
    # Load configuration
    config = get_config()
    vif_threshold = config.get_analysis_param('vif_threshold_high')
    max_iterations = config.get_analysis_param('vif_max_iterations')
    
    logger.info(f"Configuration: VIF threshold = {vif_threshold}, Max iterations = {max_iterations}")
    
    print_header(logger, "PHASE 03a: VIF-BASED MULTICOLLINEARITY CONTROL", width=80)
    
    # Create output directory
    output_dir = PROJECT_ROOT / config.get('paths', 'results') / '03_multicollinearity'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print_section(logger, "LOADING DATA", width=60)
    df = load_data(config, logger)
    features = get_feature_columns(df, logger)
    
    # Process each horizon
    all_metadata = []
    
    for horizon in range(1, 6):
        print_section(logger, f"HORIZON {horizon} (H{horizon})", width=60)
        
        df_h = df[df['horizon'] == horizon]
        logger.info(f"Filtered: {len(df_h):,} observations")
        
        X = df_h[features].copy()
        
        # Pre-processing
        X_clean, removed_pre = preprocess_features(X, logger)
        
        # VIF pruning
        X_final, iterations_log, removed_vif = iterative_vif_pruning(
            X_clean, threshold=vif_threshold, max_iterations=max_iterations, logger=logger)
        
        # Save outputs
        metadata = save_horizon_outputs(
            horizon, X, X_final, iterations_log, removed_pre, removed_vif, output_dir, logger, vif_threshold=vif_threshold)
        all_metadata.append(metadata)
        
        logger.info(f"✅ H{horizon} Complete")
    
    # Consolidated reports
    create_consolidated_reports(all_metadata, output_dir, logger, vif_threshold)
    
    print_header(logger, "PHASE 03a COMPLETE", width=80)
    logger.info(f"Total files created: {5 * 3 + 2} (15 per-horizon + 2 consolidated)")


if __name__ == '__main__':
    main()
