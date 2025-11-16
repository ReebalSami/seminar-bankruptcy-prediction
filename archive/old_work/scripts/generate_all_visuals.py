"""
Generate All Publication-Quality Visualizations
================================================

This script creates all visualizations with proper feature names and categories.
Run this after data preprocessing and modeling to generate all figures.

Usage:
    python scripts/generate_all_visuals.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    plot_class_imbalance_by_horizon,
    plot_missingness,
    plot_distributions_by_class,
    plot_correlation_heatmap,
    find_high_correlations,
    plot_roc_pr_curves,
    plot_calibration_curve,
    plot_feature_importance,
    plot_odds_ratios
)
from src.metadata import FEATURE_NAMES


def main():
    print("\n" + "="*70)
    print("  GENERATING ALL VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Setup paths
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "data" / "processed"
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print(f"📂 Data directory: {data_dir}")
    print(f"📊 Figures directory: {figures_dir}\n")
    
    # =========================================================================
    # 1. CLASS IMBALANCE BY HORIZON
    # =========================================================================
    print("\n[1/9] Class Imbalance by Horizon...")
    
    df_full = pd.read_parquet(data_dir / "poland_clean_full.parquet")
    
    plot_class_imbalance_by_horizon(
        df_full,
        output_path=figures_dir / "01_class_imbalance_by_horizon.png"
    )
    
    # =========================================================================
    # 2. MISSINGNESS ANALYSIS
    # =========================================================================
    print("\n[2/9] Missingness Analysis...")
    
    # Load raw data before imputation to show missingness
    # If you don't have this, skip or use imputed data
    try:
        # You'll need to save raw data before imputation in your preprocessing
        df_raw = pd.read_parquet(data_dir / "poland_raw_before_imputation.parquet")
        plot_missingness(
            df_raw,
            top_n=20,
            output_path=figures_dir / "02_missingness_top20.png"
        )
    except FileNotFoundError:
        print("   ⚠️  Raw data not found - skipping missingness plot")
        print("   💡 Save data before imputation as 'poland_raw_before_imputation.parquet'")
    
    # =========================================================================
    # 3. FEATURE DISTRIBUTIONS BY CLASS
    # =========================================================================
    print("\n[3/9] Feature Distributions by Class...")
    
    # Select top 10 important features (you can customize this)
    top_features = [
        'Attr1',   # Net Profit / Assets
        'Attr2',   # Liabilities / Assets
        'Attr7',   # EBIT / Assets
        'Attr10',  # Equity / Assets
        'Attr18',  # Gross Profit / Assets
        'Attr22',  # Operating Profit / Assets
        'Attr29',  # Log(Total Assets)
        'Attr36',  # Sales / Assets
        'Attr48',  # EBITDA / Assets
        'Attr56',  # Gross Margin
    ]
    
    df_h1 = df_full[df_full['horizon'] == 1].copy()
    
    # Check which features exist
    available_features = [f for f in top_features if f in df_h1.columns]
    
    if len(available_features) > 0:
        plot_distributions_by_class(
            df_h1,
            features=available_features[:10],
            target_col='y',
            output_path=figures_dir / "03_distributions_by_class.png",
            log_scale=False
        )
    else:
        print("   ⚠️  No features found - check column names")
    
    # =========================================================================
    # 4. CORRELATION HEATMAP
    # =========================================================================
    print("\n[4/9] Correlation Heatmap...")
    
    df_reduced = pd.read_parquet(data_dir / "poland_clean_reduced.parquet")
    df_reduced_h1 = df_reduced[df_reduced['horizon'] == 1].drop(columns=['y', 'horizon'])
    
    plot_correlation_heatmap(
        df_reduced_h1,
        method='pearson',
        output_path=figures_dir / "04_correlation_heatmap.png"
    )
    
    # =========================================================================
    # 5. HIGH CORRELATIONS
    # =========================================================================
    print("\n[5/9] Finding High Correlations...")
    
    high_corr = find_high_correlations(
        df_reduced_h1,
        threshold=0.9,
        output_path=figures_dir / "05_high_correlations.csv"
    )
    
    print(f"   Found {len(high_corr)} highly correlated pairs (|r| ≥ 0.9)")
    
    # =========================================================================
    # 6. ROC & PR CURVES
    # =========================================================================
    print("\n[6/9] ROC & Precision-Recall Curves...")
    
    # Load predictions
    preds_file = data_dir / "poland_h1_test_predictions.csv"
    if preds_file.exists():
        preds = pd.read_csv(preds_file)
        
        y_probs_dict = {
            'Logistic Regression': preds['p_logit'].values,
            'Random Forest': preds['p_rf'].values,
            'GLM': preds['p_glm'].values,
        }
        
        plot_roc_pr_curves(
            preds['y_true'].values,
            y_probs_dict,
            output_path=figures_dir / "06_roc_pr_curves.png"
        )
    else:
        print("   ⚠️  Predictions file not found - skipping ROC/PR curves")
    
    # =========================================================================
    # 7. CALIBRATION CURVES
    # =========================================================================
    print("\n[7/9] Calibration Curves...")
    
    preds_cal_file = data_dir / "poland_h1_test_predictions_calibrated.csv"
    if preds_cal_file.exists():
        preds_cal = pd.read_csv(preds_cal_file)
        
        y_probs_cal_dict = {
            'Logit (uncalibrated)': preds_cal['p_logit_uncal'].values,
            'Logit (calibrated)': preds_cal['p_logit_cal'].values,
            'RF (uncalibrated)': preds_cal['p_rf_uncal'].values,
            'RF (calibrated)': preds_cal['p_rf_cal'].values,
        }
        
        plot_calibration_curve(
            preds_cal['y_true'].values,
            y_probs_cal_dict,
            n_bins=10,
            output_path=figures_dir / "07_calibration_curves.png"
        )
    else:
        print("   ⚠️  Calibrated predictions not found - skipping calibration curves")
    
    # =========================================================================
    # 8. FEATURE IMPORTANCE
    # =========================================================================
    print("\n[8/9] Feature Importance (Random Forest)...")
    
    # You'll need to save feature importances from your RF model
    # For now, create a dummy example
    print("   ⚠️  This requires RF feature importances from your model")
    print("   💡 Add this line in your modeling notebook:")
    print("      pd.Series(rf.feature_importances_, index=X.columns).to_csv('rf_importances.csv')")
    
    importance_file = data_dir / "rf_feature_importances.csv"
    if importance_file.exists():
        importances = pd.read_csv(importance_file, index_col=0, squeeze=True)
        plot_feature_importance(
            importances,
            top_n=20,
            output_path=figures_dir / "08_feature_importance_rf.png"
        )
    
    # =========================================================================
    # 9. ODDS RATIOS
    # =========================================================================
    print("\n[9/9] Odds Ratios (GLM)...")
    
    # You'll need GLM coefficients
    print("   ⚠️  This requires GLM coefficients from your model")
    print("   💡 Add this line in your modeling notebook:")
    print("      glm_res.summary().tables[1].to_csv('glm_coefficients.csv')")
    
    coef_file = data_dir / "glm_coefficients.csv"
    if coef_file.exists():
        coefs = pd.read_csv(coef_file)
        # Assuming columns: feature, coef, std_err, z, pval
        plot_odds_ratios(
            coefs,
            top_n=20,
            output_path=figures_dir / "09_odds_ratios_glm.png"
        )
    
    # =========================================================================
    print("\n" + "="*70)
    print("  ✅ VISUALIZATION GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📊 All figures saved to: {figures_dir}")
    print(f"   Total files: {len(list(figures_dir.glob('*.png')))}")
    print()


if __name__ == '__main__':
    main()
