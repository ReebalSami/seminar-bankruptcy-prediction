"""
Demo: Quick Visual Generation
==============================

Simple demo showing how to use the visualization functions.
This creates example plots using your existing processed data.

Run: python scripts/demo_visuals.py
"""

from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    plot_class_imbalance_by_horizon,
    plot_roc_pr_curves,
    plot_calibration_curve,
)

def main():
    print("\n🎨 Visualization Demo\n")
    
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "data" / "processed"
    figures_dir = repo_root / "figures" / "demo"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Demo 1: Class Imbalance
    # =========================================================================
    print("📊 Demo 1: Class Imbalance by Horizon")
    
    df_full = pd.read_parquet(data_dir / "poland_clean_full.parquet")
    
    plot_class_imbalance_by_horizon(
        df_full,
        output_path=figures_dir / "demo_class_imbalance.png"
    )
    
    # =========================================================================
    # Demo 2: ROC & PR Curves  
    # =========================================================================
    print("📊 Demo 2: ROC & Precision-Recall Curves")
    
    preds = pd.read_csv(data_dir / "poland_h1_test_predictions.csv")
    
    y_probs = {
        'Logistic Regression': preds['p_logit'].values,
        'Random Forest': preds['p_rf'].values,
        'GLM': preds['p_glm'].values,
    }
    
    plot_roc_pr_curves(
        preds['y_true'].values,
        y_probs,
        output_path=figures_dir / "demo_roc_pr_curves.png"
    )
    
    # =========================================================================
    # Demo 3: Calibration Curves
    # =========================================================================
    print("📊 Demo 3: Calibration Curves")
    
    preds_cal = pd.read_csv(data_dir / "poland_h1_test_predictions_calibrated.csv")
    
    y_probs_cal = {
        'Logit (before)': preds_cal['p_logit_uncal'].values,
        'Logit (after)': preds_cal['p_logit_cal'].values,
        'RF (before)': preds_cal['p_rf_uncal'].values,
        'RF (after)': preds_cal['p_rf_cal'].values,
    }
    
    plot_calibration_curve(
        preds_cal['y_true'].values,
        y_probs_cal,
        output_path=figures_dir / "demo_calibration.png"
    )
    
    print(f"\n✅ Demo complete! Check: {figures_dir}")


if __name__ == '__main__':
    main()
