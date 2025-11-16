"""
Cross-Horizon Robustness Analysis
==================================

Train models on one horizon and evaluate on others to test generalization.

Run: python scripts/run_cross_horizon.py
"""

from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.cross_horizon import (
    full_cross_horizon_matrix,
    plot_cross_horizon_heatmap,
    summarize_degradation
)


def main():
    print("\n" + "="*70)
    print("  CROSS-HORIZON ROBUSTNESS ANALYSIS")
    print("="*70 + "\n")
    
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "data" / "processed"
    results_dir = repo_root / "results" / "cross_horizon"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📂 Loading data...")
    df_full = pd.read_parquet(data_dir / "poland_clean_full.parquet")
    df_reduced = pd.read_parquet(data_dir / "poland_clean_reduced.parquet")
    
    print(f"   Full dataset: {df_full.shape}")
    print(f"   Reduced dataset: {df_reduced.shape}")
    
    # Define models (unfitted)
    print("\n🤖 Defining models...")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ),
        
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                penalty='l2',
                C=1.0,
                class_weight='balanced',
                solver='liblinear',
                max_iter=200,
                random_state=42
            ))
        ])
    }
    
    # Prepare datasets for each model
    # RF uses full data, Logit uses reduced
    print("\n🔄 Running cross-horizon evaluation...")
    print("   This may take several minutes...\n")
    
    # RF on full data
    print("🌲 Random Forest (full features)...")
    rf_results = full_cross_horizon_matrix(
        models_dict={'Random Forest': models['Random Forest']},
        df_full=df_full,
        horizons=[1, 2, 3],  # Start with first 3 horizons
        output_path=results_dir / "rf_cross_horizon_results.csv"
    )
    
    # Logit on reduced data
    print("\n📊 Logistic Regression (reduced features)...")
    logit_results = full_cross_horizon_matrix(
        models_dict={'Logistic Regression': models['Logistic Regression']},
        df_full=df_reduced,
        horizons=[1, 2, 3],
        output_path=results_dir / "logit_cross_horizon_results.csv"
    )
    
    # Combine results
    all_results = pd.concat([rf_results, logit_results], ignore_index=True)
    all_results.to_csv(results_dir / "all_cross_horizon_results.csv", index=False)
    
    # Generate heatmaps
    print("\n📊 Generating heatmaps...")
    
    for model_name in ['Random Forest', 'Logistic Regression']:
        plot_cross_horizon_heatmap(
            all_results,
            model_name=model_name,
            metric='roc_auc',
            output_path=results_dir / f"heatmap_{model_name.lower().replace(' ', '_')}_roc.png"
        )
        
        plot_cross_horizon_heatmap(
            all_results,
            model_name=model_name,
            metric='pr_auc',
            output_path=results_dir / f"heatmap_{model_name.lower().replace(' ', '_')}_pr.png"
        )
    
    # Summarize degradation
    print("\n📉 Analyzing metric degradation...")
    
    degradation = summarize_degradation(all_results)
    degradation.to_csv(results_dir / "performance_degradation_summary.csv", index=False)
    
    print("\n📊 Summary Statistics:")
    print("\nAverage ROC-AUC drop when transferring across horizons:")
    print(degradation.groupby('model')['roc_auc_drop'].mean().round(3))
    
    print("\nAverage PR-AUC drop when transferring across horizons:")
    print(degradation.groupby('model')['pr_auc_drop'].mean().round(3))
    
    print("\n" + "="*70)
    print("  ✅ CROSS-HORIZON ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {results_dir}")
    print(f"   - Full results CSV")
    print(f"   - Heatmaps for each model")
    print(f"   - Performance degradation summary")
    print()


if __name__ == '__main__':
    main()
