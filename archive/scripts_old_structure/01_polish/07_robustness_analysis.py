#!/usr/bin/env python3
"""Cross-Horizon Robustness Analysis - ALL 5 HORIZONS"""

import sys
from pathlib import Path
# Add project root to path (scripts/01_polish -> root is 2 levels up)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.bankruptcy_prediction.data import DataLoader

# Setup
# PROJECT_ROOT already defined as PROJECT_ROOT above
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / '07_robustness_analysis'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("SCRIPT 07: Cross-Horizon Robustness Analysis")
print("="*70)

loader = DataLoader()

print("\n[1/4] Loading data for ALL 5 horizons...")
df_all = loader.load_poland(horizon=None, dataset_type='full')
print(f"✓ Total samples: {len(df_all):,}")
print(f"  Horizons: {sorted(df_all['horizon'].unique())}")

# Prepare data for each horizon
horizon_data = {}
for h in [1, 2, 3, 4, 5]:
    df_h = df_all[df_all['horizon'] == h].copy()
    X, y = loader.get_features_target(df_h)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    horizon_data[h] = {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test
    }
    print(f"  H={h}: {len(y_train):,} train, {len(y_test):,} test ({y_test.mean():.2%} bankrupt)")

print("\n[2/4] Running cross-horizon validation...")
print("Training on each horizon, testing on all others...\n")

rf_results = []
for train_h in [1, 2, 3, 4, 5]:
    print(f"Training on horizon {train_h}...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(horizon_data[train_h]['X_train'], horizon_data[train_h]['y_train'])
    
    for test_h in [1, 2, 3, 4, 5]:
        y_pred = rf.predict_proba(horizon_data[test_h]['X_test'])[:, 1]
        y_true = horizon_data[test_h]['y_test']
        
        roc_auc = roc_auc_score(y_true, y_pred)
        pr_auc = average_precision_score(y_true, y_pred)
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        idx_1pct = np.where(fpr <= 0.01)[0]
        recall_1pct = tpr[idx_1pct[-1]] if len(idx_1pct) > 0 else 0.0
        
        rf_results.append({
            'train_horizon': train_h,
            'test_horizon': test_h,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'recall_1pct_fpr': recall_1pct
        })
    print(f"  ✓ Tested on all 5 horizons")

rf_results_df = pd.DataFrame(rf_results)
rf_results_df.to_csv(output_dir / 'cross_horizon_results.csv', index=False)

print("\n[3/4] Analyzing performance degradation...")
# Create matrix
rf_matrix = rf_results_df.pivot(index='train_horizon', columns='test_horizon', values='roc_auc')
print("\nCross-Horizon Performance Matrix (ROC-AUC):")
print(rf_matrix.round(4))

# Calculate degradation
degradation = []
for train_h in [1, 2, 3, 4, 5]:
    same_horizon = rf_matrix.loc[train_h, train_h]
    for test_h in [1, 2, 3, 4, 5]:
        if test_h != train_h:
            cross_horizon = rf_matrix.loc[train_h, test_h]
            drop = same_horizon - cross_horizon
            drop_pct = (drop / same_horizon) * 100
            degradation.append({
                'train_horizon': train_h,
                'test_horizon': test_h,
                'same_horizon_auc': same_horizon,
                'cross_horizon_auc': cross_horizon,
                'absolute_drop': drop,
                'percent_drop': drop_pct
            })

degrad_df = pd.DataFrame(degradation)
degrad_df.to_csv(output_dir / 'performance_degradation.csv', index=False)

print(f"\nPerformance Degradation:")
print(f"  Average AUC drop: {degrad_df['absolute_drop'].mean():.4f} ({degrad_df['percent_drop'].mean():.2f}%)")
print(f"  Max AUC drop: {degrad_df['absolute_drop'].max():.4f} ({degrad_df['percent_drop'].max():.2f}%)")
print(f"  Min AUC drop: {degrad_df['absolute_drop'].min():.4f} ({degrad_df['percent_drop'].min():.2f}%)")

print("\n[4/4] Creating heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(rf_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0.75, vmax=1.0, cbar_kws={'label': 'ROC-AUC'})
plt.xlabel('Test Horizon', fontweight='bold')
plt.ylabel('Train Horizon', fontweight='bold')
plt.title('Cross-Horizon Performance - Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'cross_horizon_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved heatmap")

# Save summary
summary = {
    'total_experiments': len(rf_results_df),
    'horizons_tested': 5,
    'avg_same_horizon_auc': float(rf_matrix.values.diagonal().mean()),
    'avg_cross_horizon_auc': float(degrad_df['cross_horizon_auc'].mean()),
    'avg_degradation_pct': float(degrad_df['percent_drop'].mean()),
    'max_degradation_pct': float(degrad_df['percent_drop'].max()),
    'min_degradation_pct': float(degrad_df['percent_drop'].min()),
}

import json
with open(output_dir / 'robustness_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("✓ SCRIPT 07 COMPLETE - ALL 5 HORIZONS TESTED")
print(f"  Same horizon avg: {summary['avg_same_horizon_auc']:.4f}")
print(f"  Cross horizon avg: {summary['avg_cross_horizon_auc']:.4f}")
print(f"  Avg degradation: {summary['avg_degradation_pct']:.2f}%")
print("="*70)
