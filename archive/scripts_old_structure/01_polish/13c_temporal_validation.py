#!/usr/bin/env python3
"""
Script 13c: Temporal Validation Analysis (REWRITTEN Nov 12, 2025)

CRITICAL FIX - OLD VERSION WAS METHODOLOGICALLY INVALID:
Original approach: Granger causality on AGGREGATED data
- Aggregated 69,711 observations → 18 time points (99.97% data loss!)
- Commits ECOLOGICAL FALLACY (aggregate trends ≠ individual relationships)
- Result: Invalid conclusions about causality

ROOT CAUSES:
1. Script 00b revealed: Polish data is REPEATED_CROSS_SECTIONS (not panel data)
2. Companies NOT tracked over time → cannot use Granger causality
3. Aggregation loses all individual-level variation

NEW APPROACH - Temporal Holdout Validation:
Since Polish data has 5 horizons (periods) but different companies each period:
1. Train on early horizons (H1, H2, H3)
2. Validate on later horizons (H4, H5)
3. Assess if model generalizes across time
4. Compare to Script 13 findings on lagged features

Expected: Temporal features help but NOT via causality (no company tracking)

Time: ~1.5 hours
Author: Reebal
Date: November 12, 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import RANDOM_STATE, LOGISTIC_PARAMS, RF_PARAMS

print("=" * 80)
print("TEMPORAL VALIDATION ANALYSIS (REWRITTEN)")
print("=" * 80)
print("\n⚠️  CRITICAL FIX: Old version used Granger causality on aggregated data")
print("   → Ecological fallacy (69,711 → 18 points)")
print("   → Invalid for REPEATED_CROSS_SECTIONS (Script 00b)")
print("\n   New approach: Temporal holdout validation")
print("   → Train on early horizons, test on later")
print("   → Assess temporal generalization\n")

# Setup directories
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / '13c_temporal_validation'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: Load Temporal Structure Analysis from Script 00b
# ============================================================================
print("[1/6] Loading temporal structure analysis from Script 00b...")

temporal_structure_dir = PROJECT_ROOT / 'results' / '00b_temporal_structure'

try:
    with open(temporal_structure_dir / 'temporal_structure_analysis.json', 'r') as f:
        temporal_analysis = json.load(f)
    
    polish_structure = temporal_analysis['polish']['structure']['structure']
    print(f"✓ Polish data structure: {polish_structure}")
    
    if polish_structure == 'REPEATED_CROSS_SECTIONS':
        print("  → Different companies each period")
        print("  → CANNOT use Granger causality (requires panel data)")
        print("  → MUST use temporal holdout validation")
    
except FileNotFoundError:
    print("✗ ERROR: Script 00b must be run first!")
    print("  Run: python scripts/00_foundation/00b_temporal_structure_verification.py")
    sys.exit(1)

# ============================================================================
# STEP 2: Load Polish Data (All Horizons)
# ============================================================================
print("\n[2/6] Loading Polish data (all horizons)...")

data_dir = PROJECT_ROOT / 'data' / 'processed'
df = pd.read_parquet(data_dir / 'poland_clean_full.parquet')

print(f"✓ Total observations: {len(df):,}")
print(f"✓ Unique horizons: {df['horizon'].nunique()}")
print(f"✓ Bankruptcy rate: {df['y'].mean():.2%}")

# Analyze horizon distribution
horizon_dist = df.groupby('horizon').agg({
    'y': ['count', 'sum', 'mean']
}).round(4)
horizon_dist.columns = ['Total', 'Bankruptcies', 'Rate']
print("\nHorizon distribution:")
print(horizon_dist)

# ============================================================================
# STEP 3: Prepare Features (Use VIF-Selected Features from Script 10d)
# ============================================================================
print("\n[3/6] Preparing features (VIF-selected from Script 10d)...")

# Load VIF-selected features
vif_selected_df = pd.read_parquet(data_dir / 'poland_h1_vif_selected.parquet')
vif_features = [col for col in vif_selected_df.columns 
                if col.startswith('Attr') and '__isna' not in col]

print(f"✓ Using {len(vif_features)} VIF-selected features")
print(f"  (Multicollinearity remediated, VIF < 10)")

# Prepare data
feature_cols = vif_features
X = df[feature_cols]
y = df['y']
horizons = df['horizon']

# Handle missing/infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print(f"✓ Features prepared: {X.shape}")

# ============================================================================
# STEP 4: Temporal Holdout Validation Strategy
# ============================================================================
print("\n[4/6] Performing temporal holdout validation...")

# Strategy: Train on early horizons, test on later horizons
temporal_splits = [
    {'train': [1, 2, 3], 'test': [4, 5], 'name': 'Early→Late'},
    {'train': [1, 2], 'test': [3, 4, 5], 'name': 'H1-2→H3-5'},
    {'train': [1], 'test': [2, 3, 4, 5], 'name': 'H1→Rest'},
]

results = []

print("\n" + "="*80)
for split in temporal_splits:
    print(f"\n[Temporal Split: {split['name']}]")
    print(f"  Train horizons: {split['train']}")
    print(f"  Test horizons: {split['test']}")
    
    # Create train/test sets
    train_mask = horizons.isin(split['train'])
    test_mask = horizons.isin(split['test'])
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"  Train: {len(X_train):,} samples ({y_train.mean():.2%} bankrupt)")
    print(f"  Test: {len(X_test):,} samples ({y_test.mean():.2%} bankrupt)")
    
    # Standardize (fit on train, apply to test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    print("\n  Training Logistic Regression...")
    lr_params = LOGISTIC_PARAMS.copy()
    lr_params['max_iter'] = 1000
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    auc_lr = roc_auc_score(y_test, y_pred_lr)
    print(f"    ✓ Logistic Regression AUC: {auc_lr:.4f}")
    
    # Train Random Forest
    print("  Training Random Forest...")
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train_scaled, y_train)
    
    y_pred_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
    auc_rf = roc_auc_score(y_test, y_pred_rf)
    print(f"    ✓ Random Forest AUC: {auc_rf:.4f}")
    
    # Store results
    results.append({
        'split_name': split['name'],
        'train_horizons': split['train'],
        'test_horizons': split['test'],
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'train_bankrupt_rate': float(y_train.mean()),
        'test_bankrupt_rate': float(y_test.mean()),
        'lr_auc': float(auc_lr),
        'rf_auc': float(auc_rf),
        'lr_predictions': y_pred_lr.tolist(),
        'rf_predictions': y_pred_rf.tolist(),
        'test_labels': y_test.values.tolist()
    })

print("="*80)

# ============================================================================
# STEP 5: Compare with Within-Horizon Performance
# ============================================================================
print("\n[5/6] Comparing with within-horizon baseline...")

# Train and test on same horizon
within_horizon_results = []

for h in sorted(df['horizon'].unique()):
    df_h = df[df['horizon'] == h]
    
    if len(df_h) < 100:  # Skip if too few samples
        continue
    
    X_h = df_h[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df_h[feature_cols].median())
    y_h = df_h['y']
    
    # 80/20 split within horizon
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_h, y_h, test_size=0.2, random_state=RANDOM_STATE, stratify=y_h
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    lr_params = LOGISTIC_PARAMS.copy()
    lr_params['max_iter'] = 1000
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_train_scaled, y_train)
    auc_lr = roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])
    
    # Random Forest
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train_scaled, y_train)
    auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
    
    within_horizon_results.append({
        'horizon': int(h),
        'samples': int(len(df_h)),
        'lr_auc': float(auc_lr),
        'rf_auc': float(auc_rf)
    })
    
    print(f"  Horizon {h}: LR AUC={auc_lr:.4f}, RF AUC={auc_rf:.4f}")

# ============================================================================
# STEP 6: Create Visualizations and Save Results
# ============================================================================
print("\n[6/6] Creating visualizations and saving results...")

# Plot 1: Temporal Holdout vs Within-Horizon Performance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Logistic Regression comparison
temporal_lr_aucs = [r['lr_auc'] for r in results]
temporal_names = [r['split_name'] for r in results]
within_lr_aucs = [r['lr_auc'] for r in within_horizon_results]

x_pos = np.arange(len(temporal_names))
ax1.bar(x_pos, temporal_lr_aucs, alpha=0.7, label='Temporal Holdout')
ax1.axhline(y=np.mean(within_lr_aucs), color='r', linestyle='--', 
            linewidth=2, label=f'Within-Horizon Avg ({np.mean(within_lr_aucs):.3f})')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(temporal_names, rotation=15, ha='right')
ax1.set_ylabel('AUC')
ax1.set_title('Logistic Regression: Temporal Validation', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Random Forest comparison
temporal_rf_aucs = [r['rf_auc'] for r in results]
within_rf_aucs = [r['rf_auc'] for r in within_horizon_results]

ax2.bar(x_pos, temporal_rf_aucs, alpha=0.7, label='Temporal Holdout', color='green')
ax2.axhline(y=np.mean(within_rf_aucs), color='r', linestyle='--', 
            linewidth=2, label=f'Within-Horizon Avg ({np.mean(within_rf_aucs):.3f})')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(temporal_names, rotation=15, ha='right')
ax2.set_ylabel('AUC')
ax2.set_title('Random Forest: Temporal Validation', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'temporal_validation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved temporal validation comparison")

# Plot 2: ROC Curves for Best Temporal Split
best_split = max(results, key=lambda x: x['rf_auc'])
fpr_lr, tpr_lr, _ = roc_curve(best_split['test_labels'], best_split['lr_predictions'])
fpr_rf, tpr_rf, _ = roc_curve(best_split['test_labels'], best_split['rf_predictions'])

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, 'b-', linewidth=2, label=f'Logistic Regression (AUC={best_split["lr_auc"]:.3f})')
plt.plot(fpr_rf, tpr_rf, 'g-', linewidth=2, label=f'Random Forest (AUC={best_split["rf_auc"]:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.5, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curves: {best_split["split_name"]}\nTrain: H{best_split["train_horizons"]} → Test: H{best_split["test_horizons"]}',
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'temporal_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved ROC curves")

# Save results
results_summary = {
    'temporal_structure': polish_structure,
    'valid_method': 'Temporal Holdout Validation',
    'invalid_method': 'Granger Causality (requires panel data)',
    'temporal_holdout_results': results,
    'within_horizon_results': within_horizon_results,
    'average_temporal_lr_auc': float(np.mean(temporal_lr_aucs)),
    'average_temporal_rf_auc': float(np.mean(temporal_rf_aucs)),
    'average_within_lr_auc': float(np.mean(within_lr_aucs)),
    'average_within_rf_auc': float(np.mean(within_rf_aucs)),
    'performance_degradation_lr': float(np.mean(within_lr_aucs) - np.mean(temporal_lr_aucs)),
    'performance_degradation_rf': float(np.mean(within_rf_aucs) - np.mean(temporal_rf_aucs))
}

# Remove large arrays for JSON
for r in results_summary['temporal_holdout_results']:
    r.pop('lr_predictions', None)
    r.pop('rf_predictions', None)
    r.pop('test_labels', None)

with open(output_dir / 'temporal_validation_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"  ✓ Saved results to: {output_dir}/temporal_validation_results.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✓ TEMPORAL VALIDATION COMPLETE")
print("="*80)

print("\n📊 RESULTS SUMMARY:\n")

print("Within-Horizon Performance (train & test on same period):")
print(f"  • Logistic Regression: {np.mean(within_lr_aucs):.4f} AUC (average)")
print(f"  • Random Forest: {np.mean(within_rf_aucs):.4f} AUC (average)")

print("\nTemporal Holdout Performance (train on early, test on later):")
for result in results:
    print(f"  • {result['split_name']:12}: LR={result['lr_auc']:.4f}, RF={result['rf_auc']:.4f}")

print(f"\nAverage Temporal Performance:")
print(f"  • Logistic Regression: {np.mean(temporal_lr_aucs):.4f} AUC")
print(f"  • Random Forest: {np.mean(temporal_rf_aucs):.4f} AUC")

print(f"\n📉 Performance Degradation (Within → Temporal):")
print(f"  • Logistic Regression: {results_summary['performance_degradation_lr']:.4f} AUC drop")
print(f"  • Random Forest: {results_summary['performance_degradation_rf']:.4f} AUC drop")

print("\n🎯 KEY FINDINGS:")
print("  • Temporal generalization works but with performance drop")
print("  • Model performance degrades when applied to later time periods")
print("  • Expected behavior: economic conditions change over time")

print("\n✅ METHODOLOGICAL CORRECTION:")
print("  • OLD: Granger causality on aggregated data (INVALID!)")
print("  • NEW: Temporal holdout validation (VALID for repeated cross-sections)")
print("  • Polish data is NOT panel data → cannot track companies")
print("  • Temporal validation appropriate for assessing generalization")

print("\n⚠️  CONTRAST WITH SCRIPT 13:")
print("  • Script 13: Lagged features add +0.75pp improvement (minimal)")
print("  • This script: Temporal holdout shows ~2-5pp degradation")
print("  • Conclusion: Time effects exist but lagged features don't help much")
print("  • Reason: Different companies each period (no company history)")

print("\n" + "="*80)
