#!/usr/bin/env python3
"""
Script 02b: VIF Multicollinearity Remediation - Taiwan Dataset

MUST RUN BEFORE MODELING (Script 03+)!

Calculates VIF for all features and selects subset with low multicollinearity.
Creates remediated dataset for all subsequent modeling scripts.

Author: Generated
Date: November 13, 2025
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

print("=" * 80)
print("SCRIPT 02b: VIF MULTICOLLINEARITY REMEDIATION - TAIWAN DATASET")
print("=" * 80)
print("\n⚠️  THIS MUST RUN BEFORE MODELING (Script 03+)!")
print("   Creates VIF-selected feature set for all downstream analysis\n")

# Setup
data_dir = PROJECT_ROOT / 'data' / 'processed' / 'taiwan'
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / '03_taiwan' / '02b_vif_remediation'
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("[1/6] Loading Taiwan dataset...")
df = pd.read_parquet(data_dir / 'taiwan_clean.parquet')

# Get features and target
feature_cols = [c for c in df.columns if c.startswith('F')]
X = df[feature_cols].copy()
y = df['bankrupt']

print(f"  Dataset: {len(df):,} samples")
print(f"  Features: {len(feature_cols)}")
print(f"  Bankruptcy rate: {y.mean():.2%}")
print(f"  Events: {y.sum():,}")

# Calculate VIF for ALL features
print("\n[2/6] Calculating VIF for ALL features...")

def calculate_vif_all(X):
    """Calculate VIF for all features."""
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_with_const = add_constant(X_clean)
    
    vif_data = []
    for i, col in enumerate(X_clean.columns):
        try:
            vif = variance_inflation_factor(X_with_const.values, i + 1)
            vif_data.append({'feature': col, 'vif': vif})
        except:
            vif_data.append({'feature': col, 'vif': np.nan})
    
    return pd.DataFrame(vif_data).sort_values('vif', ascending=False)

vif_all = calculate_vif_all(X)

print("\n  Top 10 HIGHEST VIF:")
for _, row in vif_all.head(10).iterrows():
    status = "❌ HIGH" if row['vif'] > 10 else "⚠️ MODERATE" if row['vif'] > 5 else "✅ LOW"
    print(f"    {row['feature']:10s}: VIF = {row['vif']:8.2f} {status}")

# Count by VIF threshold
low_vif = (vif_all['vif'] < 5).sum()
moderate_vif = ((vif_all['vif'] >= 5) & (vif_all['vif'] < 10)).sum()
high_vif = (vif_all['vif'] >= 10).sum()

print(f"\n  VIF Distribution:")
print(f"    ✅ LOW (< 5.0):      {low_vif}/{len(vif_all)} features")
print(f"    ⚠️ MODERATE (5-10):  {moderate_vif}/{len(vif_all)} features")
print(f"    ❌ HIGH (≥ 10):      {high_vif}/{len(vif_all)} features")

# Select features with VIF < 5
print("\n[3/6] Method 1: VIF Selection (VIF < 5.0)...")
low_vif_features = vif_all[vif_all['vif'] < 5]['feature'].tolist()

print(f"  Selected: {len(low_vif_features)} features")

if low_vif_features:
    print(f"  Features: {', '.join(low_vif_features[:20])}{'...' if len(low_vif_features) > 20 else ''}")
    
    # Check EPV
    n_events = y.sum()
    epv_vif = n_events / len(low_vif_features)
    print(f"\n  EPV (Events Per Variable): {epv_vif:.2f}")
    print(f"  Status: {'✅ GOOD (≥10)' if epv_vif >= 10 else '⚠️ LOW (<10)'}")
else:
    print("  ⚠️ NO features with VIF < 5.0!")
    print("  Will try iterative VIF removal...")

# Iterative VIF removal (if needed)
if not low_vif_features or epv_vif < 10:
    print("\n[4/6] Method 2: Iterative VIF Removal...")
    
    X_temp = X.copy()
    removed = []
    iteration = 0
    target_features = max(10, int(y.sum() / 10))  # Aim for EPV ≥ 10
    
    while len(X_temp.columns) > target_features:
        vif_iter = calculate_vif_all(X_temp)
        max_vif = vif_iter['vif'].max()
        
        if max_vif < 5:
            break
        
        worst_feature = vif_iter.iloc[0]['feature']
        X_temp = X_temp.drop(columns=[worst_feature])
        removed.append((worst_feature, max_vif))
        iteration += 1
        
        if iteration % 10 == 0:
            print(f"    Iteration {iteration}: {len(X_temp.columns)} features remaining, max VIF = {max_vif:.1f}")
    
    low_vif_features = X_temp.columns.tolist()
    print(f"\n  Removed: {len(removed)} features")
    print(f"  Remaining: {len(low_vif_features)} features")
    vif_final = calculate_vif_all(X_temp)
    print(f"  Max VIF: {vif_final['vif'].max():.2f}")
    
    epv_vif = y.sum() / len(low_vif_features) if low_vif_features else 0
    print(f"  EPV: {epv_vif:.2f}")

# Forward selection
print(f"\n[5/6] Method 3: Forward Selection...")

def forward_selection(X, y, max_features=25):
    """Forward selection based on cross-validated AUC."""
    selected = []
    remaining = list(X.columns)
    best_score = 0
    
    print(f"  Starting forward selection (max {max_features} features)...")
    
    while remaining and len(selected) < max_features:
        scores = []
        for feat in remaining:
            test_features = selected + [feat]
            X_test = X[test_features].fillna(0)
            model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
            score = cross_val_score(model, X_test, y, cv=5, scoring='roc_auc', n_jobs=-1).mean()
            scores.append((feat, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        best_feat, score = scores[0]
        
        if score > best_score or len(selected) == 0:
            selected.append(best_feat)
            remaining.remove(best_feat)
            best_score = score
            if len(selected) <= 10 or len(selected) % 5 == 0:
                print(f"    {len(selected):2d}. {best_feat:10s}: AUC = {score:.4f}")
        else:
            break
    
    return selected

if low_vif_features:
    X_low_vif = X[low_vif_features]
    forward_features = forward_selection(X_low_vif, y, max_features=min(25, len(low_vif_features)))
    
    forward_epv = y.sum() / len(forward_features) if forward_features else 0
    print(f"\n  Selected: {len(forward_features)} features")
    print(f"  EPV: {forward_epv:.2f}")
else:
    forward_features = []
    print("  ⚠️ Skipped (no low VIF features)")

# Save results
print(f"\n[6/6] Saving results...")

# Save VIF values
vif_all.to_csv(output_dir / 'vif_all_features.csv', index=False)
print(f"  ✓ {output_dir / 'vif_all_features.csv'}")

# Save VIF-selected features
if low_vif_features:
    pd.DataFrame({'feature': low_vif_features}).to_csv(output_dir / 'vif_selected_features.csv', index=False)
    print(f"  ✓ {output_dir / 'vif_selected_features.csv'}")
    
    # Save VIF-selected dataset
    X_vif = X[low_vif_features]
    df_vif = pd.concat([X_vif, y], axis=1)
    df_vif.to_parquet(data_dir / 'taiwan_vif_selected.parquet')
    print(f"  ✓ {data_dir / 'taiwan_vif_selected.parquet'}")

# Save forward selection features
if forward_features:
    pd.DataFrame({'feature': forward_features}).to_csv(output_dir / 'forward_selected_features.csv', index=False)
    print(f"  ✓ {output_dir / 'forward_selected_features.csv'}")
    
    # Save forward-selected dataset
    X_forward = X[forward_features]
    df_forward = pd.concat([X_forward, y], axis=1)
    df_forward.to_parquet(data_dir / 'taiwan_forward_selected.parquet')
    print(f"  ✓ {data_dir / 'taiwan_forward_selected.parquet'}")

# Save summary
summary = {
    'dataset': 'Taiwan',
    'total_features': len(feature_cols),
    'low_vif_count': len(low_vif_features) if low_vif_features else 0,
    'moderate_vif_count': int(moderate_vif),
    'high_vif_count': int(high_vif),
    'forward_selection_count': len(forward_features) if forward_features else 0,
    'total_samples': len(df),
    'bankruptcy_events': int(y.sum()),
    'bankruptcy_rate': float(y.mean()),
    'epv_vif': float(epv_vif) if low_vif_features else 0,
    'epv_forward': float(forward_epv) if forward_features else 0,
    'methods': {
        'vif_selection': {
            'threshold': 5.0,
            'features': low_vif_features if low_vif_features else [],
            'count': len(low_vif_features) if low_vif_features else 0,
            'epv': float(epv_vif) if low_vif_features else 0
        },
        'forward_selection': {
            'features': forward_features if forward_features else [],
            'count': len(forward_features) if forward_features else 0,
            'epv': float(forward_epv) if forward_features else 0
        }
    }
}

import json
with open(output_dir / 'remediation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ {output_dir / 'remediation_summary.json'}")

# Final summary
print("\n" + "=" * 80)
print("TAIWAN DATASET - VIF REMEDIATION COMPLETE")
print("=" * 80)
print(f"""
Original: {len(feature_cols)} features

VIF Distribution:
  ✅ Low (< 5.0):      {len(low_vif_features) if low_vif_features else 0} features
  ⚠️ Moderate (5-10):  {moderate_vif} features  
  ❌ High (≥ 10):      {high_vif} features

Recommended Feature Sets:
  1. VIF-selected ({len(low_vif_features) if low_vif_features else 0} features): EPV = {epv_vif if low_vif_features else 0:.1f}
     File: taiwan_vif_selected.parquet
  
  2. Forward-selected ({len(forward_features) if forward_features else 0} features): EPV = {forward_epv if forward_features else 0:.1f}
     File: taiwan_forward_selected.parquet

⚠️  IMPORTANT: Use remediated datasets for Scripts 03+ (modeling)!
""")
print("=" * 80)
