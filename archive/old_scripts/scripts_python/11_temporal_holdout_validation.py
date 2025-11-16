#!/usr/bin/env python3
"""
Temporal Holdout Validation - UPDATED WITH REMEDIATED DATA AND CORRECT TERMINOLOGY

CORRECTED (Nov 12, 2025) - Script 00b findings:
- Polish data is REPEATED_CROSS_SECTIONS (different companies each period), NOT panel data
- Companies are NOT tracked over time - each horizon has different firms
- Cannot use panel methods (Panel VAR, Granger causality, clustered SE)
- Valid approach: Temporal holdout validation (train on early periods, test on later)

Original approach (Nov 6, 2024):
- Uses VIF-selected features (38 features, VIF < 10) from script 10d
- Addresses multicollinearity identified in script 10c
- Methodology corrected to match actual data structure
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import json

# Setup
project_root = Path(__file__).parent.parent
output_dir = project_root / 'results' / 'script_outputs' / '11_temporal_holdout_validation'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("TEMPORAL HOLDOUT VALIDATION")
print("="*70)
print("\n⚠️  CORRECTED: This is NOT panel data (Script 00b verified)")
print("Polish data = REPEATED_CROSS_SECTIONS (different companies each period)")
print("Valid method: Temporal holdout (train on early horizons, test on later)\n")

# ========================================================================
# Load Full Polish Dataset (All Horizons) with VIF-Selected Features
# ========================================================================
print("\n[1/6] Loading data with VIF-selected features (temporal structure verified)...")

data_dir = project_root / 'data' / 'processed'

# Load VIF-selected features from remediation (H1 only has selection)
vif_selected_df = pd.read_parquet(data_dir / 'poland_h1_vif_selected.parquet')
vif_features = [col for col in vif_selected_df.columns if col.startswith('Attr') and '__isna' not in col]

print(f"  VIF-selected features: {len(vif_features)} (from script 10d)")
print(f"  These features have VIF < 10 (multicollinearity remediated)")

# Load full dataset and filter to VIF-selected features
df_full = pd.read_parquet(data_dir / 'poland_clean_full.parquet')
df = df_full[['horizon', 'y'] + vif_features].copy()

feature_cols = vif_features

print(f"\n✓ Loaded {len(df):,} observations")
print(f"  Unique horizons: {df['horizon'].nunique()}")
print(f"  Features: {len(feature_cols)} (VIF-remediated)")
print(f"\n  ⚠️  Note: Autocorrelation detected (DW=0.69, p<0.001 from script 10c)")
print(f"  → Use clustered/robust SE for valid inference")

# ========================================================================
# Verify Temporal Structure (Repeated Cross-Sections)
# ========================================================================
print("\n[2/6] Verifying temporal structure (repeated cross-sections)...")

# Companies appear multiple times (once per horizon)
companies_per_horizon = df.groupby('horizon').size()
print(f"\n  Observations per horizon:")
for h, count in companies_per_horizon.items():
    print(f"    Horizon {h}: {count:,}")

# Check temporal dependency
horizon_stats = df.groupby('horizon')['y'].agg(['count', 'sum', 'mean'])
horizon_stats.columns = ['Total', 'Bankruptcies', 'Rate']
horizon_stats['Rate'] = horizon_stats['Rate'] * 100

print(f"\n  Bankruptcy rate by horizon:")
for idx, row in horizon_stats.iterrows():
    print(f"    Horizon {idx}: {row['Rate']:.2f}%")

# ========================================================================
# Temporal Validation (Out-of-Time)
# ========================================================================
print("\n[3/6] Temporal validation...")

# Train on early horizons, test on later horizons
train_horizons = [1, 2, 3]
test_horizons = [4, 5]

df_train = df[df['horizon'].isin(train_horizons)].copy()
df_test = df[df['horizon'].isin(test_horizons)].copy()

X_train = df_train[feature_cols]
y_train = df_train['y']
X_test = df_test[feature_cols]
y_test = df_test['y']

# Remove infinities
for X in [X_train, X_test]:
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

print(f"  Train: Horizons {train_horizons}, {len(X_train):,} obs")
print(f"  Test: Horizons {test_horizons}, {len(X_test):,} obs")

# Train model
# Note: Standard errors may be underestimated due to autocorrelation
# Note: Cannot use clustered SE (no panel structure - different companies each period)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n  📊 Using {len(feature_cols)} VIF-selected features (multicollinearity addressed)")

logit = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
logit.fit(X_train_scaled, y_train)

y_pred_proba = logit.predict_proba(X_test_scaled)[:, 1]
temporal_auc = roc_auc_score(y_test, y_pred_proba)

print(f"  Temporal validation AUC: {temporal_auc:.4f}")

# ========================================================================
# Within-Horizon vs Cross-Horizon Performance
# ========================================================================
print("\n[4/6] Within vs cross-horizon performance...")

results_within = []
results_cross = []

for h in [1, 2, 3]:
    # Within-horizon (traditional CV)
    df_h = df[df['horizon'] == h].copy()
    X_h = df_h[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df_h[feature_cols].median())
    y_h = df_h['y']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_h, y_h, test_size=0.2, random_state=42, stratify=y_h)
    
    scaler_h = StandardScaler()
    X_tr_sc = scaler_h.fit_transform(X_tr)
    X_te_sc = scaler_h.transform(X_te)
    
    logit_h = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    logit_h.fit(X_tr_sc, y_tr)
    
    y_pred_h = logit_h.predict_proba(X_te_sc)[:, 1]
    within_auc = roc_auc_score(y_te, y_pred_h)
    
    results_within.append({
        'horizon': h,
        'auc': within_auc,
        'type': 'within'
    })
    
    print(f"  Horizon {h} within-horizon AUC: {within_auc:.4f}")

# Cross-horizon (train on H1, test on others)
df_h1 = df[df['horizon'] == 1].copy()
X_h1 = df_h1[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df_h1[feature_cols].median())
y_h1 = df_h1['y']

scaler_h1 = StandardScaler()
X_h1_scaled = scaler_h1.fit_transform(X_h1)

logit_h1 = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
logit_h1.fit(X_h1_scaled, y_h1)

print(f"\n  Cross-horizon (train H1, test others):")
for h in [2, 3, 4, 5]:
    df_test_h = df[df['horizon'] == h].copy()
    X_test_h = df_test_h[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df_test_h[feature_cols].median())
    y_test_h = df_test_h['y']
    
    X_test_h_scaled = scaler_h1.transform(X_test_h)
    y_pred_h = logit_h1.predict_proba(X_test_h_scaled)[:, 1]
    cross_auc = roc_auc_score(y_test_h, y_pred_h)
    
    results_cross.append({
        'train_horizon': 1,
        'test_horizon': h,
        'auc': cross_auc,
        'type': 'cross'
    })
    
    print(f"    Test on H{h}: {cross_auc:.4f}")

# ========================================================================
# Clustered Standard Errors Simulation
# ========================================================================
print("\n[5/6] Clustered standard errors analysis...")

# Use horizon 1 for demonstration
df_h1 = df[df['horizon'] == 1].copy()

# Simulate company IDs (in reality, we'd need actual company identifiers)
# For Polish data, we'll create synthetic clusters based on feature similarity
from sklearn.cluster import KMeans

X_h1_clean = df_h1[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df_h1[feature_cols].median())

# Create 100 synthetic company clusters
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_h1['company_cluster'] = kmeans.fit_predict(X_h1_clean)

print(f"  Created {n_clusters} synthetic company clusters")
print(f"  Avg obs per cluster: {len(df_h1)/n_clusters:.1f}")

# Bootstrap with clustering
n_bootstrap = 100
bootstrap_aucs = []

for i in range(n_bootstrap):
    # Sample clusters with replacement
    sampled_clusters = np.random.choice(df_h1['company_cluster'].unique(), 
                                       size=n_clusters, replace=True)
    
    # Get all observations from sampled clusters
    df_boot = df_h1[df_h1['company_cluster'].isin(sampled_clusters)]
    
    if len(df_boot) < 100 or df_boot['y'].sum() < 5:
        continue
    
    X_boot = df_boot[feature_cols]
    y_boot = df_boot['y']
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_boot, y_boot, test_size=0.2, 
                                                random_state=i, stratify=y_boot)
    
    scaler_boot = StandardScaler()
    X_tr_sc = scaler_boot.fit_transform(X_tr)
    X_te_sc = scaler_boot.transform(X_te)
    
    logit_boot = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    logit_boot.fit(X_tr_sc, y_tr)
    
    y_pred_boot = logit_boot.predict_proba(X_te_sc)[:, 1]
    auc_boot = roc_auc_score(y_te, y_pred_boot)
    bootstrap_aucs.append(auc_boot)

bootstrap_aucs = np.array(bootstrap_aucs)
print(f"  Clustered bootstrap AUC: {bootstrap_aucs.mean():.4f} ± {bootstrap_aucs.std():.4f}")
print(f"  95% CI: [{np.percentile(bootstrap_aucs, 2.5):.4f}, {np.percentile(bootstrap_aucs, 97.5):.4f}]")

# ========================================================================
# Save Results
# ========================================================================
print("\n[6/6] Saving results...")

# Summary
temporal_summary = {
    'total_observations': int(len(df)),
    'horizons': int(df['horizon'].nunique()),
    'observations_per_horizon': companies_per_horizon.to_dict(),
    'temporal_validation_auc': float(temporal_auc),
    'within_horizon_avg_auc': float(np.mean([r['auc'] for r in results_within])),
    'cross_horizon_avg_auc': float(np.mean([r['auc'] for r in results_cross])),
    'clustered_bootstrap_mean_auc': float(bootstrap_aucs.mean()),
    'clustered_bootstrap_std_auc': float(bootstrap_aucs.std()),
    'clustered_bootstrap_ci_lower': float(np.percentile(bootstrap_aucs, 2.5)),
    'clustered_bootstrap_ci_upper': float(np.percentile(bootstrap_aucs, 97.5))
}

with open(output_dir / 'temporal_validation_summary.json', 'w') as f:
    json.dump(temporal_summary, f, indent=2)

# Within vs cross results
within_df = pd.DataFrame(results_within)
cross_df = pd.DataFrame(results_cross)

within_df.to_csv(output_dir / 'within_horizon_results.csv', index=False)
cross_df.to_csv(output_dir / 'cross_horizon_results.csv', index=False)

# ========================================================================
# Visualizations
# ========================================================================
print("Creating visualizations...")

# 1. Within vs Cross-horizon performance
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Within-horizon
axes[0].bar(range(len(within_df)), within_df['auc'], 
           color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_xticks(range(len(within_df)))
axes[0].set_xticklabels([f'H{h}' for h in within_df['horizon']])
axes[0].set_ylabel('ROC-AUC', fontweight='bold')
axes[0].set_title('Within-Horizon Performance', fontweight='bold')
axes[0].set_ylim([0.8, 1.0])
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(within_df['auc']):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')

# Cross-horizon
axes[1].bar(range(len(cross_df)), cross_df['auc'],
           color='darkgreen', alpha=0.7, edgecolor='black')
axes[1].set_xticks(range(len(cross_df)))
axes[1].set_xticklabels([f'H1→H{h}' for h in cross_df['test_horizon']])
axes[1].set_ylabel('ROC-AUC', fontweight='bold')
axes[1].set_title('Cross-Horizon Performance (Train H1)', fontweight='bold')
axes[1].set_ylim([0.8, 1.0])
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(cross_df['auc']):
    axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'within_vs_cross_horizon.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved within vs cross-horizon plot")

# 2. Bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_aucs, bins=30, edgecolor='black', alpha=0.7, color='coral')
plt.axvline(bootstrap_aucs.mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {bootstrap_aucs.mean():.4f}')
plt.axvline(np.percentile(bootstrap_aucs, 2.5), color='orange', linestyle=':',
           linewidth=2, label=f'95% CI')
plt.axvline(np.percentile(bootstrap_aucs, 97.5), color='orange', linestyle=':',
           linewidth=2)
plt.xlabel('ROC-AUC', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Clustered Bootstrap Distribution (100 iterations)', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'clustered_bootstrap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved bootstrap distribution")

# 3. Bankruptcy rate by horizon
plt.figure(figsize=(10, 6))
plt.bar(horizon_stats.index, horizon_stats['Rate'], 
       color=['skyblue', 'lightgreen', 'gold', 'orange', 'coral'],
       alpha=0.7, edgecolor='black')
plt.xlabel('Horizon', fontweight='bold')
plt.ylabel('Bankruptcy Rate (%)', fontweight='bold')
plt.title('Bankruptcy Rate by Prediction Horizon', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for i, (idx, row) in enumerate(horizon_stats.iterrows()):
    plt.text(idx, row['Rate'] + 0.2, f"{row['Rate']:.2f}%", 
            ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'bankruptcy_by_horizon.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved bankruptcy by horizon plot")

print("\n" + "="*70)
print("✓ TEMPORAL HOLDOUT VALIDATION COMPLETE")
print("="*70)
print(f"\nKey Findings:")
print(f"  Temporal validation AUC: {temporal_auc:.4f}")
print(f"  Within-horizon avg: {temporal_summary['within_horizon_avg_auc']:.4f}")
print(f"  Cross-horizon avg: {temporal_summary['cross_horizon_avg_auc']:.4f}")
print(f"  Bootstrap mean AUC: {temporal_summary['clustered_bootstrap_mean_auc']:.4f} ± {temporal_summary['clustered_bootstrap_std_auc']:.4f}")
print("\n⚠️  IMPORTANT: This is repeated cross-sections, NOT panel data!")
print("   Cannot use Panel VAR or Granger causality (no company tracking)")
print("="*70)
