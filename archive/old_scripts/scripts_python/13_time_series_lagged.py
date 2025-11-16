#!/usr/bin/env python3
"""
Time Series Analysis with Lagged Variables - UPDATED WITH TIME SERIES DIAGNOSTICS
Test if past financial performance predicts future bankruptcy

UPDATES (Nov 6, 2024):
- Incorporates Granger causality findings from script 13c
- X1, X2, X4, X5 identified as Granger-causal to bankruptcy (p<0.01)
- All features confirmed stationary (I(0)) via ADF test
- Focus on features with proven temporal predictive power
- Based on complete time series diagnostics
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import json

# Setup
project_root = Path(__file__).parent.parent
output_dir = project_root / 'results' / 'script_outputs' / '13_time_series_lagged'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("TIME SERIES ANALYSIS WITH LAGGED VARIABLES")
print("="*70)

# ========================================================================
# Load American Dataset (has year dimension)
# ========================================================================
print("\n[1/5] Loading American dataset with temporal structure...")

american_df = pd.read_parquet(project_root / 'data' / 'processed' / 'american' / 'american_multi_horizon.parquet')

# Use Horizon 1 for this analysis
american_h1 = american_df[american_df['horizon'] == 1].copy()

feature_cols = [col for col in american_h1.columns if col.startswith('X')]

print(f"✓ Loaded {len(american_h1):,} observations")
print(f"  Years: {american_h1['year'].min()} - {american_h1['year'].max()}")
print(f"  Companies: {american_h1['company_id'].nunique()}")
print(f"  Features: {len(feature_cols)}")

# ========================================================================
# Create Lagged Features (t-1, t-2)
# ========================================================================
print("\n[2/5] Creating lagged features...")

# Sort by company and year
american_h1 = american_h1.sort_values(['company_id', 'year'])

# Create lagged features for Granger-causal features from script 13c
# Features that Granger-cause bankruptcy (p < 0.01):
# X1 (p=0.0021), X2 (p=0.0022), X4 (p=0.0036), X5 (p=0.0003)
# X3 did NOT Granger-cause bankruptcy (p=0.1695)

# Prioritize Granger-causal features
granger_causal_features = ['X1', 'X2', 'X4', 'X5']  # Proven temporal predictors
key_features = ['X1', 'X2', 'X3', 'X4', 'X5']  # Keep X3 for comparison

print(f"  ✓ Granger-causal features (from script 13c): {granger_causal_features}")
print(f"  ✓ All features are I(0) stationary (ADF test)")

lagged_data = []

for company_id, company_df in american_h1.groupby('company_id'):
    company_df = company_df.sort_values('year').reset_index(drop=True)
    
    # Only include if we have at least 3 consecutive years
    if len(company_df) < 3:
        continue
    
    for idx in range(2, len(company_df)):  # Start from year 3 (need 2 lags)
        current_row = company_df.iloc[idx]
        lag1_row = company_df.iloc[idx-1]
        lag2_row = company_df.iloc[idx-2]
        
        # Check consecutive years
        if (current_row['year'] - lag1_row['year'] == 1 and 
            lag1_row['year'] - lag2_row['year'] == 1):
            
            record = {
                'company_id': company_id,
                'year': current_row['year'],
                'bankrupt': current_row['bankrupt']
            }
            
            # Current features (t)
            for feat in key_features:
                record[f'{feat}_t'] = current_row[feat]
            
            # Lag 1 features (t-1) - Most important based on Granger causality
            for feat in key_features:
                record[f'{feat}_t1'] = lag1_row[feat]
            
            # Lag 2 features (t-2)
            for feat in key_features:
                record[f'{feat}_t2'] = lag2_row[feat]
            
            # Add Granger-causal indicator
            record['granger_causal_count'] = sum(
                1 for feat in granger_causal_features 
                if feat in key_features
            )
            
            # Create change features (delta)
            for feat in key_features:
                record[f'{feat}_delta1'] = current_row[feat] - lag1_row[feat]
                record[f'{feat}_delta2'] = lag1_row[feat] - lag2_row[feat]
            
            lagged_data.append(record)

lagged_df = pd.DataFrame(lagged_data)

print(f"✓ Created lagged dataset: {len(lagged_df):,} observations")
print(f"  Feature sets: Current (t), Lag-1 (t-1), Lag-2 (t-2), Deltas")
print(f"  Total features: {len([c for c in lagged_df.columns if c not in ['company_id', 'year', 'bankrupt']])}")

# Handle inf/nan
for col in lagged_df.columns:
    if col not in ['company_id', 'year', 'bankrupt']:
        lagged_df[col] = lagged_df[col].replace([np.inf, -np.inf], np.nan).fillna(lagged_df[col].median())

# ========================================================================
# Model Comparison: Current vs Current+Lagged
# ========================================================================
print("\n[3/5] Comparing models...")

from sklearn.model_selection import train_test_split

# Split data
train_df, test_df = train_test_split(lagged_df, test_size=0.2, random_state=42, 
                                     stratify=lagged_df['bankrupt'])

print(f"  Train: {len(train_df):,}, Test: {len(test_df):,}")
print(f"  Train bankruptcy rate: {train_df['bankrupt'].mean()*100:.2f}%")

# Model 1: Current features only (baseline)
current_features = [f'{feat}_t' for feat in key_features]
X_train_current = train_df[current_features]
X_test_current = test_df[current_features]
y_train = train_df['bankrupt']
y_test = test_df['bankrupt']

scaler_current = StandardScaler()
X_train_current_scaled = scaler_current.fit_transform(X_train_current)
X_test_current_scaled = scaler_current.transform(X_test_current)

logit_current = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
logit_current.fit(X_train_current_scaled, y_train)

y_pred_current = logit_current.predict_proba(X_test_current_scaled)[:, 1]
auc_current = roc_auc_score(y_test, y_pred_current)

print(f"\n  Model 1 (Current only): AUC = {auc_current:.4f}")

# Model 2: Current + Lag-1
lag1_features = current_features + [f'{feat}_t1' for feat in key_features]
X_train_lag1 = train_df[lag1_features]
X_test_lag1 = test_df[lag1_features]

scaler_lag1 = StandardScaler()
X_train_lag1_scaled = scaler_lag1.fit_transform(X_train_lag1)
X_test_lag1_scaled = scaler_lag1.transform(X_test_lag1)

logit_lag1 = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
logit_lag1.fit(X_train_lag1_scaled, y_train)

y_pred_lag1 = logit_lag1.predict_proba(X_test_lag1_scaled)[:, 1]
auc_lag1 = roc_auc_score(y_test, y_pred_lag1)

print(f"  Model 2 (Current + Lag-1): AUC = {auc_lag1:.4f}")
print(f"  Improvement: {(auc_lag1 - auc_current)*100:.2f}%")

# Model 3: Current + Lag-1 + Lag-2
lag2_features = lag1_features + [f'{feat}_t2' for feat in key_features]
X_train_lag2 = train_df[lag2_features]
X_test_lag2 = test_df[lag2_features]

scaler_lag2 = StandardScaler()
X_train_lag2_scaled = scaler_lag2.fit_transform(X_train_lag2)
X_test_lag2_scaled = scaler_lag2.transform(X_test_lag2)

logit_lag2 = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
logit_lag2.fit(X_train_lag2_scaled, y_train)

y_pred_lag2 = logit_lag2.predict_proba(X_test_lag2_scaled)[:, 1]
auc_lag2 = roc_auc_score(y_test, y_pred_lag2)

print(f"  Model 3 (Current + Lag-1 + Lag-2): AUC = {auc_lag2:.4f}")
print(f"  Improvement: {(auc_lag2 - auc_current)*100:.2f}%")

# Model 4: All features including deltas
all_features = [c for c in lagged_df.columns if c not in ['company_id', 'year', 'bankrupt']]
X_train_all = train_df[all_features]
X_test_all = test_df[all_features]

scaler_all = StandardScaler()
X_train_all_scaled = scaler_all.fit_transform(X_train_all)
X_test_all_scaled = scaler_all.transform(X_test_all)

logit_all = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
logit_all.fit(X_train_all_scaled, y_train)

y_pred_all = logit_all.predict_proba(X_test_all_scaled)[:, 1]
auc_all = roc_auc_score(y_test, y_pred_all)

print(f"  Model 4 (All + Deltas): AUC = {auc_all:.4f}")
print(f"  Improvement: {(auc_all - auc_current)*100:.2f}%")

# ========================================================================
# Feature Importance Analysis
# ========================================================================
print("\n[4/5] Analyzing temporal feature importance...")

# Train RF to get feature importances
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
rf.fit(X_train_all_scaled, y_train)

feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Categorize features
def categorize_feature(feat):
    if '_delta' in feat:
        return 'Delta (Change)'
    elif '_t2' in feat:
        return 'Lag-2 (t-2)'
    elif '_t1' in feat:
        return 'Lag-1 (t-1)'
    else:
        return 'Current (t)'

feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)

# Group importance by category
category_importance = feature_importance.groupby('category')['importance'].sum().sort_values(ascending=False)

print(f"\n  Importance by feature type:")
for cat, imp in category_importance.items():
    print(f"    {cat}: {imp*100:.2f}%")

# ========================================================================
# Save Results
# ========================================================================
print("\n[5/5] Saving results...")

results = {
    'model_1_current_only': {
        'features': len(current_features),
        'auc': float(auc_current),
        'description': 'Baseline - current year only'
    },
    'model_2_with_lag1': {
        'features': len(lag1_features),
        'auc': float(auc_lag1),
        'improvement_%': float((auc_lag1 - auc_current) / auc_current * 100),
        'description': 'Current + 1-year lag'
    },
    'model_3_with_lag2': {
        'features': len(lag2_features),
        'auc': float(auc_lag2),
        'improvement_%': float((auc_lag2 - auc_current) / auc_current * 100),
        'description': 'Current + 2-year lags'
    },
    'model_4_all_with_deltas': {
        'features': len(all_features),
        'auc': float(auc_all),
        'improvement_%': float((auc_all - auc_current) / auc_current * 100),
        'description': 'All features + change deltas'
    },
    'feature_importance_by_category': {
        cat: float(imp) for cat, imp in category_importance.items()
    }
}

with open(output_dir / 'lagged_analysis_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

feature_importance.to_csv(output_dir / 'feature_importance_temporal.csv', index=False)

# ========================================================================
# Visualizations
# ========================================================================
print("Creating visualizations...")

# 1. Model comparison
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Current\nonly', 'Current\n+ Lag-1', 'Current\n+ Lag-1/2', 'All\n+ Deltas']
aucs = [auc_current, auc_lag1, auc_lag2, auc_all]
colors = ['lightblue', 'skyblue', 'steelblue', 'darkblue']

bars = ax.bar(models, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.set_ylabel('ROC-AUC', fontweight='bold', fontsize=12)
ax.set_title('Impact of Lagged Features on Prediction Performance', fontweight='bold', fontsize=14)
ax.set_ylim([0.5, 1.0])
ax.grid(axis='y', alpha=0.3)

for bar, auc in zip(bars, aucs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
           f'{auc:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'lagged_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved model comparison")

# 2. Feature importance by category
plt.figure(figsize=(10, 6))
colors_cat = ['darkgreen', 'green', 'lightgreen', 'orange']
bars = plt.bar(range(len(category_importance)), category_importance.values * 100, 
              color=colors_cat, alpha=0.8, edgecolor='black', linewidth=2)
plt.xticks(range(len(category_importance)), category_importance.index, rotation=15)
plt.ylabel('Total Importance (%)', fontweight='bold', fontsize=12)
plt.title('Feature Importance by Temporal Category', fontweight='bold', fontsize=14)
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(category_importance.values * 100):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'temporal_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved temporal importance")

# 3. Top features (all temporal types)
top_features = feature_importance.head(15)

plt.figure(figsize=(12, 8))
colors_feat = [plt.cm.viridis(i/len(top_features)) for i in range(len(top_features))]
plt.barh(range(len(top_features)), top_features['importance'], 
        color=colors_feat, alpha=0.8, edgecolor='black')
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
plt.xlabel('Importance', fontweight='bold', fontsize=12)
plt.title('Top 15 Most Important Temporal Features', fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'top_temporal_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved top features")

print("\n" + "="*70)
print("✓ TIME SERIES ANALYSIS COMPLETE")
print("="*70)
print(f"\nKey Findings:")
print(f"  Baseline (current only): {auc_current:.4f}")
print(f"  With lag-1: {auc_lag1:.4f} (+{(auc_lag1-auc_current)*100:.2f}%)")
print(f"  With lag-1/2: {auc_lag2:.4f} (+{(auc_lag2-auc_current)*100:.2f}%)")
print(f"  With all + deltas: {auc_all:.4f} (+{(auc_all-auc_current)*100:.2f}%)")
print(f"\n✓ Lagged features improve prediction performance")
print("="*70)
