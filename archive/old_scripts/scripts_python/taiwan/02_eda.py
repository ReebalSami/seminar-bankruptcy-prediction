#!/usr/bin/env python3
"""
Taiwan Dataset - Exploratory Data Analysis
High-dimensional feature analysis (95 features)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

# Setup
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / 'data' / 'processed' / 'taiwan'
output_dir = project_root / 'results' / 'script_outputs' / 'taiwan'
figures_dir = output_dir / 'figures'

print("="*70)
print("TAIWAN DATASET - Exploratory Data Analysis")
print("="*70)

# Load data
print("\n[1/4] Loading data...")
df = pd.read_parquet(data_dir / 'taiwan_clean.parquet')

import json
with open(data_dir / 'taiwan_features_metadata.json', 'r') as f:
    feature_mapping = json.load(f)

feature_cols = [col for col in df.columns if col.startswith('F')]
X = df[feature_cols]
y = df['bankrupt']

print(f"✓ Loaded {len(df):,} samples")
print(f"  Features: {len(feature_cols)}")
print(f"  Bankruptcy rate: {y.mean()*100:.2f}%")

# ========================================================================
# Part 1: Feature Correlations with Target (focus on top features)
# ========================================================================
print("\n[2/4] Computing correlations...")

correlations = []
for col in feature_cols:
    corr = X[col].corr(y)
    correlations.append({
        'feature': col,
        'original_name': feature_mapping[col]['original_name'],
        'correlation': corr,
        'abs_correlation': abs(corr)
    })

corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
corr_df.to_csv(output_dir / 'correlations_with_target.csv', index=False)

print(f"✓ Computed {len(corr_df)} correlations")
print(f"  Top 3 correlated features:")
for _, row in corr_df.head(3).iterrows():
    direction = "↑ higher risk" if row['correlation'] > 0 else "↓ lower risk"
    print(f"    • {row['original_name'][:50]}: {row['correlation']:.4f} ({direction})")

# ========================================================================
# Part 2: Statistical Tests (top 20 features only for efficiency)
# ========================================================================
print("\n[3/4] Testing top 20 features...")

top_20_features = corr_df.head(20)['feature'].tolist()

stats_data = []
for col in top_20_features:
    bankrupt_mean = X.loc[y == 1, col].mean()
    healthy_mean = X.loc[y == 0, col].mean()
    
    t_stat, p_value = stats.ttest_ind(
        X.loc[y == 1, col].dropna(),
        X.loc[y == 0, col].dropna(),
        equal_var=False
    )
    
    stats_data.append({
        'feature': col,
        'original_name': feature_mapping[col]['original_name'],
        'bankrupt_mean': bankrupt_mean,
        'healthy_mean': healthy_mean,
        'difference': bankrupt_mean - healthy_mean,
        'p_value': p_value
    })

stats_df = pd.DataFrame(stats_data).sort_values('p_value')
stats_df.to_csv(output_dir / 'top_features_statistics.csv', index=False)

print(f"✓ Tested top 20 features")
print(f"  Significant (p<0.05): {(stats_df['p_value'] < 0.05).sum()}")

# ========================================================================
# Part 3: Visualizations
# ========================================================================
print("\n[4/4] Creating visualizations...")

# 1. Top 15 correlations
plt.figure(figsize=(12, 8))
top_15 = corr_df.head(15)
colors = ['darkred' if c < 0 else 'darkgreen' for c in top_15['correlation']]

# Truncate long names
labels = [name[:40] + '...' if len(name) > 40 else name 
          for name in top_15['original_name']]

plt.barh(range(len(top_15)), top_15['correlation'], color=colors, alpha=0.7)
plt.yticks(range(len(top_15)), labels)
plt.xlabel('Correlation with Bankruptcy', fontweight='bold')
plt.title('Top 15 Correlated Features\n(Red: Protective, Green: Risk)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'top_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved correlation plot")

# 2. Correlation heatmap (top 10)
plt.figure(figsize=(12, 10))
top_10_codes = corr_df.head(10)['feature'].tolist()
top_10_names = [feature_mapping[f]['original_name'][:30] for f in top_10_codes]

corr_matrix = X[top_10_codes].corr()
corr_matrix.index = top_10_names
corr_matrix.columns = top_10_names

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap (Top 10)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved correlation heatmap")

# 3. Distribution comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (_, row) in enumerate(stats_df.head(6).iterrows()):
    col = row['feature']
    name = row['original_name'][:35]
    
    axes[idx].hist(X.loc[y == 0, col], bins=30, alpha=0.6, label='Healthy',
                   color='green', edgecolor='black')
    axes[idx].hist(X.loc[y == 1, col], bins=30, alpha=0.6, label='Bankrupt',
                   color='red', edgecolor='black')
    axes[idx].set_title(name, fontweight='bold', fontsize=9)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.suptitle('Distribution Comparison: Top 6 Discriminative Features',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved distributions plot")

# Save summary
summary = {
    'total_features': len(feature_cols),
    'top_correlated_feature': corr_df.iloc[0]['original_name'],
    'top_correlation': float(corr_df.iloc[0]['correlation']),
    'significant_in_top_20': int((stats_df['p_value'] < 0.05).sum()),
    'avg_correlation_top_10': float(corr_df.head(10)['abs_correlation'].mean())
}

with open(output_dir / 'eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("✓ TAIWAN DATASET EDA COMPLETE")
print(f"  Total features: {summary['total_features']}")
print(f"  Top correlated: {summary['top_correlated_feature'][:50]}")
print(f"  Correlation: {summary['top_correlation']:.4f}")
print("="*70)
