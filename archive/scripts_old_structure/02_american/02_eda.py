#!/usr/bin/env python3
"""
American Dataset - Exploratory Data Analysis
Feature analysis with proper financial ratio names
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
PROJECT_ROOT = Path(__file__).parent.parent.parent
data_dir = PROJECT_ROOT / 'data' / 'processed' / 'american'
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / 'american'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("AMERICAN DATASET - Exploratory Data Analysis")
print("="*70)

# Load data and metadata
print("\n[1/5] Loading data and metadata...")
df = pd.read_parquet(data_dir / 'american_modeling.parquet')

import json
with open(data_dir / 'american_features_metadata.json', 'r') as f:
    metadata = json.load(f)

feature_cols = [col for col in df.columns if col.startswith('X')]
X = df[feature_cols]
y = df['bankrupt']

print(f"✓ Loaded {len(df):,} samples for modeling")
print(f"  Features: {len(feature_cols)}")
print(f"  Bankruptcy rate: {y.mean()*100:.2f}%")

# Create feature name mapping
feature_names = {col: metadata['features'][col]['name'] for col in feature_cols}

# ========================================================================
# Part 1: Feature Statistics
# ========================================================================
print("\n[2/5] Computing feature statistics...")

stats_data = []
for col in feature_cols:
    readable_name = feature_names[col]
    category = metadata['features'][col]['category']
    
    bankrupt_mean = X.loc[y == 1, col].mean()
    healthy_mean = X.loc[y == 0, col].mean()
    
    # T-test
    t_stat, p_value = stats.ttest_ind(
        X.loc[y == 1, col].dropna(),
        X.loc[y == 0, col].dropna(),
        equal_var=False
    )
    
    stats_data.append({
        'feature_code': col,
        'feature_name': readable_name,
        'category': category,
        'bankrupt_mean': bankrupt_mean,
        'healthy_mean': healthy_mean,
        'difference': bankrupt_mean - healthy_mean,
        'difference_pct': ((bankrupt_mean - healthy_mean) / abs(healthy_mean) * 100) if healthy_mean != 0 else 0,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    })

stats_df = pd.DataFrame(stats_data).sort_values('p_value')
stats_df.to_csv(output_dir / 'feature_statistics.csv', index=False)

print(f"✓ Computed statistics for {len(stats_df)} features")
print(f"  Significant differences (p<0.05): {stats_df['significant'].sum()}")
print(f"\n  Top 3 most discriminative features:")
for _, row in stats_df.head(3).iterrows():
    print(f"    • {row['feature_name']}: p={row['p_value']:.2e}")

# ========================================================================
# Part 2: Correlation Analysis
# ========================================================================
print("\n[3/5] Analyzing feature correlations...")

# Correlation with target
correlations = []
for col in feature_cols:
    corr = X[col].corr(y)
    correlations.append({
        'feature_code': col,
        'feature_name': feature_names[col],
        'category': metadata['features'][col]['category'],
        'correlation': corr,
        'abs_correlation': abs(corr)
    })

corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
corr_df.to_csv(output_dir / 'correlations_with_target.csv', index=False)

print(f"✓ Computed correlations with bankruptcy")
print(f"  Top 3 correlated features:")
for _, row in corr_df.head(3).iterrows():
    direction = "↑ higher risk" if row['correlation'] > 0 else "↓ lower risk"
    print(f"    • {row['feature_name']}: {row['correlation']:.4f} ({direction})")

# Feature-feature correlations
corr_matrix = X.corr()
high_corr_pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append({
                'feature_1': feature_names[feature_cols[i]],
                'feature_2': feature_names[feature_cols[j]],
                'correlation': corr_val
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False, key=abs)
    high_corr_df.to_csv(output_dir / 'high_correlations.csv', index=False)
    print(f"  High correlations (>0.7): {len(high_corr_pairs)} pairs")

# ========================================================================
# Part 3: Category Analysis
# ========================================================================
print("\n[4/5] Analyzing by category...")

category_stats = []
categories = set([metadata['features'][col]['category'] for col in feature_cols])

for category in categories:
    cat_features = [col for col in feature_cols if metadata['features'][col]['category'] == category]
    category_stats.append({
        'category': category,
        'feature_count': len(cat_features),
        'avg_correlation': corr_df[corr_df['category'] == category]['abs_correlation'].mean(),
        'significant_features': stats_df[
            (stats_df['category'] == category) & (stats_df['significant'])
        ].shape[0]
    })

cat_df = pd.DataFrame(category_stats).sort_values('avg_correlation', ascending=False)
cat_df.to_csv(output_dir / 'category_analysis.csv', index=False)

print(f"✓ Analyzed {len(categories)} categories")
for _, row in cat_df.iterrows():
    print(f"  • {row['category']}: {row['feature_count']} features, "
          f"{row['significant_features']} significant")

# ========================================================================
# Part 4: Visualizations
# ========================================================================
print("\n[5/5] Creating visualizations...")

# 1. Top discriminative features
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

top_10 = stats_df.head(10)
colors = ['red' if d < 0 else 'green' for d in top_10['difference']]

axes[0].barh(range(len(top_10)), top_10['difference'], color=colors, alpha=0.7)
axes[0].set_yticks(range(len(top_10)))
axes[0].set_yticklabels(top_10['feature_name'])
axes[0].set_xlabel('Mean Difference (Bankrupt - Healthy)', fontweight='bold')
axes[0].set_title('Top 10 Discriminative Features\n(Red: Lower when bankrupt, Green: Higher when bankrupt)', 
                  fontsize=12, fontweight='bold')
axes[0].axvline(x=0, color='black', linestyle='-', linewidth=1)
axes[0].grid(axis='x', alpha=0.3)

# Correlation plot
top_corr = corr_df.head(10)
colors_corr = ['darkred' if c < 0 else 'darkgreen' for c in top_corr['correlation']]

axes[1].barh(range(len(top_corr)), top_corr['correlation'], color=colors_corr, alpha=0.7)
axes[1].set_yticks(range(len(top_corr)))
axes[1].set_yticklabels(top_corr['feature_name'])
axes[1].set_xlabel('Correlation with Bankruptcy', fontweight='bold')
axes[1].set_title('Top 10 Correlated Features\n(Red: Protective, Green: Risk)', 
                  fontsize=12, fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'discriminative_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved discriminative features plot")

# 2. Correlation heatmap (top features)
plt.figure(figsize=(14, 12))
top_features_codes = corr_df.head(15)['feature_code'].tolist()
top_features_names = [feature_names[col] for col in top_features_codes]

corr_subset = X[top_features_codes].corr()
corr_subset.index = top_features_names
corr_subset.columns = top_features_names

sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap (Top 15 Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved correlation heatmap")

# 3. Category comparison
plt.figure(figsize=(10, 6))
cat_df_sorted = cat_df.sort_values('avg_correlation', ascending=True)

plt.barh(range(len(cat_df_sorted)), cat_df_sorted['avg_correlation'], 
         color='steelblue', alpha=0.7, edgecolor='black')
plt.yticks(range(len(cat_df_sorted)), cat_df_sorted['category'])
plt.xlabel('Average Absolute Correlation with Bankruptcy', fontweight='bold')
plt.title('Feature Category Discriminative Power', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved category analysis plot")

# 4. Distribution comparison for top 6 features
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

top_6 = stats_df.head(6)
for idx, (_, row) in enumerate(top_6.iterrows()):
    col = row['feature_code']
    
    axes[idx].hist(X.loc[y == 0, col], bins=30, alpha=0.6, label='Healthy', 
                   color='green', edgecolor='black')
    axes[idx].hist(X.loc[y == 1, col], bins=30, alpha=0.6, label='Bankrupt', 
                   color='red', edgecolor='black')
    axes[idx].set_title(row['feature_name'], fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.suptitle('Distribution Comparison: Top 6 Discriminative Features', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved feature distributions plot")

# Save summary
summary = {
    'total_features': len(feature_cols),
    'significant_features': int(stats_df['significant'].sum()),
    'high_correlations': len(high_corr_pairs) if high_corr_pairs else 0,
    'top_discriminative_feature': stats_df.iloc[0]['feature_name'],
    'top_discriminative_pvalue': float(stats_df.iloc[0]['p_value']),
    'top_correlated_feature': corr_df.iloc[0]['feature_name'],
    'top_correlation': float(corr_df.iloc[0]['correlation']),
    'categories': len(categories)
}

with open(output_dir / 'eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("✓ AMERICAN DATASET EDA COMPLETE")
print(f"  Total features: {summary['total_features']}")
print(f"  Significant features: {summary['significant_features']}")
print(f"  Top discriminative: {summary['top_discriminative_feature']}")
print(f"  Top correlated: {summary['top_correlated_feature']} ({summary['top_correlation']:.4f})")
print("="*70)
