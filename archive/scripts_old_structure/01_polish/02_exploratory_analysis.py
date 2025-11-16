#!/usr/bin/env python3
"""Exploratory Analysis - Correlations and discriminative power"""

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
from scipy import stats

from src.bankruptcy_prediction.data import DataLoader, MetadataParser

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
# PROJECT_ROOT already defined as PROJECT_ROOT above
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / '02_exploratory_analysis'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("SCRIPT 02: Exploratory Analysis")
print("="*70)

loader = DataLoader()
metadata = MetadataParser.from_default()
df = loader.load_poland(horizon=1, dataset_type='full')
X, y = loader.get_features_target(df)

# Get base features only
base_features = [col for col in X.columns if '__isna' not in col][:64]
X_base = X[base_features]

print(f"\n[1/4] Calculating correlations for {len(base_features)} features...")
corr_matrix = X_base.corr()
high_corr_threshold = 0.9
upper_tri = np.triu(np.abs(corr_matrix), k=1)
high_corr_pairs = []

for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        if upper_tri[i, j] >= high_corr_threshold:
            feat1 = corr_matrix.columns[i]
            feat2 = corr_matrix.columns[j]
            high_corr_pairs.append({
                'Feature_1': feat1,
                'Feature_2': feat2,
                'Correlation': corr_matrix.iloc[i, j],
                'Name_1': metadata.get_readable_name(feat1, short=True),
                'Name_2': metadata.get_readable_name(feat2, short=True),
            })

high_corr_df = pd.DataFrame(high_corr_pairs)
if len(high_corr_df) > 0:
    high_corr_df.to_csv(output_dir / 'high_correlations.csv', index=False)
    print(f"✓ Found {len(high_corr_df)} high correlation pairs")
else:
    print("✓ No extreme correlations found")

print("\n[2/4] Calculating discriminative power...")
discriminative_power = {}
for col in base_features:
    valid_mask = ~X_base[col].isna()
    if valid_mask.sum() < 100:
        continue
    x_clean = X_base.loc[valid_mask, col]
    y_clean = y[valid_mask]
    corr, pval = stats.pointbiserialr(y_clean, x_clean)
    discriminative_power[col] = {
        'correlation': corr,
        'abs_correlation': abs(corr),
        'p_value': pval,
        'readable_name': metadata.get_readable_name(col, short=True),
        'category': metadata.get_category(col)
    }

disc_df = pd.DataFrame.from_dict(discriminative_power, orient='index').sort_values('abs_correlation', ascending=False)
disc_df.to_csv(output_dir / 'discriminative_power.csv')
print(f"✓ Analyzed {len(disc_df)} features")
print(f"  Top 3: {', '.join(disc_df['readable_name'].head(3).tolist())}")

print("\n[3/4] Creating correlation heatmap...")
top_features = X_base.var().sort_values(ascending=False).head(30).index.tolist()
corr_subset = X_base[top_features].corr()
readable_names = [metadata.get_readable_name(f, short=True) for f in top_features]
corr_subset.index = readable_names
corr_subset.columns = readable_names

plt.figure(figsize=(16, 14))
sns.heatmap(corr_subset, annot=False, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
            square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'})
plt.title('Feature Correlation Heatmap (Top 30 by Variance)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved correlation heatmap")

print("\n[4/4] Creating discriminative power plot...")
top_20 = disc_df.head(20).copy()
category_colors = {'Profitability': '#3498db', 'Liquidity': '#2ecc71', 
                   'Leverage': '#e74c3c', 'Activity': '#f39c12', 
                   'Size': '#9b59b6', 'Other': '#95a5a6'}
colors = [category_colors.get(cat, '#95a5a6') for cat in top_20['category']]

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(range(len(top_20)), top_20['abs_correlation'], color=colors, alpha=0.8)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['readable_name'])
ax.set_xlabel('Absolute Correlation with Bankruptcy', fontweight='bold')
ax.set_title('Top 20 Most Discriminative Features', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'discriminative_power.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved discriminative power plot")

print("\n" + "="*70)
print("✓ SCRIPT 02 COMPLETE")
print("="*70)
