#!/usr/bin/env python3
"""
Taiwan Dataset - Data Cleaning
Taiwan Economic Journal (TEJ) bankruptcy data
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

# Setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
data_dir = PROJECT_ROOT / 'data' / 'taiwan-economic-journal'
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / 'taiwan'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)
processed_dir = PROJECT_ROOT / 'data' / 'processed' / 'taiwan'
processed_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("TAIWAN DATASET - Data Cleaning")
print("="*70)

# Load data
print("\n[1/5] Loading raw data...")
df = pd.read_csv(data_dir / 'taiwan-bankruptcy.csv')
print(f"✓ Loaded {len(df):,} samples")
print(f"  Columns: {len(df.columns)}")

# Clean column names
print("\n[2/5] Cleaning column names...")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
original_names = df.columns.tolist()

# Create simpler feature names (F1, F2, etc. for now)
feature_mapping = {}
feature_cols = []
for i, col in enumerate(df.columns):
    if col != 'Bankrupt?':
        new_name = f'F{i+1:02d}'
        feature_mapping[new_name] = {
            'original_name': col,
            'index': i
        }
        feature_cols.append(new_name)

# Rename columns
rename_dict = {'Bankrupt?': 'bankrupt'}
for new_name, info in feature_mapping.items():
    rename_dict[info['original_name']] = new_name

df = df.rename(columns=rename_dict)
print(f"✓ Cleaned {len(feature_cols)} feature names")
print(f"  Example: '{original_names[1]}' → 'F01'")

# Create binary target
print("\n[3/5] Analyzing target variable...")
print(f"  Bankrupt: {(df['bankrupt'] == 1).sum():,} ({df['bankrupt'].mean()*100:.2f}%)")
print(f"  Healthy: {(df['bankrupt'] == 0).sum():,} ({(1-df['bankrupt'].mean())*100:.2f}%)")

# Check data quality
print("\n[4/5] Analyzing data quality...")
missing_total = df[feature_cols].isnull().sum().sum()
print(f"  Missing values: {missing_total}")
print(f"  Duplicate rows: {df.duplicated().sum()}")

# Check for infinities and outliers
inf_counts = {}
for col in feature_cols:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        inf_counts[col] = inf_count

if inf_counts:
    print(f"  Infinities found in {len(inf_counts)} features")
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    print(f"  ✓ Replaced infinities with NaN")

# Handle missing values
if df[feature_cols].isnull().sum().sum() > 0:
    print(f"  Filling missing values with median...")
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    print(f"  ✓ Filled missing values")

# Save clean dataset
print("\n[5/5] Saving cleaned data...")

df_clean = df[['bankrupt'] + feature_cols].copy()
df_clean.to_parquet(processed_dir / 'taiwan_clean.parquet', index=False)
print(f"  ✓ Saved clean dataset: {len(df_clean):,} samples")

# Save feature mapping
import json
with open(processed_dir / 'taiwan_features_metadata.json', 'w') as f:
    json.dump(feature_mapping, f, indent=2)
print(f"  ✓ Saved feature mapping ({len(feature_mapping)} features)")

# Summary statistics
summary = {
    'total_samples': int(len(df_clean)),
    'features': len(feature_cols),
    'bankruptcy_rate': float(df_clean['bankrupt'].mean()),
    'bankruptcies': int(df_clean['bankrupt'].sum()),
    'healthycompanies': int((df_clean['bankrupt'] == 0).sum())
}

with open(output_dir / 'cleaning_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Create visualizations
print("\nCreating visualizations...")

# 1. Class distribution
plt.figure(figsize=(10, 6))
counts = df_clean['bankrupt'].value_counts()
colors = ['green', 'red']
plt.bar(['Healthy', 'Bankrupt'], [counts[0], counts[1]], color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Number of Companies', fontweight='bold')
plt.title('Class Distribution - Taiwan Dataset', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for i, (label, count) in enumerate(zip(['Healthy', 'Bankrupt'], [counts[0], counts[1]])):
    pct = count / len(df_clean) * 100
    plt.text(i, count + 50, f'{count:,}\n({pct:.2f}%)', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved class distribution plot")

# 2. Feature distributions (sample of 9)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

sample_features = feature_cols[:9]
for idx, col in enumerate(sample_features):
    original_name = feature_mapping[col]['original_name']
    axes[idx].hist(df_clean[col], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    # Truncate long names
    short_name = original_name[:30] + '...' if len(original_name) > 30 else original_name
    axes[idx].set_title(f'{col}: {short_name}', fontweight='bold', fontsize=9)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(alpha=0.3)

plt.suptitle('Feature Distributions (First 9 Features)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved feature distributions plot")

# Save summary DataFrame
pd.DataFrame([summary]).to_csv(output_dir / 'cleaning_summary.csv', index=False)

print("\n" + "="*70)
print("✓ TAIWAN DATASET CLEANING COMPLETE")
print(f"  Total samples: {summary['total_samples']:,}")
print(f"  Features: {summary['features']}")
print(f"  Bankruptcy rate: {summary['bankruptcy_rate']*100:.2f}%")
print(f"  ✓ Feature mapping saved (F01-F95 → original names)")
print("="*70)
