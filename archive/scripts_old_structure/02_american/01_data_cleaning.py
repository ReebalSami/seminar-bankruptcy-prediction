#!/usr/bin/env python3
"""
American Dataset - Data Cleaning
NYSE and NASDAQ companies bankruptcy data (1999-2018)
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
data_dir = PROJECT_ROOT / 'data' / 'NYSE-and-NASDAQ-companies'
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / 'american'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)
processed_dir = PROJECT_ROOT / 'data' / 'processed' / 'american'
processed_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("AMERICAN DATASET - Data Cleaning")
print("="*70)

# Load data
print("\n[1/5] Loading raw data...")
df = pd.read_csv(data_dir / 'american-bankruptcy.csv')
print(f"✓ Loaded {len(df):,} samples")
print(f"  Columns: {len(df.columns)}")
print(f"  Years: {df['year'].min()}-{df['year'].max()}")

# Create binary target
print("\n[2/5] Creating binary target...")
df['bankrupt'] = (df['status_label'] == 'failed').astype(int)
print(f"✓ Target created")
print(f"  Failed: {(df['bankrupt'] == 1).sum():,} ({df['bankrupt'].mean()*100:.2f}%)")
print(f"  Alive: {(df['bankrupt'] == 0).sum():,} ({(1-df['bankrupt'].mean())*100:.2f}%)")

# Check data quality
print("\n[3/5] Analyzing data quality...")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Duplicate rows: {df.duplicated().sum()}")

# Get feature columns
feature_cols = [col for col in df.columns if col.startswith('X')]
print(f"  Features: {len(feature_cols)}")

# Check for infinities
inf_counts = {}
for col in feature_cols:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        inf_counts[col] = inf_count

if inf_counts:
    print(f"  Infinities found in {len(inf_counts)} features:")
    for col, count in list(inf_counts.items())[:5]:
        print(f"    • {col}: {count}")
else:
    print(f"  ✓ No infinities")

# Clean infinities
if inf_counts:
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    print(f"  ✓ Replaced infinities with NaN")

# Handle missing values
missing_before = df[feature_cols].isnull().sum().sum()
if missing_before > 0:
    # Fill with median
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    print(f"  ✓ Filled {missing_before} missing values with median")

# Remove outliers (extreme values)
print("\n[4/5] Handling outliers...")
outliers_removed = 0
for col in feature_cols:
    q1 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    outliers = ((df[col] < q1) | (df[col] > q99)).sum()
    df[col] = df[col].clip(q1, q99)
    outliers_removed += outliers

print(f"  ✓ Clipped {outliers_removed:,} outlier values (1st-99th percentile)")

# Create clean dataset
print("\n[5/5] Saving cleaned data...")

# Save full dataset
df_clean = df[['company_name', 'year', 'bankrupt'] + feature_cols].copy()
df_clean.to_parquet(processed_dir / 'american_clean.parquet', index=False)
print(f"  ✓ Saved clean dataset: {len(df_clean):,} samples")

# Create modeling dataset from recent years (more balanced)
# Use 2015-2018 to get more bankruptcies while keeping recent data
recent_years = [2015, 2016, 2017, 2018]
df_modeling = df_clean[df_clean['year'].isin(recent_years)].copy()

# Take latest observation per company to avoid temporal leakage
df_modeling = df_modeling.sort_values('year').groupby('company_name').tail(1)
df_modeling.to_parquet(processed_dir / 'american_modeling.parquet', index=False)

bankruptcy_count_modeling = df_modeling['bankrupt'].sum()
print(f"  ✓ Saved modeling dataset (2015-2018, latest per company): {len(df_modeling):,} samples")
print(f"    Bankruptcies: {bankruptcy_count_modeling:,} ({df_modeling['bankrupt'].mean()*100:.2f}%)")

# Load metadata
import json
with open(processed_dir / 'american_features_metadata.json', 'r') as f:
    metadata = json.load(f)

# Summary statistics
summary = {
    'total_samples': int(len(df_clean)),
    'companies': int(df_clean['company_name'].nunique()),
    'years': f"{df_clean['year'].min()}-{df_clean['year'].max()}",
    'features': len(feature_cols),
    'bankruptcy_rate': float(df_clean['bankrupt'].mean()),
    'bankruptcies': int(df_clean['bankrupt'].sum()),
    'modeling_samples': int(len(df_modeling)),
    'modeling_bankruptcies': int(bankruptcy_count_modeling),
    'modeling_rate': float(df_modeling['bankrupt'].mean())
}

with open(output_dir / 'cleaning_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Save feature metadata to output for easy access
feature_names = {}
for col in feature_cols:
    feature_names[col] = metadata['features'][col]['name']

with open(output_dir / 'feature_names.json', 'w') as f:
    json.dump(feature_names, f, indent=2)

# Create visualizations
print("\nCreating visualizations...")

# 1. Bankruptcy rate over time
plt.figure(figsize=(12, 6))
yearly_rate = df_clean.groupby('year')['bankrupt'].agg(['mean', 'sum', 'count'])
yearly_rate['rate_pct'] = yearly_rate['mean'] * 100

plt.subplot(1, 2, 1)
plt.plot(yearly_rate.index, yearly_rate['rate_pct'], marker='o', linewidth=2, markersize=8, color='crimson')
plt.xlabel('Year', fontweight='bold')
plt.ylabel('Bankruptcy Rate (%)', fontweight='bold')
plt.title('Bankruptcy Rate Over Time', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(yearly_rate.index, yearly_rate['sum'], color='darkred', alpha=0.7)
plt.xlabel('Year', fontweight='bold')
plt.ylabel('Number of Bankruptcies', fontweight='bold')
plt.title('Bankruptcies by Year', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'bankruptcy_over_time.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved bankruptcy over time plot")

# 2. Feature distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(feature_cols[:9]):
    axes[idx].hist(df_clean[col], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx].set_title(col, fontweight='bold')
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
yearly_rate.to_csv(output_dir / 'bankruptcy_by_year.csv')

print("\n" + "="*70)
print("✓ AMERICAN DATASET CLEANING COMPLETE")
print(f"  Total samples: {summary['total_samples']:,}")
print(f"  Companies: {summary['companies']:,}")
print(f"  Overall bankruptcy rate: {summary['bankruptcy_rate']*100:.2f}%")
print(f"  Modeling dataset (2015-2018): {summary['modeling_samples']:,} samples")
print(f"  Modeling bankruptcy rate: {summary['modeling_rate']*100:.2f}%")
print(f"  ✓ Feature metadata integrated ({len(feature_names)} features)")
print("="*70)
