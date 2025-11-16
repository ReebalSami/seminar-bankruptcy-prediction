#!/usr/bin/env python3
import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/poland_clean_full.parquet')
feature_cols = [c for c in df.columns if c.startswith('A')]

print('='*80)
print('OUTLIER ANALYSIS - 3×IQR METHOD')
print('='*80)
print()

outlier_stats = []
for feat in feature_cols:
    Q1 = df[feat].quantile(0.25)
    Q3 = df[feat].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    outliers = ((df[feat] < lower) | (df[feat] > upper)).sum()
    valid = df[feat].notna().sum()
    outlier_pct = (outliers / valid * 100) if valid > 0 else 0
    outlier_stats.append(outlier_pct)

outlier_stats = np.array(outlier_stats)

print(f'Features analyzed: {len(feature_cols)}')
print(f'Features with outliers (>0%): {(outlier_stats > 0).sum()}/64')
print()
print('Outlier percentage statistics:')
print(f'  Min: {outlier_stats.min():.2f}%')
print(f'  Max: {outlier_stats.max():.2f}%')
print(f'  Mean: {outlier_stats.mean():.2f}%')
print(f'  Median: {np.median(outlier_stats):.2f}%')
print(f'  Std Dev: {outlier_stats.std():.2f}%')
print()
print('Distribution of outlier percentages:')
print(f'  0%: {(outlier_stats == 0).sum()} features')
print(f'  0-5%: {((outlier_stats > 0) & (outlier_stats < 5)).sum()} features')
print(f'  5-10%: {((outlier_stats >= 5) & (outlier_stats < 10)).sum()} features')
print(f'  10-15%: {((outlier_stats >= 10) & (outlier_stats < 15)).sum()} features')
print(f'  15-20%: {((outlier_stats >= 15) & (outlier_stats < 20)).sum()} features')
print(f'  >20%: {(outlier_stats >= 20).sum()} features')
print()
print('Paper claims:')
print('  "ALL 64 features have outliers (~10-15% per feature)"')
print()
print('Reality check:')
print(f'  ✓ ALL 64 have outliers: {(outlier_stats > 0).sum() == 64}')
print(f'  ✗ Mean ~10-15%: {outlier_stats.mean():.1f}% (expected 10-15%)')
print(f'  ✗ Median ~10-15%: {np.median(outlier_stats):.1f}% (expected 10-15%)')
print()

# Show worst 10
outlier_data = [(feature_cols[i], outlier_stats[i]) for i in range(len(feature_cols))]
outlier_data.sort(key=lambda x: x[1], reverse=True)
print('Worst 10 features by outlier percentage:')
for feat, pct in outlier_data[:10]:
    print(f'  {feat}: {pct:.2f}%')
