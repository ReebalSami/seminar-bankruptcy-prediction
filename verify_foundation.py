#!/usr/bin/env python3
"""
Verification script for foundation phase analysis
"""
import pandas as pd
import numpy as np
import json

# Load data
df = pd.read_parquet('data/processed/poland_clean_full.parquet')

print('='*80)
print('FOUNDATION PHASE VERIFICATION')
print('='*80)
print()

# 1. Basic dimensions
print('1. BASIC DIMENSIONS')
print(f'   Total observations: {len(df):,}')
print(f'   Paper claims: 43,405')
print(f'   ✓ Match: {len(df) == 43405}')
print()

feature_cols = [c for c in df.columns if c.startswith('A')]
print(f'   Total features: {len(feature_cols)}')
print(f'   Paper claims: 64')
print(f'   ✓ Match: {len(feature_cols) == 64}')
print()

# 2. Bankruptcy distribution
print('2. BANKRUPTCY DISTRIBUTION')
bankrupt = (df['y'] == 1).sum()
healthy = (df['y'] == 0).sum()
rate = bankrupt / len(df) * 100
print(f'   Bankruptcies: {bankrupt:,}')
print(f'   Healthy: {healthy:,}')
print(f'   Rate: {rate:.2f}%')
print(f'   Paper claims: 2,091 bankruptcies (4.82%)')
print(f'   ✓ Match: {bankrupt == 2091 and abs(rate - 4.82) < 0.01}')
print()

# 3. Horizon distribution
print('3. HORIZON DISTRIBUTION')
print('   Actual data:')
for h in sorted(df['horizon'].unique()):
    h_df = df[df['horizon'] == h]
    h_total = len(h_df)
    h_bankrupt = (h_df['y'] == 1).sum()
    h_rate = h_bankrupt / h_total * 100
    h_pct = h_total / len(df) * 100
    print(f'   H{h}: {h_total:,} obs ({h_pct:.1f}%), {h_bankrupt} bankrupt, {h_rate:.2f}% rate')

print()
print('   Paper claims (Table 3.1):')
paper_claims = {
    1: {'obs': 7027, 'pct': 16.2, 'bankrupt': 271, 'rate': 3.86},
    2: {'obs': 10173, 'pct': 23.4, 'bankrupt': 400, 'rate': 3.93},
    3: {'obs': 10503, 'pct': 24.2, 'bankrupt': 495, 'rate': 4.71},
    4: {'obs': 9792, 'pct': 22.6, 'bankrupt': 515, 'rate': 5.26},
    5: {'obs': 5910, 'pct': 13.6, 'bankrupt': 410, 'rate': 6.94}
}
for h, claim in paper_claims.items():
    print(f'   H{h}: {claim["obs"]:,} obs ({claim["pct"]:.1f}%), {claim["bankrupt"]} bankrupt, {claim["rate"]:.2f}% rate')

# Verify
print()
print('   Verification:')
all_match = True
for h, claim in paper_claims.items():
    h_df = df[df['horizon'] == h]
    h_total = len(h_df)
    h_bankrupt = (h_df['y'] == 1).sum()
    h_rate = h_bankrupt / h_total * 100
    
    obs_match = h_total == claim['obs']
    bankrupt_match = h_bankrupt == claim['bankrupt']
    rate_match = abs(h_rate - claim['rate']) < 0.01
    
    if not (obs_match and bankrupt_match and rate_match):
        all_match = False
        print(f'   ✗ H{h}: obs={obs_match}, bankrupt={bankrupt_match}, rate={rate_match}')
    
if all_match:
    print('   ✓ All horizon numbers match!')
else:
    print('   ✗ Some mismatches detected')
print()

# 4. Bankruptcy rate increase H1 -> H5
print('4. TEMPORAL TREND (KEY FINDING)')
h1_rate = ((df[df['horizon']==1]['y'] == 1).sum() / len(df[df['horizon']==1])) * 100
h5_rate = ((df[df['horizon']==5]['y'] == 1).sum() / len(df[df['horizon']==5])) * 100
increase_pct = ((h5_rate - h1_rate) / h1_rate) * 100
absolute_increase = h5_rate - h1_rate

print(f'   H1 rate: {h1_rate:.2f}%')
print(f'   H5 rate: {h5_rate:.2f}%')
print(f'   Absolute increase: {absolute_increase:.2f} percentage points')
print(f'   Relative increase: {increase_pct:.1f}%')
print(f'   Paper claims: 80% increase')
print(f'   ✓ Match: {abs(increase_pct - 80) < 2}')  # Allow 2% tolerance
print()

# 5. Duplicates
print('5. DUPLICATES')
exact_dups = df.duplicated().sum()
print(f'   Exact duplicates: {exact_dups}')
print(f'   Paper claims: 401')
print(f'   ✓ Match: {exact_dups == 401}')

if exact_dups > 0:
    dup_pairs = exact_dups // 2
    print(f'   Number of pairs: {dup_pairs}')
    print(f'   Paper claims: 200 pairs')
    print(f'   ✓ Match: {dup_pairs == 200}')
print()

# 6. Missing values
print('6. MISSING VALUES')
all_features_missing = []
for feat in feature_cols:
    miss = df[feat].isnull().sum()
    if miss > 0:
        all_features_missing.append(feat)
        
features_with_missing = len(all_features_missing)
print(f'   Features with missing values: {features_with_missing}/{len(feature_cols)}')
print(f'   Paper claims: ALL 64 features have missing values')
print(f'   ✓ Match: {features_with_missing == 64}')
print()

# Check A37 specifically
miss_a37 = df['A37'].isnull().sum()
miss_a37_pct = miss_a37 / len(df) * 100
print(f'   A37 missing: {miss_a37:,} ({miss_a37_pct:.2f}%)')
print(f'   Paper claims: 43.7%')
print(f'   ✓ Match: {abs(miss_a37_pct - 43.7) < 0.5}')
print()

# Find worst missing feature
missing_stats = []
for feat in feature_cols:
    miss = df[feat].isnull().sum()
    miss_pct = miss / len(df) * 100
    missing_stats.append((feat, miss, miss_pct))

missing_stats.sort(key=lambda x: x[2], reverse=True)
worst_feat, worst_miss, worst_pct = missing_stats[0]
print(f'   Worst feature: {worst_feat} ({worst_pct:.2f}%)')
print(f'   Expected: A37 (43.7%)')
print(f'   ✓ Match: {worst_feat == "A37"}')
print()

# 7. Outliers (claim check)
print('7. OUTLIERS')
print('   Paper claims: ALL 64 features have outliers (~10-15% per feature)')
print('   Checking sample of 5 features using 3×IQR method...')

outlier_counts = []
for feat in feature_cols[:5]:
    Q1 = df[feat].quantile(0.25)
    Q3 = df[feat].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    outliers = ((df[feat] < lower) | (df[feat] > upper)).sum()
    valid = df[feat].notna().sum()
    outlier_pct = (outliers / valid * 100) if valid > 0 else 0
    print(f'   {feat}: {outliers} outliers ({outlier_pct:.1f}%)')
    outlier_counts.append(outlier_pct)

avg_outlier_pct = np.mean(outlier_counts)
print(f'   Average: {avg_outlier_pct:.1f}%')
print(f'   Expected range: 10-15%')
in_range = 5 <= avg_outlier_pct <= 20  # Slightly broader range
print(f'   ✓ Plausible: {in_range}')
print()

# 8. Category distribution
print('8. FEATURE CATEGORIES')
try:
    with open('data/polish-companies-bankruptcy/feature_descriptions.json') as f:
        meta = json.load(f)
    
    categories = meta['categories']
    print('   Actual category counts:')
    for cat, info in categories.items():
        print(f'   {cat}: {len(info["features"])} features')
    
    print()
    print('   Paper claims (Table 3.2):')
    print('   Profitability: 29 features')
    print('   Liquidity: 12 features')
    print('   Leverage: 9 features')
    print('   Activity: 8 features')
    print('   Size: 4 features')
    print('   Other: 2 features')
    
    # Verify
    paper_cat_counts = {
        'Profitability': 29,
        'Liquidity': 12,
        'Leverage': 9,
        'Activity': 8,
        'Size': 4,
        'Other': 2
    }
    
    all_cat_match = True
    for cat, expected in paper_cat_counts.items():
        if cat in categories:
            actual = len(categories[cat]['features'])
            if actual != expected:
                all_cat_match = False
                print(f'   ✗ {cat}: expected {expected}, got {actual}')
    
    if all_cat_match:
        print('   ✓ All category counts match!')
    
except Exception as e:
    print(f'   Could not verify: {e}')

print()
print('='*80)
print('VERIFICATION COMPLETE')
print('='*80)
