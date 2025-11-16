#!/usr/bin/env python3
"""
American Dataset - Create Multi-Horizon Structure
Transform 20-year panel into prediction horizons like Polish dataset
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import json

# Setup
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / 'data' / 'NYSE-and-NASDAQ-companies'
output_dir = project_root / 'data' / 'processed' / 'american'
results_dir = project_root / 'results' / 'script_outputs' / 'american'

print("="*70)
print("AMERICAN DATASET - Multi-Horizon Creation")
print("="*70)

# Load raw data
print("\n[1/4] Loading raw data...")
df = pd.read_csv(data_dir / 'american-bankruptcy.csv')

print(f"✓ Loaded {len(df):,} observations")
print(f"  Years: {df['year'].min()} - {df['year'].max()}")
print(f"  Companies: {df['company_name'].nunique()}")

# Load metadata
with open(output_dir / 'american_features_metadata.json') as f:
    metadata = json.load(f)

# ========================================================================
# Create Horizons
# ========================================================================
print("\n[2/4] Creating prediction horizons...")

# Strategy: For each company-year, predict bankruptcy in next 1-5 years
# Horizon 1: Bankruptcy in next 1 year
# Horizon 2: Bankruptcy in next 2 years
# ... and so on

df = df.sort_values(['company_name', 'year']).reset_index(drop=True)

# Create bankruptcy indicator from status_label
df['status_code'] = df['status_label'].apply(lambda x: 1 if 'failed' in str(x).lower() or 'bankrupt' in str(x).lower() else 0)

feature_cols = [col for col in df.columns if col.startswith('X')]

# Create horizon datasets
horizon_data = []

for horizon in range(1, 6):
    print(f"  Creating Horizon {horizon}...")
    
    horizon_rows = []
    
    for company_id in df['company_name'].unique():
        company_df = df[df['company_name'] == company_id].sort_values('year')
        
        # For each year, check if bankrupt in next 'horizon' years
        for idx in range(len(company_df) - horizon):
            current_row = company_df.iloc[idx]
            future_rows = company_df.iloc[idx+1:idx+1+horizon]
            
            # Check if any bankruptcy in the next 'horizon' years
            bankrupt_future = future_rows['status_code'].max() if len(future_rows) > 0 else 0
            
            # Create record
            record = {
                'horizon': horizon,
                'company_id': company_id,
                'year': current_row['year'],
                'bankrupt': int(bankrupt_future > 0)
            }
            
            # Add features
            for col in feature_cols:
                record[col] = current_row[col]
            
            horizon_rows.append(record)
    
    horizon_df = pd.DataFrame(horizon_rows)
    horizon_data.append(horizon_df)
    
    print(f"    Samples: {len(horizon_df):,}, Bankruptcies: {horizon_df['bankrupt'].sum()} ({horizon_df['bankrupt'].mean()*100:.2f}%)")

# Combine all horizons
df_multi_horizon = pd.concat(horizon_data, ignore_index=True)

print(f"\n✓ Created multi-horizon dataset: {len(df_multi_horizon):,} observations")

# ========================================================================
# Clean and Save
# ========================================================================
print("\n[3/4] Cleaning data...")

# Handle missing values
for col in feature_cols:
    if df_multi_horizon[col].isnull().sum() > 0:
        df_multi_horizon[col].fillna(df_multi_horizon[col].median(), inplace=True)

# Handle infinities
df_multi_horizon[feature_cols] = df_multi_horizon[feature_cols].replace([np.inf, -np.inf], np.nan)
for col in feature_cols:
    if df_multi_horizon[col].isnull().sum() > 0:
        df_multi_horizon[col].fillna(df_multi_horizon[col].median(), inplace=True)

print(f"✓ Data cleaned")

# ========================================================================
# Save
# ========================================================================
print("\n[4/4] Saving...")

df_multi_horizon.to_parquet(output_dir / 'american_multi_horizon.parquet', index=False)

# Statistics by horizon
horizon_stats = []
for h in range(1, 6):
    h_df = df_multi_horizon[df_multi_horizon['horizon'] == h]
    horizon_stats.append({
        'horizon': h,
        'samples': len(h_df),
        'bankruptcies': int(h_df['bankrupt'].sum()),
        'bankruptcy_rate': float(h_df['bankrupt'].mean()),
        'companies': h_df['company_id'].nunique()
    })

horizon_stats_df = pd.DataFrame(horizon_stats)
horizon_stats_df.to_csv(results_dir / 'multi_horizon_stats.csv', index=False)

# Summary
summary = {
    'total_observations': int(len(df_multi_horizon)),
    'horizons': 5,
    'features': len(feature_cols),
    'horizon_stats': horizon_stats
}

with open(results_dir / 'multi_horizon_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Saved multi-horizon dataset and statistics")

print("\n" + "="*70)
print("✓ AMERICAN MULTI-HORIZON CREATION COMPLETE")
print("="*70)
print("\nSummary by Horizon:")
for stat in horizon_stats:
    print(f"  H{stat['horizon']}: {stat['samples']:,} samples, {stat['bankruptcies']} bankruptcies ({stat['bankruptcy_rate']*100:.2f}%)")
print("="*70)
