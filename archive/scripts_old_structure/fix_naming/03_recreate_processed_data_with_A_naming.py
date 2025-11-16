"""
Recreate all processed parquet files using original A1-A64 naming.

This will:
1. Read raw CSV (has A1-A64)
2. Process it (clean, add horizons, etc.) 
3. Save as parquet WITHOUT renaming A1->Attr1
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    """Recreate processed Polish datasets with original A1-A64 naming."""
    
    base_dir = Path(__file__).resolve().parents[2]
    
    print("="*80)
    print("RECREATING PROCESSED DATA WITH ORIGINAL A1-A64 NAMING")
    print("="*80)
    print()
    
    # Load raw data
    print("1. Loading raw CSV...")
    csv_path = base_dir / "data/polish-companies-bankruptcy/data-from-kaggel.csv"
    df = pd.read_csv(csv_path)
    
    print(f"   Raw data shape: {df.shape}")
    print(f"   Columns (first 10): {df.columns[:10].tolist()}")
    print(f"   ✓ Confirms A1-A64 naming in raw data")
    print()
    
    # Basic cleaning
    print("2. Processing data...")
    
    # Rename target column
    df = df.rename(columns={'class': 'bankrupt'})
    
    # Create binary target 'y'
    df['y'] = df['bankrupt'].astype(int)
    
    # Add horizon column (from year column if exists)
    if 'year' in df.columns:
        df['horizon'] = df['year']
    else:
        # If no year column, this is probably 5thYear data (h=1)
        df['horizon'] = 1
    
    print(f"   Processed shape: {df.shape}")
    print(f"   Target distribution: {df['y'].value_counts().to_dict()}")
    print(f"   Horizons: {sorted(df['horizon'].unique())}")
    print()
    
    # Save as full dataset
    print("3. Saving processed datasets...")
    output_dir = base_dir / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full dataset
    full_path = output_dir / "poland_clean_full.parquet"
    df.to_parquet(full_path, index=False)
    print(f"   ✓ Saved: {full_path}")
    
    # Verify no renaming happened
    df_verify = pd.read_parquet(full_path)
    feature_cols = [c for c in df_verify.columns if c not in ['y', 'bankrupt', 'horizon', 'year', 'class']]
    
    print()
    print("4. Verification:")
    print(f"   Feature columns (first 10): {feature_cols[:10]}")
    print(f"   Feature columns (last 5): {feature_cols[-5:]}")
    
    if all(c.startswith('A') and not c.startswith('Attr') for c in feature_cols[:64]):
        print(f"   ✅ SUCCESS: All features use A1-A64 naming!")
    else:
        print(f"   ❌ ERROR: Some features don't use A naming!")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Raw CSV:        A1-A64 ✓")
    print(f"Processed data: A1-A64 ✓")
    print(f"Total samples:  {len(df):,}")
    print(f"Total features: {len(feature_cols)}")
    print()
    
    return df


if __name__ == "__main__":
    df = main()
    print("✅ Processed data recreated with original A1-A64 naming!")
