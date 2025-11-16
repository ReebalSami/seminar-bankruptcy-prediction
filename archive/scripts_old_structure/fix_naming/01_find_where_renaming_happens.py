"""
Find where A1->Attr1 renaming actually happens by checking:
1. Raw CSV columns
2. Processed parquet columns  
3. Any script that creates the processed files
"""

import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]

print("="*80)
print("FINDING WHERE A1->Attr1 RENAMING HAPPENS")
print("="*80)
print()

# 1. Check raw CSV
print("1. RAW CSV COLUMNS")
print("-"*80)
csv_path = base_dir / "data/polish-companies-bankruptcy/data-from-kaggel.csv"
df_raw = pd.read_csv(csv_path, nrows=5)
print(f"First 10 columns: {df_raw.columns[:10].tolist()}")
print(f"Last 5 columns: {df_raw.columns[-5:].tolist()}")
print()

# 2. Check processed parquet
print("2. PROCESSED PARQUET COLUMNS")
print("-"*80)
parquet_path = base_dir / "data/processed/poland_clean_full.parquet"
if parquet_path.exists():
    df_processed = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df_processed.columns if c not in ['y', 'horizon', 'bankrupt', 'class', 'year']]
    print(f"First 10 feature columns: {feature_cols[:10]}")
    print(f"Last 5 feature columns: {feature_cols[-5:]}")
else:
    print("NOT FOUND")
print()

# 3. Look for any notebooks or scripts that create processed files
print("3. SEARCHING FOR SCRIPTS THAT CREATE PROCESSED FILES")
print("-"*80)

# Check for notebooks
notebook_dir = base_dir / "notebooks"
if notebook_dir.exists():
    notebooks = list(notebook_dir.glob("*.ipynb"))
    print(f"Found {len(notebooks)} notebooks:")
    for nb in notebooks[:5]:
        print(f"  - {nb.name}")
else:
    print("No notebooks directory found")

print()

# The renaming likely happens in one of these places:
print("LIKELY LOCATIONS FOR RENAMING:")
print("  1. A preprocessing notebook (notebooks/*.ipynb)")
print("  2. An initial data cleaning script")
print("  3. Directly in the parquet creation process")
print()
print("ACTION: We need to recreate the processed parquet files with A1-A64 naming")
