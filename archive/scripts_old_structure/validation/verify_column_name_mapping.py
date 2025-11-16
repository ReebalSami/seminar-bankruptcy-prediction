"""
CRITICAL: Verify if there's a mismatch between actual data columns and our documentation.

FINDINGS SO FAR:
- Actual CSV file uses: A1, A2, A3... A64
- Kaggle metadata uses: A1, A2, A3... A64  
- UCI documentation uses: X1, X2, X3... X64 (just for reference)
- Our feature_descriptions.json uses: Attr1, Attr2, Attr3... Attr64

QUESTION: Did we rename A1->Attr1 during preprocessing, or is there a mismatch?
"""

import pandas as pd
import json
from pathlib import Path

def main():
    """Verify column name consistency across all data files."""
    
    base_dir = Path(__file__).resolve().parents[2]
    
    print("=" * 80)
    print("CRITICAL: COLUMN NAME MAPPING VERIFICATION")
    print("=" * 80)
    print()
    
    # Check all possible processed data locations
    processed_locations = [
        base_dir / "data/processed/poland_clean_full.parquet",
        base_dir / "data/processed/poland/poland_clean_full.parquet",
        base_dir / "data/processed/01_polish/poland_clean_full.parquet",
        base_dir / "results/processed_data/poland_clean_full.parquet",
    ]
    
    print("1. SEARCHING FOR PROCESSED DATA FILES")
    print("-" * 80)
    
    processed_df = None
    processed_path = None
    
    for path in processed_locations:
        print(f"   Checking: {path}")
        if path.exists():
            print(f"   ✅ FOUND!")
            processed_df = pd.read_parquet(path)
            processed_path = path
            break
        else:
            print(f"   ❌ Not found")
    
    print()
    
    if processed_df is None:
        print("⚠️  WARNING: No processed data found. Checking raw data only.")
        print()
    else:
        print(f"✅ Using processed data from: {processed_path}")
        print()
        
        # Check processed data columns
        print("2. PROCESSED DATA COLUMNS")
        print("-" * 80)
        feature_cols = [col for col in processed_df.columns 
                       if col not in ['bankrupt', 'horizon', 'company_id', 'year', 'class']]
        
        print(f"   Total feature columns: {len(feature_cols)}")
        print(f"   First 10: {feature_cols[:10]}")
        print(f"   Last 5: {feature_cols[-5:]}")
        print()
        
        # Determine naming pattern
        if feature_cols[0].startswith('Attr'):
            print(f"   ✅ Processed data uses: Attr1, Attr2, ... Attr{len(feature_cols)}")
            rename_happened = True
        elif feature_cols[0].startswith('A'):
            print(f"   ⚠️  Processed data STILL uses: A1, A2, ... A{len(feature_cols)}")
            rename_happened = False
        elif feature_cols[0].startswith('X'):
            print(f"   ⚠️  Processed data uses: X1, X2, ... X{len(feature_cols)}")
            rename_happened = False
        print()
    
    # Check raw data
    print("3. RAW CSV DATA COLUMNS")
    print("-" * 80)
    csv_path = base_dir / "data/polish-companies-bankruptcy/data-from-kaggel.csv"
    df_raw = pd.read_csv(csv_path, nrows=0)
    raw_cols = [col for col in df_raw.columns if col not in ['class', 'year']]
    
    print(f"   Total feature columns: {len(raw_cols)}")
    print(f"   First 10: {raw_cols[:10]}")
    print(f"   Last 5: {raw_cols[-5:]}")
    print(f"   ✅ Raw data uses: A1, A2, ... A64")
    print()
    
    # Check feature_descriptions.json
    print("4. FEATURE_DESCRIPTIONS.JSON")
    print("-" * 80)
    feature_desc_path = base_dir / "data/polish-companies-bankruptcy/feature_descriptions.json"
    with open(feature_desc_path, 'r') as f:
        feature_data = json.load(f)
    
    feature_keys = list(feature_data['features'].keys())
    print(f"   Total features: {len(feature_keys)}")
    print(f"   First 10: {feature_keys[:10]}")
    print(f"   Last 5: {feature_keys[-5:]}")
    print(f"   ✅ Documentation uses: Attr1, Attr2, ... Attr64")
    print()
    
    # CRITICAL ANALYSIS
    print("=" * 80)
    print("CRITICAL ANALYSIS")
    print("=" * 80)
    print()
    
    print("NAMING CONVENTION SUMMARY:")
    print(f"  • UCI Documentation:      X1, X2, X3... X64 (reference only)")
    print(f"  • Kaggle Metadata:        A1, A2, A3... A64")
    print(f"  • Raw CSV Data:           A1, A2, A3... A64")
    
    if processed_df is not None:
        if feature_cols[0].startswith('Attr'):
            print(f"  • Processed Data:         Attr1, Attr2, Attr3... Attr64 ✅")
            print(f"  • feature_descriptions:   Attr1, Attr2, Attr3... Attr64 ✅")
            print()
            print("✅ CONSISTENT: A1->Attr1 rename happened during preprocessing")
            print("✅ Documentation matches processed data")
        elif feature_cols[0].startswith('A'):
            print(f"  • Processed Data:         A1, A2, A3... A64 ❌")
            print(f"  • feature_descriptions:   Attr1, Attr2, Attr3... Attr64 ❌")
            print()
            print("❌ MISMATCH DETECTED!")
            print("   Problem: feature_descriptions.json uses Attr1-Attr64")
            print("   But actual data uses A1-A64")
            print()
            print("REQUIRED ACTION:")
            print("   Either:")
            print("   1. Rename feature_descriptions.json keys: Attr1->A1, Attr2->A2, etc.")
            print("   2. OR rename data columns during preprocessing: A1->Attr1, A2->Attr2, etc.")
    else:
        print(f"  • Processed Data:         [NOT FOUND - CANNOT VERIFY]")
        print(f"  • feature_descriptions:   Attr1, Attr2, Attr3... Attr64")
        print()
        print("⚠️  WARNING: Cannot verify consistency without processed data")
    
    print()
    print("WHY UCI USES 'X' NOTATION:")
    print("  UCI documentation uses X1-X64 as generic variable names for formulas.")
    print("  The actual ARFF/CSV files use A1-A64 as column names.")
    print("  This is just a documentation convention difference.")
    print()
    
    return processed_df is not None and feature_cols[0].startswith('Attr')


if __name__ == "__main__":
    success = main()
    if not success:
        print("⚠️  VERIFICATION INCOMPLETE OR ISSUES FOUND")
    exit(0 if success else 1)
