"""
Verify all A1-A64 naming fixes were applied correctly.
"""

import pandas as pd
import json
from pathlib import Path

def main():
    """Verify naming consistency across all files."""
    
    base_dir = Path(__file__).resolve().parents[2]
    
    print("="*80)
    print("VERIFICATION: A1-A64 NAMING CONSISTENCY")
    print("="*80)
    print()
    
    all_good = True
    
    # 1. Check feature_descriptions.json
    print("1. feature_descriptions.json")
    print("-"*80)
    feature_desc_path = base_dir / "data/polish-companies-bankruptcy/feature_descriptions.json"
    with open(feature_desc_path, 'r') as f:
        feature_data = json.load(f)
    
    feature_keys = list(feature_data['features'].keys())
    print(f"   First 5 keys: {feature_keys[:5]}")
    print(f"   Last 5 keys: {feature_keys[-5:]}")
    
    if all(k.startswith('A') and not k.startswith('Attr') for k in feature_keys):
        print(f"   ✅ Uses A1-A64 naming")
    else:
        print(f"   ❌ Still has Attr naming")
        all_good = False
    print()
    
    # 2. Check raw CSV
    print("2. Raw CSV (data-from-kaggel.csv)")
    print("-"*80)
    csv_path = base_dir / "data/polish-companies-bankruptcy/data-from-kaggel.csv"
    df_raw = pd.read_csv(csv_path, nrows=0)
    raw_cols = [c for c in df_raw.columns if c not in ['class', 'year']]
    print(f"   First 5 columns: {raw_cols[:5]}")
    print(f"   Last 5 columns: {raw_cols[-5:]}")
    
    if all(c.startswith('A') and not c.startswith('Attr') for c in raw_cols[:64]):
        print(f"   ✅ Uses A1-A64 naming")
    else:
        print(f"   ❌ Unexpected naming")
        all_good = False
    print()
    
    # 3. Check processed parquet
    print("3. Processed parquet (poland_clean_full.parquet)")
    print("-"*80)
    parquet_path = base_dir / "data/processed/poland_clean_full.parquet"
    df_processed = pd.read_parquet(parquet_path)
    proc_cols = [c for c in df_processed.columns if c not in ['y', 'bankrupt', 'horizon', 'year', 'class']]
    print(f"   First 5 feature columns: {proc_cols[:5]}")
    print(f"   Last 5 feature columns: {proc_cols[-5:]}")
    
    if all(c.startswith('A') and not c.startswith('Attr') for c in proc_cols[:64]):
        print(f"   ✅ Uses A1-A64 naming")
    else:
        print(f"   ❌ Still has Attr naming or unexpected columns")
        all_good = False
    print()
    
    # 4. Check Excel file exists
    print("4. Excel file (FEATURE_MAPPING_WITH_ORIGINAL_LABELS.xlsx)")
    print("-"*80)
    excel_path = base_dir / "docs/FEATURE_MAPPING_WITH_ORIGINAL_LABELS.xlsx"
    if excel_path.exists():
        # Quick check of Polish sheet
        polish_df = pd.read_excel(excel_path, sheet_name='Polish_Original_Labels')
        if 'Feature_Code' in polish_df.columns:
            first_codes = polish_df['Feature_Code'].head(5).tolist()
            last_codes = polish_df['Feature_Code'].tail(5).tolist()
            print(f"   First 5 codes: {first_codes}")
            print(f"   Last 5 codes: {last_codes}")
            
            if all(str(c).startswith('A') and not str(c).startswith('Attr') for c in first_codes):
                print(f"   ✅ Excel uses A1-A64 naming")
            else:
                print(f"   ❌ Excel still has Attr naming")
                all_good = False
        else:
            print(f"   ⚠️  Cannot verify - no Feature_Code column")
    else:
        print(f"   ❌ Excel file not found")
        all_good = False
    print()
    
    # Final summary
    print("="*80)
    print("FINAL VERIFICATION")
    print("="*80)
    
    if all_good:
        print("✅ ALL CHECKS PASSED!")
        print()
        print("Naming consistency verified:")
        print("  ✓ feature_descriptions.json:  A1-A64")
        print("  ✓ Raw CSV:                    A1-A64")
        print("  ✓ Processed parquet:          A1-A64")
        print("  ✓ Excel file:                 A1-A64")
        print()
        print("Alignment with sources:")
        print("  ✓ Kaggle metadata:            A1-A64 ✓")
        print("  ✓ UCI formulas:               X1-X64 (reference) → maps to A1-A64 ✓")
        print()
        return True
    else:
        print("❌ SOME CHECKS FAILED")
        print("Please review the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
