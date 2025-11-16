"""
Investigate the actual column naming in Polish dataset files.

Check:
1. What column names are in the CSV file (A1, X1, or Attr1?)
2. Why UCI documentation says X1-X64
3. Why Kaggle metadata says A1-A64  
4. Why we created Attr1-Attr64
"""

import pandas as pd
import json
from pathlib import Path

def main():
    """Check all Polish data sources for column naming conventions."""
    
    base_dir = Path(__file__).resolve().parents[2]
    
    print("=" * 80)
    print("INVESTIGATING POLISH DATASET COLUMN NAMING")
    print("=" * 80)
    print()
    
    # 1. Check the actual CSV file
    print("1. ACTUAL CSV FILE COLUMNS")
    print("-" * 80)
    csv_path = base_dir / "data/polish-companies-bankruptcy/data-from-kaggel.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path, nrows=0)
        actual_columns = df.columns.tolist()
        
        print(f"   Total columns: {len(actual_columns)}")
        print(f"   First 10 columns: {actual_columns[:10]}")
        print(f"   Last 5 columns: {actual_columns[-5:]}")
        
        # Determine naming pattern
        if actual_columns[0].startswith('Attr'):
            print(f"   ✅ Naming pattern: Attr1, Attr2, ... Attr{len(actual_columns)-1}")
        elif actual_columns[0].startswith('X'):
            print(f"   ✅ Naming pattern: X1, X2, ... X{len(actual_columns)-1}")
        elif actual_columns[0].startswith('A'):
            print(f"   ✅ Naming pattern: A1, A2, ... A{len(actual_columns)-1}")
        else:
            print(f"   ⚠️  Unexpected naming pattern: {actual_columns[0]}")
    else:
        print(f"   ⚠️  CSV file not found at {csv_path}")
    
    print()
    
    # 2. Check Kaggle metadata JSON
    print("2. KAGGLE METADATA JSON (polish-metadata.json)")
    print("-" * 80)
    kaggle_meta_path = base_dir / "data/polish-companies-bankruptcy/polish-metadata.json"
    
    if kaggle_meta_path.exists():
        with open(kaggle_meta_path, 'r') as f:
            kaggle_meta = json.load(f)
        
        # Extract field names
        fields = kaggle_meta.get('recordSet', [{}])[0].get('field', [])
        field_names = [f.get('name') for f in fields if 'name' in f]
        
        print(f"   Total fields: {len(field_names)}")
        print(f"   First 10 fields: {field_names[:10]}")
        print(f"   Last 5 fields: {field_names[-5:]}")
        
        if field_names and field_names[0].startswith('A'):
            print(f"   ✅ Naming pattern: A1, A2, ... A{len([f for f in field_names if f.startswith('A')])}")
    else:
        print(f"   ⚠️  Kaggle metadata not found")
    
    print()
    
    # 3. Check our feature_descriptions.json
    print("3. OUR FEATURE_DESCRIPTIONS.JSON")
    print("-" * 80)
    feature_desc_path = base_dir / "data/polish-companies-bankruptcy/feature_descriptions.json"
    
    if feature_desc_path.exists():
        with open(feature_desc_path, 'r') as f:
            feature_data = json.load(f)
        
        feature_keys = list(feature_data.get('features', {}).keys())
        
        print(f"   Total features: {len(feature_keys)}")
        print(f"   First 10 features: {feature_keys[:10]}")
        print(f"   Last 5 features: {feature_keys[-5:]}")
        
        if feature_keys and feature_keys[0].startswith('Attr'):
            print(f"   ✅ Naming pattern: Attr1, Attr2, ... Attr{len(feature_keys)}")
    else:
        print(f"   ⚠️  Feature descriptions not found")
    
    print()
    
    # 4. Check processed data
    print("4. OUR PROCESSED PARQUET FILE")
    print("-" * 80)
    processed_path = base_dir / "data/processed/poland/poland_clean_full.parquet"
    
    if processed_path.exists():
        df_processed = pd.read_parquet(processed_path)
        processed_columns = [col for col in df_processed.columns if col not in ['bankrupt', 'horizon', 'company_id', 'year']]
        
        print(f"   Total feature columns: {len(processed_columns)}")
        print(f"   First 10 columns: {processed_columns[:10]}")
        print(f"   Last 5 columns: {processed_columns[-5:]}")
        
        if processed_columns and processed_columns[0].startswith('Attr'):
            print(f"   ✅ Naming pattern: Attr1, Attr2, ... Attr{len(processed_columns)}")
        elif processed_columns[0].startswith('X'):
            print(f"   ✅ Naming pattern: X1, X2, ...")
        elif processed_columns[0].startswith('A'):
            print(f"   ✅ Naming pattern: A1, A2, ...")
    else:
        print(f"   ⚠️  Processed file not found")
    
    print()
    
    # 5. Summary and Analysis
    print("=" * 80)
    print("SUMMARY & ANALYSIS")
    print("=" * 80)
    print()
    print("UCI Documentation says:  X1, X2, X3, ... X64")
    print("Kaggle metadata says:    A1, A2, A3, ... A64")
    print("Our files use:           [TO BE DETERMINED FROM ABOVE]")
    print()
    print("CRITICAL QUESTIONS:")
    print("1. Did the original ARFF files use A1-A64?")
    print("2. Did we rename them to Attr1-Attr64 during processing?")
    print("3. Is there a mismatch between our data and our documentation?")
    print()
    print("ACTION REQUIRED:")
    print("Verify that column names in actual data match feature_descriptions.json keys")
    print()


if __name__ == "__main__":
    main()
