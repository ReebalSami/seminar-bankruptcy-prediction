"""
Create feature mapping Excel with ORIGINAL labels from dataset metadata.

This script properly extracts:
- Polish: Original Attr names + descriptions from feature_descriptions.json
- American: Original X names + descriptions from american-metadata.json  
- Taiwan: Original descriptive column names from CSV

Each sheet shows:
1. Original Feature Code/Name (from dataset)
2. Original Description (from metadata)
3. VIF (if available)
4. Category (if available)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

def main():
    """Create properly labeled feature mapping Excel."""
    
    base_dir = Path(__file__).resolve().parents[2]
    
    # =====================================================
    # 1. POLISH FEATURES - Load from feature_descriptions.json
    # =====================================================
    print("Loading Polish feature descriptions...")
    polish_desc_path = base_dir / "data/polish-companies-bankruptcy/feature_descriptions.json"
    
    with open(polish_desc_path, 'r') as f:
        polish_meta = json.load(f)
    
    polish_features = polish_meta['features']
    polish_data = []
    
    for attr_code, attr_info in polish_features.items():
        polish_data.append({
            'Feature_Code': attr_code,
            'Original_Name': attr_info['name'],
            'Short_Name': attr_info.get('short_name', ''),
            'Category': attr_info.get('category', ''),
            'Formula': attr_info.get('formula', ''),
            'Interpretation': attr_info.get('interpretation', ''),
            'VIF': ''  # Will be filled later if calculated
        })
    
    polish_df = pd.DataFrame(polish_data)
    print(f"✓ Polish: {len(polish_df)} features loaded")
    
    # =====================================================
    # 2. AMERICAN FEATURES - Load from american-metadata.json
    # =====================================================
    print("Loading American feature descriptions...")
    american_meta_path = base_dir / "data/NYSE-and-NASDAQ-companies/american-metadata.json"
    
    with open(american_meta_path, 'r') as f:
        american_meta = json.load(f)
    
    # Extract field descriptions from recordSet
    american_fields = [
        f for f in american_meta['recordSet'][0]['field'] 
        if f.get('@type') == 'cr:Field' and f.get('name') != 'Status_Label'
    ]
    
    american_data = []
    for field in american_fields:
        feature_name = field.get('name', '')
        if feature_name.startswith('X'):  # Only X1-X18 features
            american_data.append({
                'Feature_Code': feature_name,
                'Original_Description': field.get('description', ''),
                'Data_Type': field.get('dataType', [''])[0].replace('sc:', ''),
                'VIF': ''  # Will be filled later if calculated
            })
    
    american_df = pd.DataFrame(american_data)
    print(f"✓ American: {len(american_df)} features loaded")
    
    # =====================================================
    # 3. TAIWAN FEATURES - Load from CSV headers
    # =====================================================
    print("Loading Taiwan feature names from CSV...")
    taiwan_csv_path = base_dir / "data/taiwan-economic-journal/taiwan-bankruptcy.csv"
    
    taiwan_df_temp = pd.read_csv(taiwan_csv_path, nrows=0)
    taiwan_columns = [col for col in taiwan_df_temp.columns if col != 'Bankrupt?']
    
    taiwan_data = []
    for idx, col_name in enumerate(taiwan_columns, start=1):
        taiwan_data.append({
            'Feature_Number': f'F{idx:02d}',  # F01, F02, etc. for reference
            'Original_Feature_Name': col_name.strip(),  # Remove leading/trailing spaces
            'VIF': ''  # Will be filled later if calculated
        })
    
    taiwan_df = pd.DataFrame(taiwan_data)
    print(f"✓ Taiwan: {len(taiwan_df)} features loaded")
    
    # =====================================================
    # 4. LOAD VIF RESULTS FOR ALL DATASETS
    # =====================================================
    print("\nLoading VIF results...")
    
    # Polish VIF
    polish_vif_path = base_dir / "results/script_outputs/10d_remediation_save/vif_all_features.csv"
    if polish_vif_path.exists():
        polish_vif = pd.read_csv(polish_vif_path)
        polish_vif_dict = dict(zip(polish_vif['Feature'], polish_vif['VIF']))
        polish_df['VIF'] = polish_df['Feature_Code'].map(polish_vif_dict)
        polish_df['VIF_Status'] = polish_df['VIF'].apply(
            lambda x: 'High (>10)' if pd.notna(x) and x > 10 else ('Moderate (5-10)' if pd.notna(x) and x > 5 else 'Low (<5)')
        )
        print(f"  ✓ Polish VIF loaded ({len(polish_vif)} features)")
    else:
        print(f"  ⚠️  Polish VIF not found at {polish_vif_path}")
    
    # American VIF  
    american_vif_path = base_dir / "results/script_outputs/02_american/02b_vif_remediation/vif_all_features.csv"
    if american_vif_path.exists():
        american_vif = pd.read_csv(american_vif_path)
        # Handle both 'Feature'/'VIF' and 'feature'/'vif' column names
        feature_col = 'Feature' if 'Feature' in american_vif.columns else 'feature'
        vif_col = 'VIF' if 'VIF' in american_vif.columns else 'vif'
        american_vif_dict = dict(zip(american_vif[feature_col], american_vif[vif_col]))
        american_df['VIF'] = american_df['Feature_Code'].map(american_vif_dict)
        american_df['VIF_Status'] = american_df['VIF'].apply(
            lambda x: 'High (>10)' if pd.notna(x) and x > 10 else ('Moderate (5-10)' if pd.notna(x) and x > 5 else 'Low (<5)')
        )
        print(f"  ✓ American VIF loaded ({len(american_vif)} features)")
    else:
        print(f"  ⚠️  American VIF not found at {american_vif_path}")
    
    # Taiwan VIF
    taiwan_vif_path = base_dir / "results/script_outputs/03_taiwan/02b_vif_remediation/vif_all_features.csv"
    if taiwan_vif_path.exists():
        taiwan_vif = pd.read_csv(taiwan_vif_path)
        # Handle both 'Feature'/'VIF' and 'feature'/'vif' column names
        feature_col = 'Feature' if 'Feature' in taiwan_vif.columns else 'feature'
        vif_col = 'VIF' if 'VIF' in taiwan_vif.columns else 'vif'
        
        # Taiwan VIF uses F-codes (F01-F95), need to map to feature names
        # Create F-code to name mapping
        taiwan_csv_temp = pd.read_csv(base_dir / "data/taiwan-economic-journal/taiwan-bankruptcy.csv", nrows=0)
        taiwan_feature_names = [c.strip() for c in taiwan_csv_temp.columns if c != 'Bankrupt?']
        f_code_to_name = {f'F{i+1:02d}': name for i, name in enumerate(taiwan_feature_names)}
        
        # Map F-codes to descriptive names
        taiwan_vif['Feature_Name'] = taiwan_vif[feature_col].map(f_code_to_name)
        taiwan_vif_dict = dict(zip(taiwan_vif['Feature_Name'], taiwan_vif[vif_col]))
        
        taiwan_df['VIF'] = taiwan_df['Original_Feature_Name'].map(taiwan_vif_dict)
        taiwan_df['VIF_Status'] = taiwan_df['VIF'].apply(
            lambda x: 'High (>10)' if pd.notna(x) and x > 10 else ('Moderate (5-10)' if pd.notna(x) and x > 5 else 'Low (<5)')
        )
        print(f"  ✓ Taiwan VIF loaded ({len(taiwan_vif)} features)")
    else:
        print(f"  ⚠️  Taiwan VIF not found at {taiwan_vif_path}")
    
    # =====================================================
    # 5. CREATE EXCEL WITH ORIGINAL LABELS + VIF
    # =====================================================
    print("\nCreating Excel file with original dataset labels and VIF...")
    
    output_path = base_dir / "docs/FEATURE_MAPPING_WITH_ORIGINAL_LABELS.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            'Dataset': ['Polish', 'American', 'Taiwan'],
            'Feature_Count': [len(polish_df), len(american_df), len(taiwan_df)],
            'Original_Format': [
                'A1-A64 with descriptions + VIF',
                'X1-X18 with descriptions + VIF',
                'Descriptive column names + VIF'
            ],
            'Metadata_Source': [
                'feature_descriptions.json',
                'american-metadata.json',
                'CSV column headers'
            ],
            'VIF_Analyzed': [
                'Yes' if 'VIF' in polish_df.columns else 'No',
                'Yes' if 'VIF' in american_df.columns else 'No',
                'Yes' if 'VIF' in taiwan_df.columns else 'No'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # VIF Summary sheet
        vif_summary_data = []
        for dataset_name, df in [('Polish', polish_df), ('American', american_df), ('Taiwan', taiwan_df)]:
            if 'VIF' in df.columns:
                vif_vals = df['VIF'].replace([np.inf, -np.inf], np.nan).dropna()
                vif_summary_data.append({
                    'Dataset': dataset_name,
                    'Total_Features': len(df),
                    'Features_with_VIF': len(vif_vals),
                    'High_VIF_Count': len(vif_vals[vif_vals > 10]),
                    'Moderate_VIF_Count': len(vif_vals[(vif_vals > 5) & (vif_vals <= 10)]),
                    'Low_VIF_Count': len(vif_vals[vif_vals <= 5]),
                    'Max_VIF': vif_vals.max() if len(vif_vals) > 0 else None,
                    'Mean_VIF': vif_vals.mean() if len(vif_vals) > 0 else None,
                    'Median_VIF': vif_vals.median() if len(vif_vals) > 0 else None
                })
        if vif_summary_data:
            vif_summary_df = pd.DataFrame(vif_summary_data)
            vif_summary_df.to_excel(writer, sheet_name='VIF_Summary', index=False)
        
        # Polish sheet
        polish_df.to_excel(writer, sheet_name='Polish_Original_Labels', index=False)
        
        # American sheet
        american_df.to_excel(writer, sheet_name='American_Original_Labels', index=False)
        
        # Taiwan sheet
        taiwan_df.to_excel(writer, sheet_name='Taiwan_Original_Labels', index=False)
        
        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                adjusted_width = min(max_length + 2, 80)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"\n✅ Excel file created: {output_path}")
    print("\nFile structure:")
    print("  • Summary: Overview of all datasets with metadata sources")
    print("  • Polish_Original_Labels: A1-A64 with full descriptions from feature_descriptions.json")
    print("  • American_Original_Labels: X1-X18 with descriptions from american-metadata.json")
    print("  • Taiwan_Original_Labels: Original descriptive column names from CSV")
    
    return output_path


if __name__ == "__main__":
    try:
        output_file = main()
        print(f"\n🎉 SUCCESS! Feature mapping with original labels created at:")
        print(f"   {output_file}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
