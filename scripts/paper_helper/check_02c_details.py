"""Check 02c detailed columns."""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results' / '02_exploratory_analysis'

print("Checking 02c_H1_correlation.xlsx Metadata sheet:")
df = pd.read_excel(RESULTS_DIR / '02c_H1_correlation.xlsx', sheet_name='Metadata')
print(f"  Columns: {list(df.columns)}")
for col in df.columns:
    print(f"    {col}: {df[col].values[0]}")

print("\nChecking Economic_Validation sheet:")
df_econ = pd.read_excel(RESULTS_DIR / '02c_H1_correlation.xlsx', sheet_name='Economic_Validation')
print(f"  Columns: {list(df_econ.columns)}")
print(f"  First 3 rows:\n{df_econ.head(3)}")
