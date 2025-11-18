"""Check 02b detailed columns."""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results' / '02_exploratory_analysis'

print("Checking 02b_H1_univariate_tests.xlsx Metadata sheet:")
df = pd.read_excel(RESULTS_DIR / '02b_H1_univariate_tests.xlsx', sheet_name='Metadata')
print(f"  Columns: {list(df.columns)}")
for col in df.columns:
    print(f"    {col}: {df[col].values[0]}")

print("\nChecking if Bankrupt_N exists in other sheets...")
xl = pd.ExcelFile(RESULTS_DIR / '02b_H1_univariate_tests.xlsx')
print(f"  Available sheets: {xl.sheet_names}")
