"""Check what columns exist in Phase 02 Excel files."""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results' / '02_exploratory_analysis'

print("Checking 02a_H1_distributions.xlsx:")
df = pd.read_excel(RESULTS_DIR / '02a_H1_distributions.xlsx', sheet_name='Metadata')
print(f"  Columns: {list(df.columns)}")
print(f"  Data:\n{df}")

print("\nChecking 02b_H1_univariate_tests.xlsx:")
df = pd.read_excel(RESULTS_DIR / '02b_H1_univariate_tests.xlsx', sheet_name='Metadata')
print(f"  Columns: {list(df.columns)}")
print(f"  Data:\n{df}")

print("\nChecking 02c_H1_correlation.xlsx:")
df = pd.read_excel(RESULTS_DIR / '02c_H1_correlation.xlsx', sheet_name='Metadata')
print(f"  Columns: {list(df.columns)}")
print(f"  Data:\n{df}")
