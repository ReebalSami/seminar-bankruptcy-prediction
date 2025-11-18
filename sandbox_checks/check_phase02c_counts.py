from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

all_counts = {}
excel = PROJECT_ROOT / 'results' / '02_exploratory_analysis' / '02c_ALL_correlation.xlsx'
if not excel.exists():
    raise SystemExit(f"Missing consolidated Excel: {excel}")

df_overview = pd.read_excel(excel, sheet_name='Overview')
# Expected columns: Horizon, High_Correlations, Economically_Plausible, Economically_Implausible
for row in df_overview.itertuples():
    all_counts[f'H{row.Horizon}'] = {
        'High_Correlations': int(row.High_Correlations),
        'Economically_Plausible': int(row.Economically_Plausible),
        'Economically_Implausible': int(row.Economically_Implausible),
    }

print(json.dumps(all_counts, indent=2))
