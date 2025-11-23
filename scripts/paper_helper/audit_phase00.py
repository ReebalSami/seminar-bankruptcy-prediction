import pandas as pd
import json
import os

# Paths
BASE_DIR = "/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction"
RESULTS_DIR = os.path.join(BASE_DIR, "results/00_foundation")
OVERVIEW_FILE = os.path.join(RESULTS_DIR, "00a_polish_overview.xlsx")
TEMPORAL_FILE = os.path.join(RESULTS_DIR, "00c_temporal_structure.xlsx")
QUALITY_FILE = os.path.join(RESULTS_DIR, "00d_data_quality.xlsx")

def load_excel_sheet(file_path, sheet_name=0):
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def audit_foundation():
    print("=== AUDIT PHASE 00: FOUNDATION ===\n")

    # 1. Overview & Structure
    df_overview = load_excel_sheet(OVERVIEW_FILE, sheet_name="Summary")
    if df_overview is not None:
        print("--- Dataset Overview ---")
        print(df_overview.to_string())
        # Check Total N
        try:
            total_n = df_overview.loc[df_overview['Metric'] == 'Total Observations', 'Value'].values[0]
            print(f"\nTotal N check: {total_n} (Expected: 43405)")
        except IndexError:
            print("\nCould not find 'Total Observations' in Overview")

    # 2. Temporal Structure (Horizons)
    df_temporal = load_excel_sheet(TEMPORAL_FILE)
    if df_temporal is not None:
        print("\n--- Temporal Structure (Horizons) ---")
        # Adjust column names if necessary based on file content
        print(df_temporal.to_string(index=False))

    # 3. Data Quality (Missing)
    df_quality = load_excel_sheet(QUALITY_FILE, sheet_name="Missing_Values")
    if df_quality is not None:
        print("\n--- Missing Values (Top 5) ---")
        print(df_quality.head(5).to_string(index=False))
        
        # Check A37 specifically - usually 'Attr37' or similar
        # We will search for it
        a37 = df_quality[df_quality.iloc[:, 0].astype(str).str.contains('37')]
        if not a37.empty:
             print(f"\nA37 Row:\n{a37.to_string(index=False)}")

    # 4. Duplicates
    # Check Quality Summary for duplicates
    df_q_summary = load_excel_sheet(QUALITY_FILE, sheet_name="Summary")
    if df_q_summary is not None:
        print("\n--- Quality Summary (Duplicates Check) ---")
        print(df_q_summary.to_string())

if __name__ == "__main__":
    audit_foundation()
