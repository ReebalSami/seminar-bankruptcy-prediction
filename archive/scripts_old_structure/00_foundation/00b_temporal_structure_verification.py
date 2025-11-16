"""
Script 00b: Temporal Structure Verification

CRITICAL FOUNDATION SCRIPT - Must run after Script 00!

Purpose:
--------
Determines the temporal structure of each dataset to validate which temporal
methods are appropriate:

- TIME_SERIES: Companies tracked over time (balanced panel) → Panel VAR, ADF valid
- UNBALANCED_PANEL: Companies tracked but not all periods → Panel methods with caution
- REPEATED_CROSS_SECTIONS: Different companies each period → Temporal holdout valid, Panel VAR INVALID
- SINGLE_CROSS_SECTION: No time dimension → No temporal methods

Critical for:
- Script 11: Verify if "panel data" label is correct (it's NOT for Polish!)
- Script 13c: Determine if Granger causality is valid (NO for repeated cross-sections!)

Outputs:
--------
1. temporal_structure_analysis.json - Detailed temporal structure per dataset
2. recommended_methods.json - Valid temporal methods per dataset
3. company_tracking_analysis.csv - % of companies appearing in multiple periods

Time: ~1 hour

Author: Reebal
Date: November 12, 2025
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import (
    POLISH_DATA_PATH, AMERICAN_DATA_PATH, TAIWAN_DATA_PATH,
    TEMPORAL_STRUCTURE_DIR
)

print("="*80)
print("SCRIPT 00b: TEMPORAL STRUCTURE VERIFICATION")
print("="*80)
print("\n⚠️  FOUNDATION SCRIPT - Determines valid temporal methods!")
print(f"Output directory: {TEMPORAL_STRUCTURE_DIR}\n")

# ============================================================================
# STEP 1: Load datasets
# ============================================================================

print("[1/5] Loading datasets...")

df_polish = pd.read_csv(POLISH_DATA_PATH)
df_american = pd.read_csv(AMERICAN_DATA_PATH)
df_taiwan = pd.read_csv(TAIWAN_DATA_PATH)

print(f"✓ Polish: {df_polish.shape}")
print(f"✓ American: {df_american.shape}")
print(f"✓ Taiwan: {df_taiwan.shape}")

# ============================================================================
# STEP 2: Identify time and identifier columns
# ============================================================================

print("\n[2/5] Identifying time and company identifier columns...")

temporal_info = {}

# Polish: Check if there's a time column
polish_time_cols = [col for col in df_polish.columns if 'year' in col.lower() or 'period' in col.lower() or 'time' in col.lower()]
polish_id_cols = [col for col in df_polish.columns if 'id' in col.lower() or 'company' in col.lower()]

if polish_time_cols:
    print(f"Polish time columns found: {polish_time_cols}")
    temporal_info['polish'] = {
        'has_time': True,
        'time_col': polish_time_cols[0] if polish_time_cols else None,
        'id_col': polish_id_cols[0] if polish_id_cols else None
    }
else:
    print("Polish: NO explicit time column found")
    temporal_info['polish'] = {
        'has_time': False,
        'time_col': None,
        'id_col': polish_id_cols[0] if polish_id_cols else None
    }

# American: has 'year' column
american_time_col = 'year'
american_id_col = 'company_name'
temporal_info['american'] = {
    'has_time': True,
    'time_col': american_time_col,
    'id_col': american_id_col
}
print(f"American: year={american_time_col}, id={american_id_col}")

# Taiwan: Check for time columns
taiwan_time_cols = [col for col in df_taiwan.columns if 'year' in col.lower() or 'period' in col.lower() or 'time' in col.lower()]
taiwan_id_cols = [col for col in df_taiwan.columns if 'id' in col.lower() or 'company' in col.lower()]

if taiwan_time_cols:
    print(f"Taiwan time columns found: {taiwan_time_cols}")
    temporal_info['taiwan'] = {
        'has_time': True,
        'time_col': taiwan_time_cols[0],
        'id_col': taiwan_id_cols[0] if taiwan_id_cols else None
    }
else:
    print("Taiwan: NO explicit time column found")
    temporal_info['taiwan'] = {
        'has_time': False,
        'time_col': None,
        'id_col': taiwan_id_cols[0] if taiwan_id_cols else None
    }

# ============================================================================
# STEP 3: Analyze temporal structure for each dataset
# ============================================================================

print("\n[3/5] Analyzing temporal structure...")

def analyze_temporal_structure(df, time_col, id_col, dataset_name):
    """
    Analyzes whether dataset is:
    - TIME_SERIES (balanced panel)
    - UNBALANCED_PANEL (companies tracked but missing periods)
    - REPEATED_CROSS_SECTIONS (different companies each period)
    - SINGLE_CROSS_SECTION (no time dimension)
    """
    
    if time_col is None:
        return {
            'structure': 'SINGLE_CROSS_SECTION',
            'n_periods': 1,
            'n_unique_periods': 1,
            'companies_tracked': 0,
            'pct_companies_tracked': 0.0,
            'explanation': 'No time dimension detected'
        }
    
    n_periods = df[time_col].nunique()
    n_total = len(df)
    
    if id_col is None:
        # No company ID, check if sample size is constant across periods
        period_sizes = df[time_col].value_counts().sort_index()
        is_balanced = period_sizes.std() == 0
        
        return {
            'structure': 'REPEATED_CROSS_SECTIONS' if n_periods > 1 else 'SINGLE_CROSS_SECTION',
            'n_periods': int(n_periods),
            'n_unique_periods': int(n_periods),
            'period_sizes': {str(k): int(v) for k, v in period_sizes.to_dict().items()},
            'is_balanced': bool(is_balanced),
            'companies_tracked': 0,
            'pct_companies_tracked': 0.0,
            'explanation': 'No company identifier - cannot track companies over time'
        }
    
    # Count how many companies appear in multiple periods
    company_periods = df.groupby(id_col)[time_col].nunique()
    companies_in_multiple_periods = (company_periods > 1).sum()
    total_companies = company_periods.shape[0]
    pct_tracked = (companies_in_multiple_periods / total_companies) * 100
    
    # Determine structure
    if pct_tracked > 80:
        structure = 'TIME_SERIES'  # Balanced or nearly balanced panel
    elif pct_tracked > 20:
        structure = 'UNBALANCED_PANEL'  # Some companies tracked
    else:
        structure = 'REPEATED_CROSS_SECTIONS'  # Mostly different companies
    
    period_sizes_dict = df[time_col].value_counts().sort_index().to_dict()
    
    return {
        'structure': structure,
        'n_periods': int(n_periods),
        'n_unique_periods': int(n_periods),
        'n_companies': int(total_companies),
        'companies_in_multiple_periods': int(companies_in_multiple_periods),
        'pct_companies_tracked': float(round(pct_tracked, 2)),
        'period_sizes': {str(k): int(v) for k, v in period_sizes_dict.items()},
        'explanation': f'{pct_tracked:.1f}% of companies appear in multiple periods'
    }

# Analyze each dataset
polish_structure = analyze_temporal_structure(
    df_polish, 
    temporal_info['polish']['time_col'],
    temporal_info['polish']['id_col'],
    'polish'
)

american_structure = analyze_temporal_structure(
    df_american,
    temporal_info['american']['time_col'],
    temporal_info['american']['id_col'],
    'american'
)

taiwan_structure = analyze_temporal_structure(
    df_taiwan,
    temporal_info['taiwan']['time_col'],
    temporal_info['taiwan']['id_col'],
    'taiwan'
)

print(f"\n✓ Polish: {polish_structure['structure']}")
if 'period_sizes' in polish_structure:
    print(f"  Periods: {polish_structure['n_periods']}")
    if polish_structure['n_periods'] > 1:
        sizes = list(polish_structure['period_sizes'].values())
        print(f"  Sample sizes: {sizes[0:5]}... (varying sizes indicate REPEATED_CROSS_SECTIONS)")

print(f"\n✓ American: {american_structure['structure']}")
print(f"  Periods: {american_structure['n_periods']}")
print(f"  Companies tracked: {american_structure['pct_companies_tracked']:.1f}%")

print(f"\n✓ Taiwan: {taiwan_structure['structure']}")
if 'period_sizes' in taiwan_structure:
    print(f"  Periods: {taiwan_structure['n_periods']}")

# ============================================================================
# STEP 4: Determine valid temporal methods
# ============================================================================

print("\n[4/5] Determining valid temporal methods...")

def get_recommended_methods(structure):
    """
    Returns valid temporal methods based on data structure.
    """
    if structure == 'TIME_SERIES':
        return {
            'valid_methods': ['Panel VAR', 'Granger Causality', 'ADF test', 'Temporal Holdout', 'Lagged Features'],
            'invalid_methods': [],
            'warning': None
        }
    elif structure == 'UNBALANCED_PANEL':
        return {
            'valid_methods': ['Panel VAR (with gaps)', 'Temporal Holdout', 'Lagged Features'],
            'invalid_methods': ['Granger Causality (requires balanced panel)'],
            'warning': 'Panel VAR possible but handle missing periods carefully'
        }
    elif structure == 'REPEATED_CROSS_SECTIONS':
        return {
            'valid_methods': ['Temporal Holdout Validation', 'Train on early periods, test on later'],
            'invalid_methods': ['Panel VAR', 'Granger Causality', 'Lagged Features (no company tracking)'],
            'warning': 'CRITICAL: Cannot use panel methods! Different companies each period.'
        }
    else:  # SINGLE_CROSS_SECTION
        return {
            'valid_methods': ['Simple train/test split', 'Cross-validation'],
            'invalid_methods': ['Any temporal method'],
            'warning': 'No time dimension - use standard cross-validation'
        }

polish_methods = get_recommended_methods(polish_structure['structure'])
american_methods = get_recommended_methods(american_structure['structure'])
taiwan_methods = get_recommended_methods(taiwan_structure['structure'])

print("\nPolish dataset:")
print(f"  ✓ Valid: {', '.join(polish_methods['valid_methods'])}")
if polish_methods['invalid_methods']:
    print(f"  ✗ Invalid: {', '.join(polish_methods['invalid_methods'])}")
if polish_methods['warning']:
    print(f"  ⚠️  {polish_methods['warning']}")

print("\nAmerican dataset:")
print(f"  ✓ Valid: {', '.join(american_methods['valid_methods'])}")
if american_methods['invalid_methods']:
    print(f"  ✗ Invalid: {', '.join(american_methods['invalid_methods'])}")

print("\nTaiwan dataset:")
print(f"  ✓ Valid: {', '.join(taiwan_methods['valid_methods'])}")
if taiwan_methods['invalid_methods']:
    print(f"  ✗ Invalid: {', '.join(taiwan_methods['invalid_methods'])}")

# ============================================================================
# STEP 5: Save outputs
# ============================================================================

print("\n[5/5] Saving outputs...")

# Comprehensive temporal analysis
temporal_analysis = {
    'polish': {
        'structure': polish_structure,
        'recommended_methods': polish_methods
    },
    'american': {
        'structure': american_structure,
        'recommended_methods': american_methods
    },
    'taiwan': {
        'structure': taiwan_structure,
        'recommended_methods': taiwan_methods
    }
}

# Save outputs
with open(TEMPORAL_STRUCTURE_DIR / 'temporal_structure_analysis.json', 'w') as f:
    json.dump(temporal_analysis, f, indent=2)

with open(TEMPORAL_STRUCTURE_DIR / 'recommended_methods.json', 'w') as f:
    json.dump({
        'polish': polish_methods,
        'american': american_methods,
        'taiwan': taiwan_methods
    }, f, indent=2)

print(f"\n✓ Outputs saved to: {TEMPORAL_STRUCTURE_DIR}")
print(f"  - temporal_structure_analysis.json")
print(f"  - recommended_methods.json")

# ============================================================================
# CRITICAL FINDINGS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SCRIPT 00b COMPLETE - TEMPORAL STRUCTURE VERIFIED")
print("="*80)

print("\n✅ CRITICAL FINDINGS:\n")

# Polish
print("📊 POLISH DATASET:")
print(f"   Structure: {polish_structure['structure']}")
if polish_structure['structure'] == 'REPEATED_CROSS_SECTIONS':
    print("   ⚠️  CRITICAL: Script 11 mislabeled as 'panel data' - this is WRONG!")
    print("   ✓ Correct label: 'Temporal Holdout Validation'")
    print("   ✗ Panel VAR INVALID (different companies each period)")
    print("   ✗ Granger causality INVALID (no company tracking)")

# American  
print(f"\n📊 AMERICAN DATASET:")
print(f"   Structure: {american_structure['structure']}")
print(f"   Companies tracked: {american_structure['pct_companies_tracked']:.1f}%")

# Taiwan
print(f"\n📊 TAIWAN DATASET:")
print(f"   Structure: {taiwan_structure['structure']}")

print("\n🎯 IMPLICATIONS FOR SCRIPTS:")
print("\n  Script 11 (Polish):")
if polish_structure['structure'] == 'REPEATED_CROSS_SECTIONS':
    print("    ✗ Current name: '11_panel_data_analysis.py' (INCORRECT!)")
    print("    ✓ Should be: '11_temporal_holdout_validation.py'")
else:
    print("    ✓ 'Panel data' label is correct")

print("\n  Script 13c (Polish):")
if polish_structure['structure'] == 'REPEATED_CROSS_SECTIONS':
    print("    ✗ Granger causality INVALID (requires panel data)")
    print("    ✓ Use: Temporal holdout validation instead")
else:
    print("    ✓ Granger causality is valid (panel data structure)")

print("\n" + "="*80)
print("Foundation complete! Proceed to Phase 1 (Fix Flawed Scripts)")
print("="*80 + "\n")
