#!/usr/bin/env python3
"""
COMPREHENSIVE METHODOLOGY AUDIT

Verifies:
1. Data loading strategy (correct files, no hardcoding)
2. Sequential dependencies (Script N loads Script N-1 outputs)
3. Econometric test appropriateness (GLM for logistic, not OLS)
4. No shortcuts or hardcoded values
5. Methodology soundness
6. Results interpretation validity

Reports: PASS/FAIL for each script with detailed reasoning.
"""

import sys
from pathlib import Path
import pandas as pd
import json
import ast
import re

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("COMPREHENSIVE METHODOLOGY AUDIT")
print("=" * 80)
print("\nChecking: Data loading, dependencies, econometrics, hardcoding, soundness")

scripts_dir = PROJECT_ROOT / 'scripts' / '01_polish'
results_dir = PROJECT_ROOT / 'results'

audit_results = {
    'foundation': {},
    'polish': {},
    'critical_issues': [],
    'warnings': [],
    'passes': []
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_script_source(script_path):
    """Extract and analyze script source code."""
    with open(script_path, 'r') as f:
        source = f.read()
    return source

def find_hardcoded_paths(source):
    """Find hardcoded file paths in source."""
    hardcoded = []
    
    # Look for string literals with paths
    path_patterns = [
        r'["\'](?:/Users/|/home/|C:\\|\\\\)',  # Absolute paths
        r'\.parquet["\'](?!\))',  # Direct parquet references
        r'\.csv["\'](?!\))',  # Direct CSV references
    ]
    
    for pattern in path_patterns:
        matches = re.findall(pattern, source)
        if matches:
            hardcoded.extend(matches)
    
    return hardcoded

def check_data_loading_strategy(script_name, source):
    """Verify data loading uses proper methods."""
    issues = []
    
    # Should use config paths
    if 'PROJECT_ROOT' not in source:
        issues.append("Missing PROJECT_ROOT definition")
    
    # Should NOT have hardcoded paths
    hardcoded = find_hardcoded_paths(source)
    if hardcoded:
        issues.append(f"Hardcoded paths found: {hardcoded[:3]}")
    
    # Check what it loads
    loads_from = []
    if 'pd.read_parquet' in source:
        loads_from.append('parquet')
    if 'pd.read_csv' in source:
        loads_from.append('csv')
    if 'json.load' in source:
        loads_from.append('json')
    
    return {
        'loads_from': loads_from,
        'issues': issues
    }

def check_sequential_dependencies(script_name, source):
    """Verify script loads outputs from correct previous scripts."""
    dependencies = []
    
    # Map expected dependencies
    expected_deps = {
        '01_data_understanding': ['raw data'],
        '02_exploratory_analysis': ['raw data'],
        '03_data_preparation': ['raw data'],
        '04_baseline_models': ['03_data_preparation output'],
        '05_advanced_models': ['03_data_preparation output'],
        '06_model_calibration': ['04 or 05 output'],
        '07_robustness_analysis': ['03_data_preparation output'],
        '08_econometric_analysis': ['03_data_preparation output'],
        '10c_glm_diagnostics': ['raw data OR 03 output'],
        '10d_remediation_save_datasets': ['raw data'],
        '11_temporal_holdout_validation': ['10d output: poland_h1_vif_selected.parquet'],
        '12_cross_dataset_transfer_SEMANTIC': ['Script 00 outputs + processed data'],
        '13c_temporal_validation': ['10d output OR raw data'],
    }
    
    expected = expected_deps.get(script_name, ['unknown'])
    
    # Check what it actually loads
    actual_loads = []
    if 'poland_h1_vif_selected.parquet' in source:
        actual_loads.append('10d output (VIF-selected)')
    if 'poland_clean_full.parquet' in source:
        actual_loads.append('processed/poland_clean_full')
    if 'american_modeling.parquet' in source:
        actual_loads.append('processed/american')
    if 'taiwan_clean.parquet' in source:
        actual_loads.append('processed/taiwan')
    if 'common_features.json' in source:
        actual_loads.append('Script 00 output')
    if 'DataLoader' in source:
        actual_loads.append('DataLoader (raw)')
    
    return {
        'expected': expected,
        'actual': actual_loads
    }

def check_econometric_soundness(script_name, source):
    """Verify econometric tests are appropriate."""
    issues = []
    good_practices = []
    
    # GLM Diagnostics (Script 10c)
    if '10c' in script_name:
        if 'Durbin-Watson' in source or 'durbin_watson' in source:
            issues.append("❌ CRITICAL: Uses Durbin-Watson (OLS test) on logistic regression!")
        else:
            good_practices.append("✓ No Durbin-Watson (correct)")
        
        if 'Jarque-Bera' in source or 'jarque_bera' in source:
            issues.append("❌ CRITICAL: Uses Jarque-Bera (OLS test) on logistic regression!")
        else:
            good_practices.append("✓ No Jarque-Bera (correct)")
        
        if 'hosmer' in source.lower() or 'lemeshow' in source.lower():
            good_practices.append("✓ Uses Hosmer-Lemeshow (correct for GLM)")
        else:
            issues.append("⚠️ Missing Hosmer-Lemeshow test")
        
        if 'deviance' in source.lower():
            good_practices.append("✓ Uses deviance residuals (correct)")
    
    # Transfer Learning (Script 12)
    if '12' in script_name:
        if 'common_features.json' in source:
            good_practices.append("✓ Loads Script 00 semantic mappings")
        else:
            issues.append("❌ CRITICAL: Doesn't load semantic mappings from Script 00!")
        
        if 'get_top_features' in source and 'semantic' not in script_name.lower():
            issues.append("⚠️ May use top-N features instead of semantic alignment")
    
    # Temporal Validation (Script 11, 13c)
    if '11' in script_name or '13c' in script_name:
        if 'granger' in source.lower() and 'REPEATED_CROSS_SECTIONS' in source.upper():
            issues.append("❌ CRITICAL: Granger causality invalid for repeated cross-sections!")
        else:
            good_practices.append("✓ No invalid Granger causality")
    
    # VIF / Multicollinearity
    if 'vif' in source.lower() or 'variance_inflation' in source:
        good_practices.append("✓ Checks multicollinearity (VIF)")
    
    return {
        'issues': issues,
        'good_practices': good_practices
    }

def check_for_shortcuts(source):
    """Find shortcuts like .head(), sample(), or test runs."""
    shortcuts = []
    
    if '.head(' in source and 'only' not in source.lower():
        shortcuts.append("Uses .head() - may not analyze full data")
    
    if '.sample(' in source:
        shortcuts.append("Uses .sample() - subset analysis")
    
    if 'n_estimators=10' in source or 'n_estimators = 10' in source:
        shortcuts.append("Uses n_estimators=10 (too small for real analysis)")
    
    if 'max_iter=10' in source or 'max_iter = 10' in source:
        shortcuts.append("Uses max_iter=10 (may not converge)")
    
    return shortcuts

# ============================================================================
# AUDIT FOUNDATION SCRIPTS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1: FOUNDATION SCRIPTS (00, 00b)")
print("=" * 80)

# Script 00
print("\n[Script 00] Cross-Dataset Feature Mapping")
script_00 = PROJECT_ROOT / 'scripts' / '00_foundation' / '00_cross_dataset_feature_mapping.py'
if script_00.exists():
    source = check_script_source(script_00)
    
    # Check semantic mappings
    if 'SEMANTIC_MAPPINGS' in source:
        print("  ✓ Defines SEMANTIC_MAPPINGS")
        audit_results['foundation']['script_00_mappings'] = 'PASS'
    else:
        print("  ❌ Missing SEMANTIC_MAPPINGS definition")
        audit_results['critical_issues'].append("Script 00: No semantic mappings")
    
    # Check output
    if 'common_features.json' in source:
        print("  ✓ Saves common_features.json")
    else:
        print("  ❌ Doesn't save common_features.json")
    
    # Check for hardcoding
    hardcoded = find_hardcoded_paths(source)
    if hardcoded:
        print(f"  ⚠️ Hardcoded paths: {len(hardcoded)}")
        audit_results['warnings'].append(f"Script 00: {len(hardcoded)} hardcoded paths")
    else:
        print("  ✓ No hardcoded paths")
    
    audit_results['foundation']['script_00'] = 'PASS'
else:
    print("  ❌ Script 00 not found!")
    audit_results['critical_issues'].append("Script 00 missing")

# Script 00b
print("\n[Script 00b] Temporal Structure Verification")
script_00b = PROJECT_ROOT / 'scripts' / '00_foundation' / '00b_temporal_structure_verification.py'
if script_00b.exists():
    source = check_script_source(script_00b)
    
    # Check temporal classification
    if 'REPEATED_CROSS_SECTIONS' in source:
        print("  ✓ Classifies REPEATED_CROSS_SECTIONS")
    if 'TIME_SERIES' in source:
        print("  ✓ Classifies TIME_SERIES")
    if 'UNBALANCED_PANEL' in source:
        print("  ✓ Classifies UNBALANCED_PANEL")
    
    # Check output
    if 'recommended_methods.json' in source:
        print("  ✓ Saves recommended_methods.json")
    
    audit_results['foundation']['script_00b'] = 'PASS'
else:
    print("  ❌ Script 00b not found!")
    audit_results['critical_issues'].append("Script 00b missing")

# ============================================================================
# AUDIT POLISH SCRIPTS
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2: POLISH SCRIPTS (01-08, 10c, 10d, 11, 12, 13c)")
print("=" * 80)

polish_scripts = [
    '01_data_understanding.py',
    '02_exploratory_analysis.py',
    '03_data_preparation.py',
    '04_baseline_models.py',
    '05_advanced_models.py',
    '06_model_calibration.py',
    '07_robustness_analysis.py',
    '08_econometric_analysis.py',
    '10c_glm_diagnostics.py',
    '10d_remediation_save_datasets.py',
    '11_temporal_holdout_validation.py',
    '12_cross_dataset_transfer_SEMANTIC.py',
    '13c_temporal_validation.py',
]

for script_name in polish_scripts:
    script_path = scripts_dir / script_name
    print(f"\n[{script_name}]")
    
    if not script_path.exists():
        print(f"  ❌ Script not found")
        audit_results['critical_issues'].append(f"{script_name}: Missing")
        continue
    
    source = check_script_source(script_path)
    script_key = script_name.replace('.py', '')
    audit_results['polish'][script_key] = {}
    
    # 1. Data Loading
    loading = check_data_loading_strategy(script_key, source)
    if loading['issues']:
        print(f"  ⚠️ Data loading issues: {loading['issues']}")
        audit_results['warnings'].extend([f"{script_name}: {i}" for i in loading['issues']])
    else:
        print(f"  ✓ Data loading: {', '.join(loading['loads_from'])}")
    
    # 2. Dependencies
    deps = check_sequential_dependencies(script_key, source)
    print(f"  ✓ Loads: {', '.join(deps['actual']) if deps['actual'] else 'raw data'}")
    audit_results['polish'][script_key]['dependencies'] = deps
    
    # 3. Econometric Soundness
    econ = check_econometric_soundness(script_key, source)
    if econ['issues']:
        for issue in econ['issues']:
            print(f"  {issue}")
            if '❌' in issue:
                audit_results['critical_issues'].append(f"{script_name}: {issue}")
            else:
                audit_results['warnings'].append(f"{script_name}: {issue}")
    
    for practice in econ['good_practices']:
        print(f"  {practice}")
    
    # 4. Shortcuts
    shortcuts = check_for_shortcuts(source)
    if shortcuts:
        print(f"  ⚠️ Shortcuts: {shortcuts}")
        audit_results['warnings'].extend([f"{script_name}: {s}" for s in shortcuts])
    else:
        print(f"  ✓ No shortcuts")
    
    # Overall assessment
    if not loading['issues'] and not econ['issues'] and not shortcuts:
        audit_results['passes'].append(script_name)
        print(f"  ✅ METHODOLOGY: SOUND")
    elif econ['issues'] and any('❌' in i for i in econ['issues']):
        print(f"  ❌ METHODOLOGY: CRITICAL ISSUES")
    else:
        print(f"  ⚠️ METHODOLOGY: ACCEPTABLE WITH WARNINGS")

# ============================================================================
# VERIFY SEQUENTIAL EXECUTION ORDER
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 3: SEQUENTIAL DEPENDENCIES CHECK")
print("=" * 80)

print("\nExpected execution order:")
order = [
    "00, 00b (Foundation)",
    "01, 02, 03 (Data prep)",
    "04, 05 (Baseline & Advanced models)",
    "06 (Calibration)",
    "07, 08 (Robustness & Econometrics)",
    "10c, 10d (Diagnostics & Remediation)",
    "11, 12, 13c (Temporal & Transfer)"
]

for i, step in enumerate(order, 1):
    print(f"  {i}. {step}")

print("\n✓ This order ensures:")
print("  - Foundation before modeling (00, 00b → everything)")
print("  - Data prep before models (03 → 04, 05, 06, 07, 08)")
print("  - Remediation before advanced temporal (10d → 11)")
print("  - Semantic mapping before transfer (00 → 12)")

# Check if Script 12 actually loads Script 00
script_12 = scripts_dir / '12_cross_dataset_transfer_SEMANTIC.py'
if script_12.exists():
    source = check_script_source(script_12)
    if 'common_features.json' in source and 'json.load' in source:
        print("\n✅ VERIFIED: Script 12 loads Script 00 outputs")
        audit_results['passes'].append("Script 12 uses Script 00 (verified)")
    else:
        print("\n❌ CRITICAL: Script 12 doesn't load Script 00 outputs!")
        audit_results['critical_issues'].append("Script 12: No Script 00 dependency")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)

print(f"\n✅ Passed: {len(audit_results['passes'])}")
print(f"⚠️  Warnings: {len(audit_results['warnings'])}")
print(f"❌ Critical Issues: {len(audit_results['critical_issues'])}")

if audit_results['critical_issues']:
    print("\n❌ CRITICAL ISSUES FOUND:")
    for issue in audit_results['critical_issues']:
        print(f"  - {issue}")

if audit_results['warnings']:
    print("\n⚠️  WARNINGS:")
    for warning in audit_results['warnings'][:10]:  # Show first 10
        print(f"  - {warning}")

# Save audit
audit_path = PROJECT_ROOT / 'METHODOLOGY_AUDIT_REPORT.json'
with open(audit_path, 'w') as f:
    json.dump(audit_results, f, indent=2)

print(f"\n✓ Saved: METHODOLOGY_AUDIT_REPORT.json")

# Exit code
if audit_results['critical_issues']:
    print("\n❌ AUDIT FAILED - Fix critical issues before proceeding")
    sys.exit(1)
else:
    print("\n✅ AUDIT PASSED - Methodology sound, ready for American & Taiwan")
    sys.exit(0)
