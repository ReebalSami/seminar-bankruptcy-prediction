#!/usr/bin/env python3
"""
FINAL METHODOLOGY CHECK - Code Execution Only

Checks ACTUAL function calls, not comments/docstrings.
Verifies what code RUNS, not what's documented.
"""

import sys
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("FINAL METHODOLOGY CHECK - ACTUAL CODE EXECUTION")
print("=" * 80)

scripts_dir = PROJECT_ROOT / 'scripts' / '01_polish'

def get_actual_code(script_path):
    """Get only executable code, excluding comments and docstrings."""
    with open(script_path) as f:
        lines = f.readlines()
    
    code_lines = []
    in_docstring = False
    docstring_char = None
    
    for line in lines:
        stripped = line.strip()
        
        # Handle docstrings
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if not in_docstring:
                in_docstring = True
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2:  # Single-line docstring
                    in_docstring = False
                continue
            elif stripped.endswith(docstring_char):
                in_docstring = False
                continue
        
        if in_docstring:
            continue
        
        # Skip comments
        if stripped.startswith('#'):
            continue
        
        # Remove inline comments
        if '#' in line:
            line = line[:line.index('#')]
        
        if line.strip():
            code_lines.append(line)
    
    return '\n'.join(code_lines)

critical_issues = []
warnings = []
passes = []

# ============================================================================
# CHECK EACH SCRIPT
# ============================================================================

scripts_to_check = {
    '10c_glm_diagnostics.py': {
        'invalid_functions': ['durbin_watson', 'jarque_bera', 'statsmodels.stats.stattools.durbin_watson'],
        'required_functions': ['hosmer_lemeshow'],
        'description': 'GLM Diagnostics'
    },
    '11_temporal_holdout_validation.py': {
        'invalid_functions': ['grangercausalitytests', 'granger'],
        'required_functions': [],
        'description': 'Temporal Holdout'
    },
    '13c_temporal_validation.py': {
        'invalid_functions': ['grangercausalitytests', 'granger'],
        'required_functions': [],
        'description': 'Temporal Validation'
    },
    '12_cross_dataset_transfer_SEMANTIC.py': {
        'invalid_functions': ['get_top_features'],  # Should use semantic, not top-N
        'required_functions': ['common_features.json'],
        'description': 'Transfer Learning'
    }
}

print("\n🔍 Checking ACTUAL code execution (not comments)...\n")

for script_name, checks in scripts_to_check.items():
    script_path = scripts_dir / script_name
    print(f"[{checks['description']}] {script_name}")
    
    if not script_path.exists():
        print(f"  ❌ Script not found\n")
        critical_issues.append(f"{script_name}: Missing")
        continue
    
    # Get actual executable code
    code = get_actual_code(script_path)
    
    # Check for invalid function CALLS
    issues_found = []
    for invalid in checks['invalid_functions']:
        # Look for actual function calls (not just mentions)
        patterns = [
            rf'{invalid}\s*\(',  # Function call
            rf'from .* import .*{invalid}',  # Import
        ]
        
        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues_found.append(f"Calls {invalid}()")
    
    if issues_found:
        print(f"  ❌ CRITICAL: {', '.join(issues_found)}")
        critical_issues.extend([f"{script_name}: {i}" for i in issues_found])
    else:
        print(f"  ✅ No invalid function calls")
    
    # Check for required elements
    missing_required = []
    for required in checks['required_functions']:
        if required not in code:
            missing_required.append(required)
    
    if missing_required:
        if script_name == '12_cross_dataset_transfer_SEMANTIC.py':
            print(f"  ❌ CRITICAL: Missing {missing_required}")
            critical_issues.append(f"{script_name}: Missing {missing_required}")
        else:
            print(f"  ⚠️  Missing: {missing_required}")
            warnings.append(f"{script_name}: Missing {missing_required}")
    else:
        if checks['required_functions']:
            print(f"  ✅ Has required: {checks['required_functions']}")
        passes.append(script_name)
    
    print()

# ============================================================================
# DATA LOADING CHECK
# ============================================================================
print("=" * 80)
print("DATA LOADING VERIFICATION")
print("=" * 80)
print("\nChecking correct data sources...")

data_flow = {
    '01-08': 'DataLoader (raw data)',
    '10c': 'data/processed/poland_clean_full.parquet',
    '10d': 'data/processed/poland_clean_full.parquet',
    '11': 'data/processed/poland_h1_vif_selected.parquet (from 10d)',
    '12': 'results/00_feature_mapping/common_features.json + processed data',
    '13c': 'data/processed/poland_h1_vif_selected.parquet (from 10d)',
}

print("\nExpected data flow:")
for script, source in data_flow.items():
    print(f"  {script}: {source}")

# Verify Script 12 loads Script 00
script_12 = scripts_dir / '12_cross_dataset_transfer_SEMANTIC.py'
code_12 = get_actual_code(script_12)

if "json.load" in code_12 and "common_features.json" in code_12:
    print("\n✅ Script 12: Loads Script 00 outputs (common_features.json)")
    passes.append("Script 12 → Script 00 dependency verified")
else:
    print("\n❌ Script 12: Doesn't load Script 00 outputs!")
    critical_issues.append("Script 12: No Script 00 dependency")

# Verify Script 11 loads Script 10d
script_11 = scripts_dir / '11_temporal_holdout_validation.py'
code_11 = get_actual_code(script_11)

if "poland_h1_vif_selected.parquet" in code_11:
    print("✅ Script 11: Loads Script 10d output (VIF-selected features)")
else:
    print("⚠️  Script 11: Doesn't load 10d output")
    warnings.append("Script 11: May not use VIF-selected features")

# ============================================================================
# HARDCODING CHECK
# ============================================================================
print("\n" + "=" * 80)
print("HARDCODING CHECK")
print("=" * 80)

print("\nChecking for hardcoded values that should be configurable...")

hardcode_patterns = {
    r'n_estimators\s*=\s*10\b': 'n_estimators=10 (too small)',
    r'max_iter\s*=\s*10\b': 'max_iter=10 (may not converge)',
    r'test_size\s*=\s*0\.\d+': None,  # This is OK
    r'random_state\s*=\s*\d+': None,  # This is OK
}

for script_file in scripts_dir.glob('[0-9]*.py'):
    code = get_actual_code(script_file)
    script_warnings = []
    
    for pattern, msg in hardcode_patterns.items():
        if msg and re.search(pattern, code):
            script_warnings.append(msg)
    
    if script_warnings:
        warnings.extend([f"{script_file.name}: {w}" for w in script_warnings])

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\n✅ Passed checks: {len(passes)}")
print(f"⚠️  Warnings: {len(warnings)}")
print(f"❌ Critical Issues: {len(critical_issues)}")

if critical_issues:
    print("\n❌ CRITICAL ISSUES (MUST FIX):")
    for issue in critical_issues:
        print(f"  - {issue}")

if warnings and len(warnings) <= 10:
    print("\n⚠️  WARNINGS (Acceptable but review):")
    for warning in warnings:
        print(f"  - {warning}")

# Final verdict
print("\n" + "=" * 80)
if critical_issues:
    print("❌ METHODOLOGY CHECK FAILED")
    print("=" * 80)
    print("\nFix critical issues before proceeding to American & Taiwan datasets.")
    sys.exit(1)
else:
    print("✅ METHODOLOGY CHECK PASSED")
    print("=" * 80)
    print("\nAll scripts use correct methods:")
    print("  • No invalid OLS tests on logistic regression")
    print("  • No Granger causality on repeated cross-sections")
    print("  • Transfer learning uses semantic mapping")
    print("  • Proper data dependencies")
    print("\n🎯 Ready to proceed with American & Taiwan datasets!")
    sys.exit(0)
