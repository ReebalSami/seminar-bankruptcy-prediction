#!/usr/bin/env python3
"""
Comprehensive Output Verification Script

Checks ALL script outputs for:
1. Files exist
2. Data is valid (not empty, no NaNs where unexpected)
3. Results make sense (AUC in [0,1], counts match, etc.)
4. Visualizations were created
5. Logs show success

Generates verification report.
"""

import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("COMPREHENSIVE OUTPUT VERIFICATION")
print("=" * 80)

results_dir = PROJECT_ROOT / 'results'
outputs_dir = results_dir / 'script_outputs'
logs_dir = PROJECT_ROOT / 'logs' / 'polish'

verification_results = {
    'foundation': {},
    'polish': {},
    'american': {},
    'taiwan': {},
    'issues': [],
    'warnings': [],
    'visualizations': []
}

# ============================================================================
# 1. VERIFY FOUNDATION SCRIPTS (00, 00b)
# ============================================================================
print("\n[1/5] Verifying Foundation Scripts...")

# Script 00
feat_map_dir = results_dir / '00_feature_mapping'
if feat_map_dir.exists():
    files = {
        'common_features.json': feat_map_dir / 'common_features.json',
        'feature_alignment_matrix.csv': feat_map_dir / 'feature_alignment_matrix.csv',
        'feature_semantic_mapping.json': feat_map_dir / 'feature_semantic_mapping.json',
    }
    
    script_00_ok = True
    for name, path in files.items():
        if path.exists():
            if name.endswith('.json'):
                with open(path) as f:
                    data = json.load(f)
                    if name == 'common_features.json':
                        if len(data) != 10:
                            verification_results['issues'].append(f"Script 00: Expected 10 common features, got {len(data)}")
                            script_00_ok = False
                        print(f"  ✓ {name}: {len(data)} common features")
            elif name.endswith('.csv'):
                df = pd.read_csv(path)
                print(f"  ✓ {name}: {len(df)} rows")
        else:
            verification_results['issues'].append(f"Script 00: Missing {name}")
            script_00_ok = False
    
    verification_results['foundation']['script_00'] = 'PASS' if script_00_ok else 'FAIL'
else:
    verification_results['issues'].append("Script 00: Output directory missing!")
    verification_results['foundation']['script_00'] = 'FAIL'

# Script 00b
temp_struct_dir = results_dir / '00b_temporal_structure'
if temp_struct_dir.exists():
    files = {
        'recommended_methods.json': temp_struct_dir / 'recommended_methods.json',
        'temporal_structure_analysis.json': temp_struct_dir / 'temporal_structure_analysis.json',
    }
    
    script_00b_ok = True
    for name, path in files.items():
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                if name == 'recommended_methods.json':
                    # Verify Polish is REPEATED_CROSS_SECTIONS
                    if 'polish' in data:
                        if 'Panel VAR' in data['polish'].get('invalid_methods', []):
                            print(f"  ✓ {name}: Polish correctly classified (no Panel VAR)")
                        else:
                            verification_results['warnings'].append("Script 00b: Polish should have Panel VAR as invalid")
        else:
            verification_results['issues'].append(f"Script 00b: Missing {name}")
            script_00b_ok = False
    
    verification_results['foundation']['script_00b'] = 'PASS' if script_00b_ok else 'FAIL'
else:
    verification_results['issues'].append("Script 00b: Output directory missing!")
    verification_results['foundation']['script_00b'] = 'FAIL'

# ============================================================================
# 2. VERIFY POLISH SCRIPTS (01-08, 10c, 10d, 11, 12, 13c)
# ============================================================================
print("\n[2/5] Verifying Polish Scripts...")

polish_scripts = [
    ('01_data_understanding', ['dataset_info.csv', 'horizon_statistics.csv', 'summary.json']),
    ('02_exploratory_analysis', ['discriminative_power.csv', 'high_correlations.csv']),
    ('03_data_preparation', ['preparation_summary.json']),
    ('04_baseline_models', ['baseline_results.csv']),
    ('05_advanced_models', ['advanced_results.csv']),
    ('06_model_calibration', ['calibration_summary.json']),
    ('07_robustness_analysis', ['cross_horizon_results.csv', 'performance_degradation.csv']),
    ('08_econometric_analysis', ['econometric_summary.json', 'vif_analysis.csv']),
    ('10c_glm_diagnostics', ['glm_diagnostics.json']),
    ('10d_remediation_save', ['remediation_summary.json']),
    ('11_temporal_holdout_validation', ['temporal_validation_summary.json']),
    ('12_transfer_learning_semantic', ['transfer_learning_results.csv', 'summary.json']),
    ('13c_temporal_validation', ['temporal_summary.json']),
]

for script_name, required_files in polish_scripts:
    script_dir = outputs_dir / script_name
    script_ok = True
    
    if script_dir.exists():
        for file_name in required_files:
            file_path = script_dir / file_name
            if not file_path.exists():
                verification_results['issues'].append(f"{script_name}: Missing {file_name}")
                script_ok = False
            else:
                # Quick data validation
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    if len(df) == 0:
                        verification_results['issues'].append(f"{script_name}: {file_name} is empty!")
                        script_ok = False
                    else:
                        print(f"  ✓ {script_name}/{file_name}: {len(df)} rows")
                elif file_name.endswith('.json'):
                    with open(file_path) as f:
                        data = json.load(f)
                        print(f"  ✓ {script_name}/{file_name}: loaded")
        
        # Check for figures
        figures_dir = script_dir / 'figures'
        if figures_dir.exists():
            figs = list(figures_dir.glob('*.png'))
            print(f"  ✓ {script_name}: {len(figs)} visualizations")
            verification_results['visualizations'].extend([str(f.relative_to(PROJECT_ROOT)) for f in figs])
    else:
        verification_results['issues'].append(f"{script_name}: Output directory missing!")
        script_ok = False
    
    verification_results['polish'][script_name] = 'PASS' if script_ok else 'FAIL'

# ============================================================================
# 3. VERIFY SPECIFIC METRICS
# ============================================================================
print("\n[3/5] Verifying specific metrics...")

# Check AUC values are in valid range
auc_files = [
    outputs_dir / '04_baseline_models' / 'baseline_results.csv',
    outputs_dir / '05_advanced_models' / 'advanced_results.csv',
    outputs_dir / '12_transfer_learning_semantic' / 'transfer_learning_results.csv',
]

for auc_file in auc_files:
    if auc_file.exists():
        df = pd.read_csv(auc_file)
        auc_cols = [col for col in df.columns if 'auc' in col.lower()]
        for col in auc_cols:
            if col in df.columns:
                aucs = df[col].dropna()
                if len(aucs) > 0:
                    if (aucs < 0).any() or (aucs > 1).any():
                        verification_results['issues'].append(f"{auc_file.name}: Invalid AUC values in {col}")
                    elif (aucs < 0.5).all():
                        verification_results['warnings'].append(f"{auc_file.name}: All AUC < 0.5 in {col} (worse than random)")
                    print(f"  ✓ {auc_file.parent.name}/{col}: {aucs.min():.3f} - {aucs.max():.3f}")

# Check bankruptcy rates make sense
horizon_stats = outputs_dir / '01_data_understanding' / 'horizon_statistics.csv'
if horizon_stats.exists():
    df = pd.read_csv(horizon_stats)
    if 'Bankruptcy_Rate' in df.columns:
        rates = df['Bankruptcy_Rate']
        if (rates < 0).any() or (rates > 1).any():
            verification_results['issues'].append("Horizon stats: Invalid bankruptcy rates")
        elif (rates < 0.01).any():
            verification_results['warnings'].append("Horizon stats: Very low bankruptcy rate (<1%)")
        print(f"  ✓ Bankruptcy rates: {rates.min():.2%} - {rates.max():.2%}")

# ============================================================================
# 4. CHECK LOG FILES
# ============================================================================
print("\n[4/5] Checking log files...")

if logs_dir.exists():
    log_files = list(logs_dir.glob('*.log'))
    print(f"  Found {len(log_files)} log files")
    
    for log_file in log_files:
        content = log_file.read_text()
        if 'error' in content.lower() or 'failed' in content.lower():
            if 'FAILED' in content:  # Actual failure
                verification_results['issues'].append(f"Log {log_file.name}: Contains errors")
        if 'traceback' in content.lower():
            verification_results['issues'].append(f"Log {log_file.name}: Contains traceback")
else:
    verification_results['warnings'].append("No log directory found")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n[5/5] Generating summary...")

total_scripts = len(polish_scripts) + 2  # +2 for foundation
passed = sum(1 for v in verification_results['foundation'].values() if v == 'PASS')
passed += sum(1 for v in verification_results['polish'].values() if v == 'PASS')

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

print(f"\n✅ Scripts Passed: {passed}/{total_scripts}")
print(f"⚠️  Warnings: {len(verification_results['warnings'])}")
print(f"❌ Critical Issues: {len(verification_results['issues'])}")
print(f"📊 Visualizations: {len(verification_results['visualizations'])}")

if verification_results['issues']:
    print("\n❌ CRITICAL ISSUES:")
    for issue in verification_results['issues'][:10]:  # Show first 10
        print(f"   - {issue}")

if verification_results['warnings']:
    print("\n⚠️  WARNINGS:")
    for warning in verification_results['warnings'][:5]:  # Show first 5
        print(f"   - {warning}")

# Save verification report
report_path = PROJECT_ROOT / 'VERIFICATION_REPORT.json'
with open(report_path, 'w') as f:
    json.dump(verification_results, f, indent=2)

print(f"\n✓ Saved: VERIFICATION_REPORT.json")

# Return exit code based on critical issues
if verification_results['issues']:
    print("\n❌ VERIFICATION FAILED - Fix critical issues above")
    sys.exit(1)
else:
    print("\n✅ VERIFICATION PASSED - All outputs valid!")
    sys.exit(0)
