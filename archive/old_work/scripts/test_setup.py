"""
Test Setup Script
=================

Quick test to verify all modules load correctly.
Run: python scripts/test_setup.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*70)
print("  TESTING MODULE SETUP")
print("="*70 + "\n")

# Test 1: Metadata module
print("[1/3] Testing metadata module...")
try:
    from src.metadata import (
        FEATURE_NAMES, SHORT_NAMES, CATEGORIES,
        get_readable_name, get_category, rename_dataframe
    )
    
    assert len(FEATURE_NAMES) == 64, "Should have 64 features"
    assert len(SHORT_NAMES) == 64, "Should have 64 short names"
    assert len(CATEGORIES) == 6, "Should have 6 categories"
    
    # Test conversion
    name = get_readable_name('Attr1', short=True)
    assert name == 'Net Profit / Assets', f"Got: {name}"
    
    category = get_category('Attr1')
    assert category == 'Profitability', f"Got: {category}"
    
    print("   ✅ Metadata module OK")
    print(f"      - {len(FEATURE_NAMES)} feature mappings loaded")
    print(f"      - {len(CATEGORIES)} categories defined")
    print(f"      - Example: Attr1 → {name}")
    
except Exception as e:
    print(f"   ❌ Metadata module FAILED: {e}")
    sys.exit(1)

# Test 2: Visualization module
print("\n[2/3] Testing visualization module...")
try:
    from src.visualization import (
        plot_class_imbalance_by_horizon,
        plot_missingness,
        plot_distributions_by_class,
        plot_correlation_heatmap,
        find_high_correlations,
        plot_roc_pr_curves,
        plot_calibration_curve,
        plot_feature_importance,
        plot_odds_ratios
    )
    
    print("   ✅ Visualization module OK")
    print("      - 9 plotting functions available")
    
except Exception as e:
    print(f"   ❌ Visualization module FAILED: {e}")
    sys.exit(1)

# Test 3: Cross-horizon module
print("\n[3/3] Testing cross-horizon module...")
try:
    from src.cross_horizon import (
        prepare_horizon_data,
        cross_horizon_evaluation,
        full_cross_horizon_matrix,
        plot_cross_horizon_heatmap,
        summarize_degradation
    )
    
    print("   ✅ Cross-horizon module OK")
    print("      - 5 evaluation functions available")
    
except Exception as e:
    print(f"   ❌ Cross-horizon module FAILED: {e}")
    sys.exit(1)

# Test 4: Check data availability
print("\n[4/4] Checking data files...")
repo_root = Path(__file__).parent.parent
data_dir = repo_root / "data" / "processed"

required_files = [
    "poland_clean_full.parquet",
    "poland_clean_reduced.parquet",
    "poland_h1_test_predictions.csv",
]

missing = []
for file in required_files:
    if (data_dir / file).exists():
        size = (data_dir / file).stat().st_size / (1024*1024)  # MB
        print(f"   ✅ {file} ({size:.1f} MB)")
    else:
        print(f"   ❌ {file} - NOT FOUND")
        missing.append(file)

if missing:
    print(f"\n   ⚠️  Missing {len(missing)} file(s)")
    print("   💡 These files are created by your notebooks")
else:
    print("\n   ✅ All required data files present")

# Summary
print("\n" + "="*70)
print("  ✅ SETUP TEST COMPLETE!")
print("="*70)
print("\n📋 Next Steps:")
print("   1. Run: python scripts/demo_visuals.py")
print("   2. Or: python scripts/generate_all_visuals.py")
print("   3. Or: python scripts/run_cross_horizon.py")
print("\n   ⚠️  Note: Scripts require pandas, matplotlib, etc.")
print("   Run from virtual environment: source .venv/bin/activate\n")

