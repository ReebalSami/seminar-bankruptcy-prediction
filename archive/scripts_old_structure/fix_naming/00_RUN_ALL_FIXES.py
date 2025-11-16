"""
MASTER SCRIPT: Fix all A1-A64 vs Attr1-Attr64 naming inconsistencies.

This will:
1. Update feature_descriptions.json: Attr1-Attr64 → A1-A64
2. Recreate processed parquet files with A1-A64 naming
3. Update Excel creation script to use A1-A64
4. Recreate the Excel file with correct naming
5. Update all documentation

Run this script to fix everything at once!
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and report results."""
    print("="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print()
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent.parent.parent,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"❌ FAILED with exit code {result.returncode}")
        return False
    else:
        print(f"✅ SUCCESS")
        print()
        return True


def main():
    """Run all naming fix scripts in sequence."""
    
    base_dir = Path(__file__).resolve().parents[2]
    fix_dir = base_dir / "scripts/fix_naming"
    
    print("\n" + "="*80)
    print("MASTER SCRIPT: FIXING ALL A1-A64 NAMING ISSUES")
    print("="*80)
    print()
    print("This will:")
    print("  1. Update feature_descriptions.json (Attr→A)")
    print("  2. Recreate processed data with A1-A64 naming")
    print("  3. Fix Excel creation script")
    print("  4. Recreate Excel file with correct naming")
    print()
    input("Press ENTER to continue or Ctrl+C to cancel...")
    print()
    
    scripts = [
        (fix_dir / "02_fix_feature_descriptions_json.py", 
         "Step 1: Update feature_descriptions.json"),
        
        (fix_dir / "03_recreate_processed_data_with_A_naming.py",
         "Step 2: Recreate processed parquet files"),
        
        (fix_dir / "04_fix_excel_creation_script.py",
         "Step 3: Fix Excel creation script"),
        
        (base_dir / "scripts/analysis/create_correct_feature_mapping_excel.py",
         "Step 4: Recreate Excel file"),
    ]
    
    all_success = True
    for script_path, description in scripts:
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            all_success = False
            continue
        
        success = run_script(script_path, description)
        if not success:
            all_success = False
            print(f"\n⚠️  ERROR in {description}")
            print("Do you want to continue anyway? (y/n): ", end="")
            response = input().strip().lower()
            if response != 'y':
                print("\nAborting...")
                sys.exit(1)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if all_success:
        print("✅ ALL FIXES COMPLETED SUCCESSFULLY!")
        print()
        print("Changes made:")
        print("  ✓ feature_descriptions.json now uses A1-A64")
        print("  ✓ Processed parquet files recreated with A1-A64")
        print("  ✓ Excel file regenerated with A1-A64")
        print()
        print("Next steps:")
        print("  • Verify the Excel file: docs/FEATURE_MAPPING_WITH_ORIGINAL_LABELS.xlsx")
        print("  • Check processed data: data/processed/poland_clean_full.parquet")
        print("  • Review feature_descriptions.json")
    else:
        print("⚠️  SOME STEPS FAILED")
        print("Please review the errors above and fix manually if needed.")
    
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
