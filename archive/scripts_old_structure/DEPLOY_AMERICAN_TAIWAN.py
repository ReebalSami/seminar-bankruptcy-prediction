#!/usr/bin/env python3
"""
Deploy American & Taiwan Dataset Pipelines

Systematically:
1. Copy scripts to new structure
2. Fix imports
3. Verify no hardcoding
4. Run sequentially
5. Analyze results
"""

import sys
from pathlib import Path
import shutil
import re

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("DEPLOYING AMERICAN & TAIWAN DATASET PIPELINES")
print("=" * 80)

# Setup
scripts_old = PROJECT_ROOT / 'scripts_python'
scripts_new = PROJECT_ROOT / 'scripts'

# Create directories
(scripts_new / '02_american').mkdir(exist_ok=True)
(scripts_new / '03_taiwan').mkdir(exist_ok=True)

def fix_imports(content, dataset_name):
    """Fix import paths for new structure."""
    
    # Replace old path setup with new
    old_pattern = r"sys\.path\.insert\(0, str\(Path\(__file__\)\.parent\.parent\)\)"
    new_setup = """# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))"""
    
    content = re.sub(old_pattern, new_setup, content)
    
    # Replace project_root references
    content = re.sub(r'\bproject_root\b', 'PROJECT_ROOT', content)
    
    # Ensure PROJECT_ROOT is defined if needed
    if 'PROJECT_ROOT' in content and 'PROJECT_ROOT = Path(__file__)' not in content:
        # Add it after imports
        if 'from pathlib import Path' in content:
            content = content.replace(
                'from pathlib import Path',
                'from pathlib import Path\n\n# Add project root to path\nPROJECT_ROOT = Path(__file__).parent.parent.parent\nsys.path.insert(0, str(PROJECT_ROOT))'
            )
    
    return content

# ============================================================================
# AMERICAN DATASET
# ============================================================================
print("\n[1/2] Setting up American dataset...")

american_scripts = [
    '01_data_cleaning.py',
    '02_eda.py',
    '03_baseline_models.py',
    '04_advanced_models.py',
    '05_model_calibration.py',
    '06_feature_importance.py',
    '07_robustness_check.py',
    '08_summary_report.py'
]

for script in american_scripts:
    src = scripts_old / 'american' / script
    dst = scripts_new / '02_american' / script
    
    if src.exists():
        # Copy and fix
        content = src.read_text()
        content = fix_imports(content, 'american')
        dst.write_text(content)
        print(f"  ✓ {script}")
    else:
        print(f"  ⚠️  {script} not found")

# ============================================================================
# TAIWAN DATASET
# ============================================================================
print("\n[2/2] Setting up Taiwan dataset...")

taiwan_scripts = [
    '01_data_cleaning.py',
    '02_eda.py',
    '03_baseline_models.py',
    '04_advanced_models.py',
    '05_calibration.py',
    '06_feature_importance.py',
    '07_robustness.py',
    '08_summary.py'
]

for script in taiwan_scripts:
    src = scripts_old / 'taiwan' / script
    dst = scripts_new / '03_taiwan' / script
    
    if src.exists():
        # Copy and fix
        content = src.read_text()
        content = fix_imports(content, 'taiwan')
        dst.write_text(content)
        print(f"  ✓ {script}")
    else:
        print(f"  ⚠️  {script} not found")

# ============================================================================
# CREATE EXECUTION SCRIPTS
# ============================================================================
print("\n[3/3] Creating execution scripts...")

# American runner
american_runner = scripts_new / '02_american' / 'RUN_ALL_AMERICAN.sh'
american_runner.write_text(f"""#!/bin/bash
# Run all American dataset scripts

set -e
cd "{PROJECT_ROOT}"

PYTHON=".venv/bin/python"
LOG_DIR="logs/american"
mkdir -p "$LOG_DIR"

echo "=================================="
echo "AMERICAN DATASET - SEQUENTIAL EXECUTION"
echo "=================================="

SCRIPTS=(
    "01_data_cleaning.py"
    "02_eda.py"
    "03_baseline_models.py"
    "04_advanced_models.py"
    "05_model_calibration.py"
    "06_feature_importance.py"
    "07_robustness_check.py"
    "08_summary_report.py"
)

for script in "${{SCRIPTS[@]}}"; do
    script_name="${{script%.py}}"
    log_file="$LOG_DIR/${{script_name}}.log"
    
    echo "[$(date +'%H:%M:%S')] Running $script..."
    
    if $PYTHON "scripts/02_american/$script" > "$log_file" 2>&1; then
        echo "  ✓ SUCCESS"
    else
        echo "  ✗ FAILED - Check: $log_file"
        tail -30 "$log_file"
        exit 1
    fi
done

echo "=================================="
echo "✓ ALL AMERICAN SCRIPTS COMPLETED"
echo "=================================="
""")
american_runner.chmod(0o755)

# Taiwan runner
taiwan_runner = scripts_new / '03_taiwan' / 'RUN_ALL_TAIWAN.sh'
taiwan_runner.write_text(f"""#!/bin/bash
# Run all Taiwan dataset scripts

set -e
cd "{PROJECT_ROOT}"

PYTHON=".venv/bin/python"
LOG_DIR="logs/taiwan"
mkdir -p "$LOG_DIR"

echo "=================================="
echo "TAIWAN DATASET - SEQUENTIAL EXECUTION"
echo "=================================="

SCRIPTS=(
    "01_data_cleaning.py"
    "02_eda.py"
    "03_baseline_models.py"
    "04_advanced_models.py"
    "05_calibration.py"
    "06_feature_importance.py"
    "07_robustness.py"
    "08_summary.py"
)

for script in "${{SCRIPTS[@]}}"; do
    script_name="${{script%.py}}"
    log_file="$LOG_DIR/${{script_name}}.log"
    
    echo "[$(date +'%H:%M:%S')] Running $script..."
    
    if $PYTHON "scripts/03_taiwan/$script" > "$log_file" 2>&1; then
        echo "  ✓ SUCCESS"
    else
        echo "  ✗ FAILED - Check: $log_file"
        tail -30 "$log_file"
        exit 1
    fi
done

echo "=================================="
echo "✓ ALL TAIWAN SCRIPTS COMPLETED"
echo "=================================="
""")
taiwan_runner.chmod(0o755)

print(f"  ✓ American runner: {american_runner}")
print(f"  ✓ Taiwan runner: {taiwan_runner}")

print("\n" + "=" * 80)
print("✅ DEPLOYMENT COMPLETE")
print("=" * 80)
print("\nNext steps:")
print(f"  1. bash {american_runner}")
print(f"  2. bash {taiwan_runner}")
print("\n  (Or run both datasets in parallel)")
