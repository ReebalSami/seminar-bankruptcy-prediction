#!/bin/bash
# Run all Taiwan dataset scripts

set -e
cd "/Users/reebal/FH-Wedel/WS25/seminar-bankruptcy-prediction"

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

for script in "${SCRIPTS[@]}"; do
    script_name="${script%.py}"
    log_file="$LOG_DIR/${script_name}.log"
    
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
