#!/bin/bash
# Master execution script for Polish dataset analysis
# Runs all scripts in correct sequence and logs results

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=".venv/bin/python"
LOG_DIR="logs/polish"
mkdir -p "$LOG_DIR"

echo "=================================="
echo "POLISH DATASET - SEQUENTIAL EXECUTION"
echo "=================================="
echo ""

# Array of scripts in correct order
SCRIPTS=(
    "01_data_understanding.py"
    "02_exploratory_analysis.py"
    "03_data_preparation.py"
    "04_baseline_models.py"
    "05_advanced_models.py"
    "06_model_calibration.py"
    "07_robustness_analysis.py"
    "08_econometric_analysis.py"
)

# Run each script
for script in "${SCRIPTS[@]}"; do
    script_name="${script%.py}"
    log_file="$LOG_DIR/${script_name}.log"
    
    echo "[$(date +'%H:%M:%S')] Running $script..."
    
    if $PYTHON "scripts/01_polish/$script" > "$log_file" 2>&1; then
        echo "  ✓ SUCCESS - Log: $log_file"
    else
        echo "  ✗ FAILED - Check log: $log_file"
        echo ""
        echo "Last 20 lines of error log:"
        tail -20 "$log_file"
        exit 1
    fi
    echo ""
done

echo "=================================="
echo "✓ ALL POLISH SCRIPTS COMPLETED"
echo "=================================="
echo ""
echo "Outputs saved to: results/script_outputs/"
echo "Logs saved to: $LOG_DIR/"
