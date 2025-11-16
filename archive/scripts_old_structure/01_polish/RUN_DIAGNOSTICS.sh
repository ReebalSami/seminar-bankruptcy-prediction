#!/bin/bash
# Run Polish diagnostic and temporal scripts

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=".venv/bin/python"
LOG_DIR="logs/polish"
mkdir -p "$LOG_DIR"

echo "=================================="
echo "POLISH DIAGNOSTICS & TEMPORAL"
echo "=================================="

SCRIPTS=(
    "10c_glm_diagnostics.py"
    "10d_remediation_save_datasets.py"
    "11_temporal_holdout_validation.py"
    "13c_temporal_validation.py"
)

for script in "${SCRIPTS[@]}"; do
    script_name="${script%.py}"
    log_file="$LOG_DIR/${script_name}.log"
    
    echo "[$(date +'%H:%M:%S')] Running $script..."
    
    if $PYTHON "scripts/01_polish/$script" > "$log_file" 2>&1; then
        echo "  ✓ SUCCESS"
    else
        echo "  ✗ FAILED - Check: $log_file"
        tail -30 "$log_file"
        exit 1
    fi
done

echo "=================================="
echo "✓ DIAGNOSTICS COMPLETE"
echo "=================================="
