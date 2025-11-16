#!/bin/bash
# Activate environment once and run all analyses

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

echo "✓ Virtual environment activated"
echo "✓ Python version: $(python --version)"
echo ""

# Run Polish dataset analysis
echo "========================================"
echo "Running Polish Dataset Analysis"
echo "========================================"
python scripts_python/01_data_understanding.py
python scripts_python/02_exploratory_analysis.py
python scripts_python/03_data_preparation.py
python scripts_python/04_baseline_models.py
python scripts_python/05_advanced_models.py
python scripts_python/06_model_calibration.py
python scripts_python/07_robustness_analysis.py

# Generate report
echo ""
echo "========================================"
echo "Generating HTML Report"
echo "========================================"
python scripts_python/generate_html_report.py

echo ""
echo "✓✓✓ ANALYSIS COMPLETE ✓✓✓"
echo "Open: results/ANALYSIS_REPORT.html"
