"""
Path configuration for all datasets and outputs.

Centralizes all file paths to avoid hardcoding throughout the project.
"""

from pathlib import Path

# Project root (3 levels up from this file: scripts/config/paths.py -> root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
SCRIPT_OUTPUTS_DIR = RESULTS_DIR / "script_outputs"

# Dataset paths
POLISH_DATA_DIR = DATA_DIR / "polish-companies-bankruptcy"
AMERICAN_DATA_DIR = DATA_DIR / "NYSE-and-NASDAQ-companies"
TAIWAN_DATA_DIR = DATA_DIR / "taiwan-economic-journal"

# Specific dataset files
POLISH_DATA_PATH = POLISH_DATA_DIR / "data-from-kaggel.csv"
AMERICAN_DATA_PATH = AMERICAN_DATA_DIR / "american-bankruptcy.csv"
TAIWAN_DATA_PATH = TAIWAN_DATA_DIR / "taiwan-bankruptcy.csv"

# Output directories for Script 00
FEATURE_MAPPING_DIR = RESULTS_DIR / "00_feature_mapping"
TEMPORAL_STRUCTURE_DIR = RESULTS_DIR / "00b_temporal_structure"

# Ensure output directories exist
FEATURE_MAPPING_DIR.mkdir(parents=True, exist_ok=True)
TEMPORAL_STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
