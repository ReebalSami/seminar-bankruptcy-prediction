"""
Configuration module for bankruptcy prediction project.

Centralizes all paths, hyperparameters, and feature definitions to ensure
consistency across all scripts and datasets.
"""

from .paths import *
from .model_params import *
from .feature_groups import *

__all__ = [
    # Paths
    'PROJECT_ROOT', 'DATA_DIR', 'RESULTS_DIR', 'MODELS_DIR',
    'POLISH_DATA_PATH', 'AMERICAN_DATA_PATH', 'TAIWAN_DATA_PATH',
    'FEATURE_MAPPING_DIR', 'TEMPORAL_STRUCTURE_DIR',
    'SCRIPT_OUTPUTS_DIR', 'FIGURES_DIR',
    
    # Model parameters
    'RANDOM_STATE', 'RF_PARAMS', 'XGBOOST_PARAMS', 'LIGHTGBM_PARAMS',
    'CATBOOST_PARAMS', 'LOGISTIC_PARAMS',
    
    # Feature groups
    'SEMANTIC_CATEGORIES', 'COMMON_FEATURES'
]
