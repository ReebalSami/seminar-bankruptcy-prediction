"""
Bankruptcy Prediction Package
==============================

A modular, object-oriented framework for bankruptcy prediction analysis.

Main Components:
- data: Data loading and metadata management
- features: Feature engineering and selection
- models: Model implementations (Logistic, RF, XGBoost, etc.)
- evaluation: Model evaluation and comparison
- visualization: Professional plotting functions

Usage:
    from bankruptcy_prediction.data import DataLoader, MetadataParser
    from bankruptcy_prediction.models import RandomForestModel
    from bankruptcy_prediction.evaluation import ModelEvaluator
"""

__version__ = "1.0.0"
__author__ = "FH-Wedel Master Thesis"

# Main exports
from src.bankruptcy_prediction.data.loader import DataLoader
from src.bankruptcy_prediction.data.metadata import MetadataParser

__all__ = [
    'DataLoader',
    'MetadataParser',
]
