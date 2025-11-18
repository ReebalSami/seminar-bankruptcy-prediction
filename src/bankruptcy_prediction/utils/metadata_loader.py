"""
Metadata Loader Utilities
==========================

Centralized feature metadata loading for Polish bankruptcy dataset.

Eliminates hard-coded category mappings and provides single source of truth.
"""

import json
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureMetadata:
    """
    Feature metadata manager for Polish bankruptcy dataset.
    
    Loads and provides access to feature descriptions, categories, and formulas.
    """
    
    def __init__(self, metadata_path: Path = None):
        """
        Initialize metadata loader.
        
        Parameters
        ----------
        metadata_path : Path, optional
            Path to feature_descriptions.json. If None, uses default location.
        """
        if metadata_path is None:
            # Default path relative to project root
            project_root = Path(__file__).resolve().parents[3]
            metadata_path = project_root / 'data' / 'polish-companies-bankruptcy' / 'feature_descriptions.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.metadata_path = metadata_path
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from JSON file."""
        with open(self.metadata_path, 'r') as f:
            data = json.load(f)
        
        self.dataset_info = data.get('dataset_info', {})
        self.features = data.get('features', {})
        self.categories = data.get('categories', {})
        
        # Build reverse mappings for fast lookup
        self._feature_to_category = {}
        for category, info in self.categories.items():
            for feature in info.get('features', []):
                self._feature_to_category[feature] = category
        
        logger.info(f"✓ Loaded metadata: {len(self.features)} features, {len(self.categories)} categories")
    
    def get_category(self, feature: str) -> str:
        """
        Get category for a feature.
        
        Parameters
        ----------
        feature : str
            Feature code (e.g., 'A1', 'A15')
        
        Returns
        -------
        str
            Category name (e.g., 'Profitability', 'Liquidity')
        
        Raises
        ------
        KeyError
            If feature not found in metadata
        """
        if feature not in self._feature_to_category:
            raise KeyError(f"Feature '{feature}' not found in metadata")
        
        return self._feature_to_category[feature]
    
    def get_feature_info(self, feature: str) -> Dict:
        """
        Get complete information for a feature.
        
        Parameters
        ----------
        feature : str
            Feature code (e.g., 'A1')
        
        Returns
        -------
        dict
            Dictionary with 'name', 'short_name', 'category', 'formula', 'interpretation'
        """
        if feature not in self.features:
            raise KeyError(f"Feature '{feature}' not found in metadata")
        
        info = self.features[feature].copy()
        # Add category from reverse mapping
        info['category'] = self._feature_to_category.get(feature, 'Unknown')
        
        return info
    
    def get_features_by_category(self, category: str) -> List[str]:
        """
        Get all features in a category.
        
        Parameters
        ----------
        category : str
            Category name
        
        Returns
        -------
        list of str
            Feature codes in that category
        """
        if category not in self.categories:
            raise KeyError(f"Category '{category}' not found. Available: {list(self.categories.keys())}")
        
        return self.categories[category].get('features', [])
    
    def get_expected_direction(self, feature: str) -> str:
        """
        Get expected correlation direction with bankruptcy for economic validation.
        
        Parameters
        ----------
        feature : str
            Feature code
        
        Returns
        -------
        str
            'positive', 'negative', or 'unknown'
        
        Notes
        -----
        - Profitability/Liquidity/Activity: Higher → Lower bankruptcy (negative correlation)
        - Leverage: Higher → Higher bankruptcy (positive correlation)
        - Size/Other: Unknown relationship
        """
        category = self.get_category(feature)
        
        if category in ['Profitability', 'Liquidity', 'Activity']:
            return 'negative'  # Better financial health → less bankruptcy
        elif category == 'Leverage':
            return 'positive'  # More debt → more bankruptcy
        else:
            return 'unknown'
    
    def list_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(self.categories.keys())
    
    def summary(self) -> Dict:
        """
        Get summary statistics about metadata.
        
        Returns
        -------
        dict
            Summary with feature counts per category
        """
        return {
            'total_features': len(self.features),
            'categories': {
                cat: len(info.get('features', []))
                for cat, info in self.categories.items()
            }
        }


# Singleton instance for easy import
_metadata_instance = None

def get_metadata(reload: bool = False) -> FeatureMetadata:
    """
    Get singleton metadata instance.
    
    Parameters
    ----------
    reload : bool, default=False
        Force reload of metadata
    
    Returns
    -------
    FeatureMetadata
        Singleton metadata instance
    """
    global _metadata_instance
    
    if _metadata_instance is None or reload:
        _metadata_instance = FeatureMetadata()
    
    return _metadata_instance
