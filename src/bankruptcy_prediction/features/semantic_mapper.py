"""
Semantic Feature Mapper
=======================

Maps features across datasets based on SEMANTIC meaning, not position.
This is the CRITICAL missing component that caused cross-dataset transfer to fail.

Key Principle:
--------------
Polish A1 ≠ American X1 ≠ Taiwan F01 (positional matching is WRONG!)
Instead: Group by semantic category (profitability, liquidity, leverage, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from ..utils.config_loader import get_config


class SemanticFeatureMapper:
    """
    Map features across datasets based on semantic meaning.
    
    This class creates mappings like:
    {
        'profitability': {
            'polish': ['A1', 'A6', 'A8'],
            'american': ['X2', 'X3'],
            'taiwan': ['ROA(C) before interest...', 'ROA(A) before interest...']
        },
        'liquidity': {
            'polish': ['A27', 'A28'],
            'american': ['X1', 'X4'],
            'taiwan': ['Current Ratio', 'Quick Ratio']
        },
        ...
    }
    
    Example
    -------
    >>> mapper = SemanticFeatureMapper()
    >>> mapper.build_mapping()
    >>> aligned = mapper.get_aligned_features('profitability')
    >>> print(aligned['polish'])  # ['A1', 'A6', 'A8']
    """
    
    def __init__(self):
        """Initialize semantic feature mapper."""
        self.config = get_config()
        self.project_root = self._find_project_root()
        self.mapping = {}
        self.metadata = self._load_metadata()
    
    def _find_project_root(self) -> Path:
        """Find project root directory."""
        current = Path(__file__).resolve()
        while current.parent != current:
            if (current / 'pyproject.toml').exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def _load_metadata(self) -> Dict:
        """Load metadata for all datasets."""
        metadata = {}
        
        # Polish
        polish_meta_path = self.project_root / self.config.get('datasets', 'polish', 'metadata_path')
        if polish_meta_path.exists():
            with open(polish_meta_path, 'r') as f:
                metadata['polish'] = json.load(f)
        
        # American (Croissant format)
        american_meta_path = self.project_root / self.config.get('datasets', 'american', 'metadata_path')
        if american_meta_path.exists():
            with open(american_meta_path, 'r') as f:
                american_meta_raw = json.load(f)
                # Extract features from Croissant recordSet format
                if 'recordSet' in american_meta_raw and len(american_meta_raw['recordSet']) > 0:
                    fields = american_meta_raw['recordSet'][0].get('field', [])
                    # Filter for X features (X1-X18)
                    x_features = {}
                    for field in fields:
                        name = field.get('name', '')
                        if name.startswith('X') and len(name) <= 3:
                            x_features[name] = {
                                'description': field.get('description', ''),
                                'dataType': field.get('dataType', '')
                            }
                    metadata['american'] = {'features': x_features}
        
        # Taiwan - extract from CSV headers
        taiwan_csv_path = self.project_root / self.config.get('datasets', 'taiwan', 'raw_path')
        if taiwan_csv_path.exists():
            df = pd.read_csv(taiwan_csv_path, nrows=0)
            taiwan_features = [col.strip() for col in df.columns if col != 'Bankrupt?']
            metadata['taiwan'] = {
                'features': taiwan_features,
                'count': len(taiwan_features)
            }
        
        return metadata
    
    def build_mapping(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Build semantic feature mapping across datasets.
        
        Uses keyword matching and manual rules to categorize features.
        
        Returns
        -------
        Dict[str, Dict[str, List[str]]]
            Nested dict: category -> dataset -> feature list
        """
        categories = self.config.get('semantic_categories', default={})
        mapping = {cat: {'polish': [], 'american': [], 'taiwan': []} for cat in categories}
        
        # =====================================================
        # POLISH FEATURES (A1-A64)
        # =====================================================
        if 'polish' in self.metadata:
            features = self.metadata['polish'].get('features', {})
            
            for feature_code, feature_info in features.items():
                category = feature_info.get('category', 'unknown')
                formula = feature_info.get('formula', '').lower()
                name = feature_info.get('name', '').lower()
                
                # Map category to semantic category
                if category in mapping:
                    mapping[category]['polish'].append(feature_code)
                else:
                    # Try keyword matching
                    for sem_cat, sem_info in categories.items():
                        keywords = sem_info.get('keywords', [])
                        if any(kw in formula or kw in name for kw in keywords):
                            mapping[sem_cat]['polish'].append(feature_code)
                            break
        
        # =====================================================
        # AMERICAN FEATURES (X1-X18)
        # =====================================================
        if 'american' in self.metadata:
            features = self.metadata['american'].get('features', {})
            
            for feature_code, feature_info in features.items():
                description = feature_info.get('description', '').lower()
                
                # Keyword matching based on description
                matched = False
                for sem_cat, sem_info in categories.items():
                    keywords = sem_info.get('keywords', [])
                    if any(kw in description for kw in keywords):
                        mapping[sem_cat]['american'].append(feature_code)
                        matched = True
                        break
                
                # Manual category assignment for tricky cases
                if not matched:
                    # Based on typical accounting metrics
                    if feature_code in ['X1', 'X4']:  # Current assets, EBITDA
                        mapping['liquidity']['american'].append(feature_code)
                    elif feature_code in ['X7', 'X8', 'X12']:  # Net income, sales
                        mapping['profitability']['american'].append(feature_code)
                    elif feature_code in ['X9', 'X10', 'X11']:  # Total debt, equity
                        mapping['leverage']['american'].append(feature_code)
                    elif feature_code in ['X2', 'X3', 'X5', 'X6']:  # COGS, depreciation, inventory
                        mapping['efficiency']['american'].append(feature_code)
                    elif feature_code in ['X13', 'X14', 'X15', 'X16', 'X17', 'X18']:
                        mapping['activity']['american'].append(feature_code)
        
        # =====================================================
        # TAIWAN FEATURES (Descriptive names)
        # =====================================================
        if 'taiwan' in self.metadata:
            taiwan_features = self.metadata['taiwan'].get('features', [])
            
            for feature_name in taiwan_features:
                feature_lower = feature_name.lower()
                
                # Keyword matching
                for sem_cat, sem_info in categories.items():
                    keywords = sem_info.get('keywords', [])
                    if any(kw in feature_lower for kw in keywords):
                        mapping[sem_cat]['taiwan'].append(feature_name)
                        break
        
        self.mapping = mapping
        return mapping
    
    def get_aligned_features(self, category: str) -> Dict[str, List[str]]:
        """
        Get aligned features for a specific semantic category.
        
        Parameters
        ----------
        category : str
            Semantic category name
            
        Returns
        -------
        Dict[str, List[str]]
            Dict mapping dataset -> feature list
        """
        if not self.mapping:
            self.build_mapping()
        
        return self.mapping.get(category, {'polish': [], 'american': [], 'taiwan': []})
    
    def get_common_categories(self, 
                             min_features_per_dataset: int = 1) -> List[str]:
        """
        Get categories that have features in all three datasets.
        
        Parameters
        ----------
        min_features_per_dataset : int
            Minimum number of features required per dataset
            
        Returns
        -------
        List[str]
            List of common categories
        """
        if not self.mapping:
            self.build_mapping()
        
        common = []
        for category, datasets in self.mapping.items():
            if all(len(features) >= min_features_per_dataset 
                   for features in datasets.values()):
                common.append(category)
        
        return common
    
    def align_dataframes(self, 
                        df_polish: pd.DataFrame,
                        df_american: pd.DataFrame,
                        df_taiwan: pd.DataFrame,
                        category: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Align three dataframes to same semantic features.
        
        Parameters
        ----------
        df_polish, df_american, df_taiwan : pd.DataFrame
            Input dataframes
        category : str
            Semantic category to align on
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Aligned dataframes (only common features + target)
        """
        aligned_features = self.get_aligned_features(category)
        
        # Extract features and targets
        df_p_aligned = df_polish[aligned_features['polish'] + ['y']].copy()
        df_a_aligned = df_american[aligned_features['american'] + ['y']].copy()
        df_t_aligned = df_taiwan[aligned_features['taiwan'] + ['y']].copy()
        
        return df_p_aligned, df_a_aligned, df_t_aligned
    
    def save_mapping(self, output_path: Path) -> None:
        """
        Save semantic mapping to CSV for documentation.
        
        Parameters
        ----------
        output_path : Path
            Path to save mapping
        """
        if not self.mapping:
            self.build_mapping()
        
        rows = []
        for category, datasets in self.mapping.items():
            max_len = max(len(features) for features in datasets.values())
            
            for i in range(max_len):
                rows.append({
                    'Category': category,
                    'Polish': datasets['polish'][i] if i < len(datasets['polish']) else '',
                    'American': datasets['american'][i] if i < len(datasets['american']) else '',
                    'Taiwan': datasets['taiwan'][i] if i < len(datasets['taiwan']) else ''
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics of semantic mapping.
        
        Returns
        -------
        pd.DataFrame
            Summary with category and feature counts per dataset
        """
        if not self.mapping:
            self.build_mapping()
        
        summary = []
        for category, datasets in self.mapping.items():
            summary.append({
                'Category': category,
                'Polish_Count': len(datasets['polish']),
                'American_Count': len(datasets['american']),
                'Taiwan_Count': len(datasets['taiwan']),
                'Total': sum(len(f) for f in datasets.values())
            })
        
        return pd.DataFrame(summary)


# Convenience function
def get_semantic_mapper() -> SemanticFeatureMapper:
    """Get semantic feature mapper instance."""
    return SemanticFeatureMapper()
