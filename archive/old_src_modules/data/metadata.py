"""
Metadata Parser
===============

Automatically parses feature metadata from JSON and validates against dataframes.
NO HARDCODING - all information comes from JSON file.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


class MetadataParser:
    """
    Parse and manage feature metadata from JSON.
    
    Validates feature labels against dataframe columns to prevent errors.
    
    Attributes:
        metadata: Complete metadata dictionary
        features: Feature descriptions
        categories: Feature categories
        
    Example:
        >>> metadata = MetadataParser.from_default()
        >>> print(metadata.get_readable_name('Attr1'))
        'Net Profit / Total Assets'
        >>> 
        >>> # Validate against dataframe
        >>> metadata.validate_dataframe(df)
        >>> 
        >>> # Apply readable names
        >>> df_readable = metadata.apply_labels(df)
    """
    
    def __init__(self, json_path: Path):
        """
        Initialize metadata parser.
        
        Parameters:
            json_path: Path to feature descriptions JSON file
        """
        self.json_path = Path(json_path)
        self._load_metadata()
    
    def _load_metadata(self):
        """Load and parse metadata from JSON file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.json_path}")
        
        with open(self.json_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Extract key sections
        self.features = self.metadata.get('features', {})
        self.categories = self.metadata.get('categories', {})
        self.dataset_info = self.metadata.get('dataset_info', {})
        
        if not self.features:
            raise ValueError("No features found in metadata JSON")
        
        print(f"✓ Loaded metadata for {len(self.features)} features")
    
    @classmethod
    def from_default(cls):
        """Load metadata from default location in project."""
        project_root = Path(__file__).parent.parent.parent.parent
        default_path = project_root / "data" / "polish-companies-bankruptcy" / "feature_descriptions.json"
        return cls(default_path)
    
    def get_readable_name(self, attr_name: str, short: bool = False) -> str:
        """
        Convert technical attribute name to readable name.
        
        Parameters:
            attr_name: Technical name (e.g., 'Attr1' or 'Attr1__isna')
            short: If True, return short version for plots
        
        Returns:
            Human-readable feature name
        
        Example:
            >>> metadata.get_readable_name('Attr1')
            'Net Profit / Total Assets'
            >>> metadata.get_readable_name('Attr1', short=True)
            'Net Profit / Assets'
        """
        # Handle missingness indicators
        is_missing = '__isna' in attr_name
        base_name = attr_name.replace('__isna', '')
        
        if base_name not in self.features:
            warnings.warn(f"Unknown feature: {base_name}")
            return attr_name
        
        feature_info = self.features[base_name]
        name = feature_info.get('short_name' if short else 'name', attr_name)
        
        if is_missing:
            name += ' (Missing)'
        
        return name
    
    def get_category(self, attr_name: str) -> str:
        """
        Get category for an attribute.
        
        Parameters:
            attr_name: Attribute name (e.g., 'Attr1')
        
        Returns:
            Category name (e.g., 'Profitability')
        """
        base_name = attr_name.replace('__isna', '')
        
        for category, info in self.categories.items():
            if base_name in info.get('features', []):
                return category
        
        return 'Unknown'
    
    def get_interpretation(self, attr_name: str) -> str:
        """
        Get business interpretation for a feature.
        
        Parameters:
            attr_name: Attribute name
        
        Returns:
            Business interpretation text
        """
        base_name = attr_name.replace('__isna', '')
        
        if base_name not in self.features:
            return "No interpretation available"
        
        return self.features[base_name].get('interpretation', 'No interpretation available')
    
    def get_formula(self, attr_name: str) -> str:
        """
        Get calculation formula for a feature.
        
        Parameters:
            attr_name: Attribute name
        
        Returns:
            Formula string
        """
        base_name = attr_name.replace('__isna', '')
        
        if base_name not in self.features:
            return "No formula available"
        
        return self.features[base_name].get('formula', 'No formula available')
    
    def get_features_by_category(self, category: str) -> List[str]:
        """
        Get all features in a category.
        
        Parameters:
            category: Category name
        
        Returns:
            List of feature names
        """
        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}. Available: {list(self.categories.keys())}")
        
        return self.categories[category].get('features', [])
    
    def get_all_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(self.categories.keys())
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          raise_error: bool = False) -> Tuple[bool, List[str], List[str]]:
        """
        Validate that dataframe columns match metadata features.
        
        Parameters:
            df: DataFrame to validate
            raise_error: If True, raise ValueError on mismatch
        
        Returns:
            Tuple of (is_valid, missing_in_metadata, extra_in_df)
        
        Example:
            >>> is_valid, missing, extra = metadata.validate_dataframe(df)
            >>> if not is_valid:
            ...     print(f"Missing: {missing}")
            ...     print(f"Extra: {extra}")
        """
        # Get feature columns (exclude target and metadata)
        exclude = ['y', 'horizon', 'class']
        df_features = [col for col in df.columns if col not in exclude]
        
        # Extract base names (remove __isna suffix)
        df_base_features = set()
        for col in df_features:
            base = col.replace('__isna', '')
            df_base_features.add(base)
        
        metadata_features = set(self.features.keys())
        
        # Find mismatches
        missing_in_metadata = sorted(df_base_features - metadata_features)
        extra_in_df = sorted(metadata_features - df_base_features)
        
        is_valid = len(missing_in_metadata) == 0 and len(extra_in_df) == 0
        
        if not is_valid:
            msg = "DataFrame validation failed:\n"
            if missing_in_metadata:
                msg += f"  Features in DataFrame not in metadata: {missing_in_metadata[:10]}\n"
            if extra_in_df:
                msg += f"  Features in metadata not in DataFrame: {extra_in_df[:10]}\n"
            
            if raise_error:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
        else:
            print("✓ DataFrame validation successful: all features match metadata")
        
        return is_valid, missing_in_metadata, extra_in_df
    
    def apply_labels(self, df: pd.DataFrame, short: bool = False, 
                    inplace: bool = False) -> pd.DataFrame:
        """
        Apply readable column names to dataframe.
        
        Parameters:
            df: Input dataframe
            short: Use short names (for plots)
            inplace: Modify dataframe in place
        
        Returns:
            DataFrame with readable column names
        
        Example:
            >>> df_readable = metadata.apply_labels(df, short=True)
            >>> print(df_readable.columns)
            ['Net Profit / Assets', 'Liabilities / Assets', ...]
        """
        if not inplace:
            df = df.copy()
        
        rename_dict = {}
        for col in df.columns:
            if col in ['y', 'horizon', 'class']:
                continue  # Keep as-is
            rename_dict[col] = self.get_readable_name(col, short=short)
        
        df.rename(columns=rename_dict, inplace=True)
        
        return df
    
    def get_feature_info(self, attr_name: str) -> Dict:
        """
        Get complete information dictionary for a feature.
        
        Parameters:
            attr_name: Attribute name
        
        Returns:
            Dictionary with all feature information
        """
        base_name = attr_name.replace('__isna', '')
        
        if base_name not in self.features:
            return {'error': f'Feature {base_name} not found'}
        
        info = self.features[base_name].copy()
        info['category'] = self.get_category(base_name)
        info['is_missing_indicator'] = '__isna' in attr_name
        
        return info
    
    def create_feature_summary(self) -> pd.DataFrame:
        """
        Create summary table of all features.
        
        Returns:
            DataFrame with feature summaries
        """
        rows = []
        for attr_name in sorted(self.features.keys()):
            info = self.get_feature_info(attr_name)
            rows.append({
                'Feature': attr_name,
                'Name': info.get('name', ''),
                'Short Name': info.get('short_name', ''),
                'Category': info.get('category', ''),
                'Formula': info.get('formula', ''),
                'Interpretation': info.get('interpretation', '')
            })
        
        return pd.DataFrame(rows)
    
    def to_latex(self, output_path: Path):
        """
        Export feature descriptions to LaTeX table for thesis appendix.
        
        Parameters:
            output_path: Path to save LaTeX file
        """
        summary = self.create_feature_summary()
        
        # Format for thesis appendix
        latex_str = summary[['Feature', 'Name', 'Category', 'Formula']].to_latex(
            index=False,
            caption='Financial Ratio Definitions',
            label='tab:feature_definitions',
            column_format='llll',
            longtable=True
        )
        
        with open(output_path, 'w') as f:
            f.write(latex_str)
        
        print(f"✓ LaTeX table saved to: {output_path}")


if __name__ == '__main__':
    # Test the metadata parser
    print("Testing MetadataParser...")
    
    try:
        metadata = MetadataParser.from_default()
        
        print("\n✓ Metadata loaded successfully!")
        print(f"  Features: {len(metadata.features)}")
        print(f"  Categories: {metadata.get_all_categories()}")
        
        print("\n Example feature - Attr1:")
        print(f"  Full name: {metadata.get_readable_name('Attr1')}")
        print(f"  Short name: {metadata.get_readable_name('Attr1', short=True)}")
        print(f"  Category: {metadata.get_category('Attr1')}")
        print(f"  Formula: {metadata.get_formula('Attr1')}")
        print(f"  Interpretation: {metadata.get_interpretation('Attr1')}")
        
        print("\n Profitability features:")
        prof_features = metadata.get_features_by_category('Profitability')
        print(f"  Count: {len(prof_features)}")
        print(f"  First 5: {prof_features[:5]}")
        
        # Try validating against data if it exists
        try:
            from src.bankruptcy_prediction.data.loader import DataLoader
            loader = DataLoader()
            df = loader.load_poland(horizon=1)
            
            print("\n Validating against actual data...")
            is_valid, missing, extra = metadata.validate_dataframe(df)
            
            if is_valid:
                print("  ✓ All features match!")
            else:
                if missing:
                    print(f"  Missing in metadata: {missing[:5]}")
                if extra:
                    print(f"  Extra in metadata: {extra[:5]}")
        
        except Exception as e:
            print(f"\n  (Could not validate against data: {e})")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
