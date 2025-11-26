"""
Data Loader
===========

Handles loading bankruptcy prediction datasets with validation.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import warnings


class DataLoader:
    """
    Load and validate bankruptcy prediction datasets.
    
    Attributes:
        data_dir: Path to data directory
        processed_dir: Path to processed data
    
    Example:
        >>> loader = DataLoader()
        >>> df = loader.load_poland(horizon=1)
        >>> X, y = loader.get_features_target(df)
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader.
        
        Parameters:
            data_dir: Root data directory. If None, auto-detects from project structure.
        """
        if data_dir is None:
            # Auto-detect from project structure
            self.data_dir = Path(__file__).parent.parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.processed_dir = self.data_dir / "processed"
        self._validate_structure()
    
    def _validate_structure(self):
        """Validate data directory structure exists."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        if not self.processed_dir.exists():
            warnings.warn(f"Processed directory not found: {self.processed_dir}")
    
    def load_poland(self, 
                    horizon: Optional[int] = None,
                    dataset_type: str = 'full') -> pd.DataFrame:
        """
        Load Polish bankruptcy dataset.
        
        Parameters:
            horizon: Prediction horizon (1-5 years). If None, loads all horizons.
            dataset_type: 'full' or 'reduced' feature set
        
        Returns:
            DataFrame with features and target column 'y'
        
        Raises:
            FileNotFoundError: If processed data doesn't exist
            ValueError: If invalid horizon or dataset_type
        """
        if dataset_type not in ['full', 'reduced']:
            raise ValueError(f"dataset_type must be 'full' or 'reduced', got: {dataset_type}")
        
        if horizon is not None and (horizon < 1 or horizon > 5):
            raise ValueError(f"horizon must be 1-5, got: {horizon}")
        
        # Load processed data
        file_path = self.processed_dir / f"poland_clean_{dataset_type}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {file_path}\n"
                f"Run data preprocessing notebook first!"
            )
        
        df = pd.read_parquet(file_path)
        
        # Filter by horizon if specified
        if horizon is not None:
            if 'horizon' not in df.columns:
                raise ValueError("Dataset doesn't have 'horizon' column")
            df = df[df['horizon'] == horizon].copy()
        
        # Validate structure
        self._validate_dataframe(df)
        
        return df
    
    def _validate_dataframe(self, df: pd.DataFrame):
        """Validate loaded dataframe has expected structure."""
        required_cols = ['y']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if df['y'].dtype not in [int, 'int64', 'int32']:
            warnings.warn(f"Target column 'y' has dtype {df['y'].dtype}, expected int")
        
        if df.empty:
            warnings.warn("Loaded dataframe is empty!")
    
    def get_features_target(self, 
                           df: pd.DataFrame,
                           drop_cols: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataframe into features (X) and target (y).
        
        Parameters:
            df: Input dataframe
            drop_cols: Additional columns to drop from features
        
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        if 'y' not in df.columns:
            raise ValueError("DataFrame must have 'y' column")
        
        # Default columns to drop
        default_drop = ['y', 'horizon'] if 'horizon' in df.columns else ['y']
        
        if drop_cols:
            drop_cols = list(set(default_drop + drop_cols))
        else:
            drop_cols = default_drop
        
        y = df['y'].copy()
        X = df.drop(columns=drop_cols)
        
        return X, y
    
    def get_train_test_split(self,
                             df: pd.DataFrame,
                             test_size: float = 0.2,
                             random_state: int = 42,
                             stratify: bool = True) -> Tuple:
        """
        Split data into train and test sets.
        
        Parameters:
            df: Input dataframe
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify by target
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        X, y = self.get_features_target(df)
        
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
    
    def get_info(self, df: pd.DataFrame) -> dict:
        """
        Get summary information about dataset.
        
        Parameters:
            df: Input dataframe
        
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,  # Exclude target
            'bankruptcy_count': df['y'].sum() if 'y' in df.columns else None,
            'bankruptcy_rate': df['y'].mean() if 'y' in df.columns else None,
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }
        
        if 'horizon' in df.columns:
            info['horizons'] = sorted(df['horizon'].unique())
        
        return info


if __name__ == '__main__':
    # Test the loader
    loader = DataLoader()
    
    print("Testing DataLoader...")
    print(f"Data directory: {loader.data_dir}")
    
    try:
        df = loader.load_poland(horizon=1, dataset_type='full')
        info = loader.get_info(df)
        
        print("\n✓ Dataset loaded successfully!")
        print(f"  Samples: {info['n_samples']:,}")
        print(f"  Features: {info['n_features']}")
        print(f"  Bankruptcy rate: {info['bankruptcy_rate']:.2%}")
        
        X, y = loader.get_features_target(df)
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
