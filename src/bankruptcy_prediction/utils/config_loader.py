"""
Configuration loader utility.
Loads project configuration from YAML file.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and access project configuration."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern - only one config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize config loader."""
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: Path = None) -> None:
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        config_path : Path, optional
            Path to config file. If None, uses default location.
        """
        if config_path is None:
            # Find project root (has pyproject.toml)
            current = Path(__file__).resolve()
            while current.parent != current:
                if (current / 'pyproject.toml').exists():
                    config_path = current / 'config' / 'project_config.yaml'
                    break
                current = current.parent
            
            if config_path is None or not config_path.exists():
                raise FileNotFoundError("Could not find project_config.yaml")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.
        
        Parameters
        ----------
        *keys : str
            Nested keys to access
        default : Any
            Default value if key not found
            
        Returns
        -------
        Any
            Configuration value
            
        Examples
        --------
        >>> config = ConfigLoader()
        >>> config.get('datasets', 'polish', 'feature_count')
        64
        >>> config.get('analysis', 'random_state')
        42
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get full configuration for a specific dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of dataset ('polish', 'american', or 'taiwan')
            
        Returns
        -------
        Dict[str, Any]
            Dataset configuration
        """
        return self.get('datasets', dataset_name, default={})
    
    def get_analysis_param(self, param_name: str, default: Any = None) -> Any:
        """
        Get analysis parameter.
        
        Parameters
        ----------
        param_name : str
            Parameter name
        default : Any
            Default value if not found
            
        Returns
        -------
        Any
            Parameter value
        """
        return self.get('analysis', param_name, default=default)
    
    @property
    def random_state(self) -> int:
        """Get random state for reproducibility."""
        return self.get_analysis_param('random_state', 42)
    
    @property
    def test_size(self) -> float:
        """Get test split size."""
        return self.get_analysis_param('test_size', 0.2)
    
    @property
    def vif_threshold(self) -> float:
        """Get VIF threshold for multicollinearity."""
        return self.get_analysis_param('vif_threshold_high', 10)


# Convenience function for global access
def get_config() -> ConfigLoader:
    """Get configuration instance."""
    return ConfigLoader()
