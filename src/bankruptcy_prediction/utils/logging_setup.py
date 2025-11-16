"""
Logging setup utility.
Creates consistent logging across all scripts.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from .config_loader import get_config


def setup_logging(
    script_name: str,
    log_dir: Optional[Path] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG"
) -> logging.Logger:
    """
    Setup logging for a script.
    
    Parameters
    ----------
    script_name : str
        Name of the script (e.g., '00a_cross_dataset_semantic_mapping')
    log_dir : Path, optional
        Directory for log files. If None, uses project logs directory.
    console_level : str
        Logging level for console output
    file_level : str
        Logging level for file output
        
    Returns
    -------
    logging.Logger
        Configured logger
        
    Examples
    --------
    >>> logger = setup_logging('00a_cross_dataset_semantic_mapping')
    >>> logger.info("Starting analysis...")
    """
    # Get config
    config = get_config()
    
    # Find project root
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / 'pyproject.toml').exists():
            project_root = current
            break
        current = current.parent
    else:
        project_root = Path.cwd()
    
    # Setup log directory
    if log_dir is None:
        log_dir = project_root / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_format = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = log_dir / f"{script_name}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_format = logging.Formatter(
        config.get('logging', 'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        datefmt=config.get('logging', 'date_format', '%Y-%m-%d %H:%M:%S')
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def print_header(logger: logging.Logger, title: str, width: int = 80) -> None:
    """
    Print a formatted header to logger.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    title : str
        Header title
    width : int
        Width of header line
    """
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)


def print_section(logger: logging.Logger, title: str, width: int = 80) -> None:
    """
    Print a section header to logger.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    title : str
        Section title
    width : int
        Width of section line
    """
    logger.info("")
    logger.info("-" * width)
    logger.info(title)
    logger.info("-" * width)
