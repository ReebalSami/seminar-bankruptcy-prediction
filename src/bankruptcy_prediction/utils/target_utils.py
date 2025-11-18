"""
Target Variable Utilities
=========================

Canonical target handling for bankruptcy prediction datasets.

Standard:
- Target column: 'y' (binary: 0=healthy, 1=bankrupt)
- All scripts should use get_canonical_target() for consistency
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_canonical_target(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Ensure canonical target column 'y' exists and remove duplicates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that may contain 'y', 'bankrupt', or both
    drop_duplicates : bool, default=True
        Whether to drop duplicate target columns after verification
    
    Returns
    -------
    pd.DataFrame
        DataFrame with canonical 'y' column only
    
    Raises
    ------
    ValueError
        If no target column found or if columns are inconsistent
    
    Examples
    --------
    >>> df = pd.read_parquet('data.parquet')
    >>> df = get_canonical_target(df)
    >>> # Now use df['y'] consistently
    """
    # Check what target columns exist
    has_y = 'y' in df.columns
    has_bankrupt = 'bankrupt' in df.columns
    
    if not has_y and not has_bankrupt:
        raise ValueError("No target column found. Expected 'y' or 'bankrupt'.")
    
    # If both exist, verify they're identical
    if has_y and has_bankrupt:
        if not (df['y'] == df['bankrupt']).all():
            raise ValueError(
                "Columns 'y' and 'bankrupt' exist but are not identical! "
                "This indicates a data integrity issue."
            )
        logger.info("✓ Verified: 'y' and 'bankrupt' columns are identical")
        
        if drop_duplicates:
            df = df.drop(columns=['bankrupt'])
            logger.info("✓ Dropped redundant 'bankrupt' column, using canonical 'y'")
    
    # If only 'bankrupt' exists, rename to 'y'
    elif has_bankrupt and not has_y:
        df = df.rename(columns={'bankrupt': 'y'})
        logger.info("✓ Renamed 'bankrupt' → 'y' (canonical)")
    
    # If only 'y' exists, all good
    else:
        logger.debug("✓ Canonical target 'y' already present")
    
    # Verify target is binary
    unique_vals = df['y'].dropna().unique()
    if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Target 'y' must be binary (0/1), found: {unique_vals}")
    
    logger.info(f"✓ Canonical target ready: {df['y'].sum()} bankrupt / {len(df)} total ({df['y'].mean()*100:.2f}%)")
    
    return df


def validate_target_distribution(df: pd.DataFrame, min_positive_rate: float = 0.01) -> None:
    """
    Validate target variable distribution for modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'y' column
    min_positive_rate : float, default=0.01
        Minimum acceptable positive class rate (1%)
    
    Raises
    ------
    ValueError
        If target distribution is invalid for modeling
    """
    if 'y' not in df.columns:
        raise ValueError("Target column 'y' not found. Run get_canonical_target() first.")
    
    # Check for missing values
    missing = df['y'].isna().sum()
    if missing > 0:
        raise ValueError(f"Target 'y' has {missing} missing values. Clean data first.")
    
    # Check class balance
    positive_rate = df['y'].mean()
    
    if positive_rate < min_positive_rate:
        raise ValueError(
            f"Severe class imbalance: only {positive_rate*100:.2f}% positive. "
            f"Need at least {min_positive_rate*100}%."
        )
    
    if positive_rate > 0.5:
        logger.warning(
            f"⚠️  Unusual: Positive class is majority ({positive_rate*100:.1f}%). "
            "Verify data or consider inverting labels."
        )
    
    logger.info(f"✓ Target distribution valid: {positive_rate*100:.2f}% positive class")
