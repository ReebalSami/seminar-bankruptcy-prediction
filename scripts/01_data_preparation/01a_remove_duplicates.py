#!/usr/bin/env python3
"""
Script 01a: Remove Duplicate Observations
==========================================

PHASE 01: DATA PREPARATION - Step 1 of 4

Purpose:
--------
Remove 401 exact duplicate observations identified in Phase 00.
Conservative approach: Keep first instance, remove duplicates.

Rationale:
----------
- 401 rows are exact duplicates (200 pairs, 1 triple)
- ALL 68 columns identical (features + year + horizon + y)
- Without company ID, cannot verify if same company or data entry error
- Assumption: Data entry errors (documented in paper Chapter 3.1.4)
- Critical: Must happen BEFORE train/test split to prevent data leakage

Input:
------
data/processed/poland_clean_full.parquet (43,405 observations)

Output:
-------
data/processed/poland_no_duplicates.parquet (43,004 observations)
results/01_data_preparation/01a_duplicate_removal.xlsx
results/01_data_preparation/01a_duplicate_removal.html
logs/01a_remove_duplicates.log

Expected Result:
----------------
43,405 - 401 = 43,004 observations
Bankruptcy rate preserved: 4.82%
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bankruptcy_prediction.utils.logging_setup import setup_logging, print_header, print_section
from src.bankruptcy_prediction.utils.config_loader import get_config


def identify_duplicates(df, logger):
    """
    Identify and analyze duplicate patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    pd.DataFrame : DataFrame with duplicate analysis
    """
    logger.info("Identifying duplicates...")
    
    # Check for duplicates across ALL columns
    duplicate_mask = df.duplicated(keep=False)
    n_duplicates = duplicate_mask.sum()
    
    logger.info(f"Total observations: {len(df):,}")
    logger.info(f"Duplicate observations: {n_duplicates:,}")
    logger.info(f"Unique observations: {len(df) - n_duplicates:,}")
    
    if n_duplicates == 0:
        logger.warning("No duplicates found! Expected 401 based on Phase 00.")
        return pd.DataFrame()
    
    # Extract duplicate rows for analysis
    dup_df = df[duplicate_mask].copy()
    
    # Analyze duplicate patterns
    dup_analysis = []
    for idx, row in dup_df.iterrows():
        # Find all rows identical to this one
        identical_mask = (df == row).all(axis=1)
        identical_indices = df[identical_mask].index.tolist()
        
        if len(identical_indices) > 1 and identical_indices[0] == idx:
            # This is the first occurrence
            dup_analysis.append({
                'First_Index': idx,
                'Duplicate_Indices': ','.join(map(str, identical_indices[1:])),
                'Count': len(identical_indices),
                'Horizon': row['horizon'],
                'Year': row['year'],
                'Bankruptcy': row['y']
            })
    
    dup_analysis_df = pd.DataFrame(dup_analysis)
    
    if len(dup_analysis_df) > 0:
        logger.info(f"Duplicate groups identified: {len(dup_analysis_df)}")
        logger.info(f"Expected ~200 pairs = 200 groups")
        
        # Analyze by horizon
        horizon_counts = dup_df.groupby('horizon').size()
        logger.info("\nDuplicates by horizon:")
        for h, count in horizon_counts.items():
            logger.info(f"  H{h}: {count} duplicate observations")
    
    return dup_analysis_df


def remove_duplicates(df, logger):
    """
    Remove duplicate observations, keeping first instance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    pd.DataFrame : Cleaned dataset without duplicates
    dict : Statistics about removal
    """
    logger.info("\nRemoving duplicates (keep='first')...")
    
    # Store original stats
    original_count = len(df)
    original_bankrupt = df['y'].sum()
    original_rate = df['y'].mean() * 100
    
    # Remove duplicates
    df_clean = df.drop_duplicates(keep='first').copy()
    
    # New stats
    new_count = len(df_clean)
    new_bankrupt = df_clean['y'].sum()
    new_rate = df_clean['y'].mean() * 100
    
    removed = original_count - new_count
    
    logger.info(f"Original observations: {original_count:,}")
    logger.info(f"After removal: {new_count:,}")
    logger.info(f"Removed: {removed:,} ({removed/original_count*100:.2f}%)")
    logger.info(f"\nBankruptcy rate (original): {original_rate:.2f}%")
    logger.info(f"Bankruptcy rate (after): {new_rate:.2f}%")
    logger.info(f"Rate change: {new_rate - original_rate:+.4f} pp")
    
    stats = {
        'original_count': original_count,
        'new_count': new_count,
        'removed': removed,
        'original_bankrupt': original_bankrupt,
        'new_bankrupt': new_bankrupt,
        'original_rate': original_rate,
        'new_rate': new_rate,
        'rate_change': new_rate - original_rate
    }
    
    return df_clean, stats


def create_report(df_original, df_clean, dup_analysis_df, stats, output_dir, logger):
    """
    Create comprehensive Excel and HTML reports.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataset
    df_clean : pd.DataFrame
        Cleaned dataset
    dup_analysis_df : pd.DataFrame
        Duplicate analysis
    stats : dict
        Removal statistics
    output_dir : Path
        Output directory
    logger : logging.Logger
        Logger instance
    """
    logger.info("\nCreating reports...")
    
    excel_path = output_dir / "01a_duplicate_removal.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Summary
        summary_data = {
            'Metric': [
                'Original Observations',
                'After Removal',
                'Duplicates Removed',
                'Removal Rate (%)',
                '',
                'Original Bankruptcies',
                'After Removal Bankruptcies',
                'Bankruptcies Lost',
                '',
                'Original Bankruptcy Rate (%)',
                'After Removal Rate (%)',
                'Rate Change (pp)'
            ],
            'Value': [
                f"{stats['original_count']:,}",
                f"{stats['new_count']:,}",
                f"{stats['removed']:,}",
                f"{stats['removed']/stats['original_count']*100:.2f}",
                '',
                f"{stats['original_bankrupt']:,}",
                f"{stats['new_bankrupt']:,}",
                f"{stats['original_bankrupt'] - stats['new_bankrupt']:,}",
                '',
                f"{stats['original_rate']:.2f}",
                f"{stats['new_rate']:.2f}",
                f"{stats['rate_change']:+.4f}"
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Duplicate Analysis
        if len(dup_analysis_df) > 0:
            dup_analysis_df.to_excel(writer, sheet_name='Duplicate_Patterns', index=False)
        
        # Sheet 3: Before - Horizon Distribution
        horizon_before = df_original.groupby('horizon').agg({
            'y': ['count', 'sum', 'mean']
        }).reset_index()
        horizon_before.columns = ['Horizon', 'Total', 'Bankruptcies', 'Rate']
        horizon_before['Rate'] = horizon_before['Rate'] * 100
        horizon_before.to_excel(writer, sheet_name='Before_By_Horizon', index=False)
        
        # Sheet 4: After - Horizon Distribution
        horizon_after = df_clean.groupby('horizon').agg({
            'y': ['count', 'sum', 'mean']
        }).reset_index()
        horizon_after.columns = ['Horizon', 'Total', 'Bankruptcies', 'Rate']
        horizon_after['Rate'] = horizon_after['Rate'] * 100
        horizon_after.to_excel(writer, sheet_name='After_By_Horizon', index=False)
        
        # Sheet 5: Comparison
        comparison = pd.DataFrame({
            'Horizon': horizon_before['Horizon'],
            'Before_Count': horizon_before['Total'],
            'After_Count': horizon_after['Total'],
            'Removed': horizon_before['Total'] - horizon_after['Total'],
            'Before_Rate': horizon_before['Rate'],
            'After_Rate': horizon_after['Rate'],
            'Rate_Change': horizon_after['Rate'] - horizon_before['Rate']
        })
        comparison.to_excel(writer, sheet_name='Horizon_Comparison', index=False)
    
    logger.info(f"Excel report saved: {excel_path}")
    
    # Create HTML report
    html_path = output_dir / "01a_duplicate_removal.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>01a: Duplicate Removal Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .summary {{ background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #ecf0f1; }}
            .metric-name {{ font-weight: bold; color: #555; }}
            .metric-value {{ color: #2c3e50; font-weight: bold; }}
            .success {{ color: #27ae60; }}
            .warning {{ color: #e67e22; }}
            table {{ width: 100%; border-collapse: collapse; background-color: white; margin-top: 20px; }}
            th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #ecf0f1; }}
            tr:hover {{ background-color: #f8f9fa; }}
        </style>
    </head>
    <body>
        <h1>Phase 01 - Script 01a: Duplicate Removal</h1>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="metric">
                <span class="metric-name">Original Observations:</span>
                <span class="metric-value">{stats['original_count']:,}</span>
            </div>
            <div class="metric">
                <span class="metric-name">After Removal:</span>
                <span class="metric-value success">{stats['new_count']:,}</span>
            </div>
            <div class="metric">
                <span class="metric-name">Duplicates Removed:</span>
                <span class="metric-value warning">{stats['removed']:,} ({stats['removed']/stats['original_count']*100:.2f}%)</span>
            </div>
            <div class="metric">
                <span class="metric-name">Original Bankruptcy Rate:</span>
                <span class="metric-value">{stats['original_rate']:.2f}%</span>
            </div>
            <div class="metric">
                <span class="metric-name">After Removal Rate:</span>
                <span class="metric-value">{stats['new_rate']:.2f}%</span>
            </div>
            <div class="metric">
                <span class="metric-name">Rate Change:</span>
                <span class="metric-value">{stats['rate_change']:+.4f} pp</span>
            </div>
        </div>
        
        <h2>Distribution by Horizon</h2>
        {horizon_after.to_html(index=False, classes='table')}
        
        <h2>Verification</h2>
        <p><strong>Expected:</strong> Remove 401 duplicates → 43,004 observations</p>
        <p><strong>Actual:</strong> Removed {stats['removed']} → {stats['new_count']:,} observations</p>
        <p><strong>Status:</strong> <span class="{'success' if stats['removed'] == 401 else 'warning'}">
            {'✓ CORRECT' if stats['removed'] == 401 else '⚠ DEVIATION FROM EXPECTATION'}
        </span></p>
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved: {html_path}")


def main():
    """Main execution function."""
    
    # Setup logging
    logger = setup_logging('01a_remove_duplicates')
    
    print_header(logger, "PHASE 01 - SCRIPT 01a: REMOVE DUPLICATES")
    logger.info("Starting duplicate removal process...")
    
    # Setup paths
    data_dir = PROJECT_ROOT / "data" / "processed"
    output_dir = PROJECT_ROOT / "results" / "01_data_preparation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = data_dir / "poland_clean_full.parquet"
    output_file = data_dir / "poland_no_duplicates.parquet"
    
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    
    # Load data
    print_section(logger, "1. Loading Data")
    logger.info(f"Reading: {input_file}")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded: {len(df):,} observations, {len(df.columns)} columns")
    
    # Identify duplicates
    print_section(logger, "2. Identifying Duplicates")
    dup_analysis_df = identify_duplicates(df, logger)
    
    # Remove duplicates
    print_section(logger, "3. Removing Duplicates")
    df_clean, stats = remove_duplicates(df, logger)
    
    # Verify
    print_section(logger, "4. Verification")
    logger.info(f"Expected: 43,405 - 401 = 43,004 observations")
    logger.info(f"Actual: {stats['original_count']:,} - {stats['removed']} = {stats['new_count']:,}")
    
    if stats['removed'] == 401:
        logger.info("✓ CORRECT: Removed exactly 401 duplicates as expected")
    else:
        logger.warning(f"⚠ DEVIATION: Expected 401, removed {stats['removed']}")
    
    # Save cleaned data
    print_section(logger, "5. Saving Clean Data")
    logger.info(f"Saving to: {output_file}")
    df_clean.to_parquet(output_file, index=False, compression='snappy')
    logger.info(f"✓ Saved: {len(df_clean):,} observations")
    
    # Create reports
    print_section(logger, "6. Creating Reports")
    create_report(df, df_clean, dup_analysis_df, stats, output_dir, logger)
    
    # Final summary
    print_section(logger, "COMPLETION SUMMARY")
    logger.info(f"✓ Input: {stats['original_count']:,} observations")
    logger.info(f"✓ Output: {stats['new_count']:,} observations")
    logger.info(f"✓ Removed: {stats['removed']:,} duplicates")
    logger.info(f"✓ Bankruptcy rate preserved: {stats['new_rate']:.2f}% (change: {stats['rate_change']:+.4f} pp)")
    logger.info(f"✓ Ready for next step: 01b_outlier_treatment.py")
    
    print("\n" + "="*80)
    print("SCRIPT 01a COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
