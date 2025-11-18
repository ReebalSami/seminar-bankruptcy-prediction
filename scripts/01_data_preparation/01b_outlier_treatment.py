#!/usr/bin/env python3
"""
Script 01b: Outlier Treatment via Winsorization
================================================

PHASE 01: DATA PREPARATION - Step 2 of 4

Purpose:
--------
Apply winsorization to all 64 financial ratio features to dampen extreme values.
Method: Cap at 1st and 99th percentiles (winsorization limits).

Rationale:
----------
- ALL 64 features have outliers (mean: 5.4%, range: 0.07%-15.5%)
- Financial ratios prone to extremes (manipulation, measurement errors, special events)
- Winsorization preserves sample size while dampening extremes
- Established method in empirical finance (better than deletion)
- From Phase 00: 2 features (A27, A6) have >15% outliers

Why NOT Remove Outliers:
-------------------------
- Information loss (would remove ~2,300 observations)
- Class imbalance worse (bankruptcy rate already only 4.82%)
- Winsorization maintains distribution shape

Input:
------
data/processed/poland_no_duplicates.parquet (43,004 observations)

Output:
-------
data/processed/poland_winsorized.parquet (43,004 observations, outliers dampened)
results/01_data_preparation/01b_outlier_treatment.xlsx
results/01_data_preparation/01b_outlier_treatment.html
logs/01b_outlier_treatment.log

Expected Result:
----------------
- All 64 features winsorized
- Min/max values capped at percentiles
- Distribution shape preserved
- No observations removed
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bankruptcy_prediction.utils.logging_setup import setup_logging, print_header, print_section
from src.bankruptcy_prediction.utils.config_loader import get_config

sns.set_style("whitegrid")


def winsorize_features(df, feature_cols, lower_percentile=1, upper_percentile=99, logger=None):
    """
    Winsorize features by capping at specified percentiles.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names to winsorize
    lower_percentile : float
        Lower percentile threshold (default: 1)
    upper_percentile : float
        Upper percentile threshold (default: 99)
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    pd.DataFrame : Winsorized dataframe
    dict : Dictionary of winsorization statistics per feature
    """
    if logger:
        logger.info(f"Winsorizing {len(feature_cols)} features at {lower_percentile}th/{upper_percentile}th percentiles...")
    
    df_winsorized = df.copy()
    winsorization_stats = {}
    
    for feature in feature_cols:
        # Skip if column doesn't exist
        if feature not in df.columns:
            if logger:
                logger.warning(f"Feature {feature} not found, skipping")
            continue
        
        # Get non-null values for percentile calculation
        non_null_values = df[feature].dropna()
        
        if len(non_null_values) == 0:
            if logger:
                logger.warning(f"Feature {feature} has no non-null values, skipping")
            continue
        
        # Calculate percentiles
        lower_bound = np.percentile(non_null_values, lower_percentile)
        upper_bound = np.percentile(non_null_values, upper_percentile)
        
        # Count values affected
        n_below = (df[feature] < lower_bound).sum()
        n_above = (df[feature] > upper_bound).sum()
        n_affected = n_below + n_above
        
        # Original min/max
        original_min = df[feature].min()
        original_max = df[feature].max()
        
        # Apply winsorization
        df_winsorized[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
        
        # New min/max
        new_min = df_winsorized[feature].min()
        new_max = df_winsorized[feature].max()
        
        # Store statistics
        winsorization_stats[feature] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_below': n_below,
            'n_above': n_above,
            'n_affected': n_affected,
            'pct_affected': (n_affected / len(df)) * 100,
            'original_min': original_min,
            'original_max': original_max,
            'new_min': new_min,
            'new_max': new_max
        }
    
    if logger:
        total_affected = sum(s['n_affected'] for s in winsorization_stats.values())
        logger.info(f"Winsorization complete: {total_affected:,} values capped across all features")
    
    return df_winsorized, winsorization_stats


def create_report(df_original, df_winsorized, stats, output_dir, logger):
    """
    Create comprehensive Excel and HTML reports.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataset
    df_winsorized : pd.DataFrame
        Winsorized dataset
    stats : dict
        Winsorization statistics
    output_dir : Path
        Output directory
    logger : logging.Logger
        Logger instance
    """
    logger.info("\nCreating reports...")
    
    excel_path = output_dir / "01b_outlier_treatment.xlsx"
    
    # Prepare statistics dataframe
    stats_df = pd.DataFrame.from_dict(stats, orient='index').reset_index()
    stats_df.columns = ['Feature', 'Lower_Bound', 'Upper_Bound', 'N_Below', 'N_Above', 
                        'N_Affected', 'Pct_Affected', 'Original_Min', 'Original_Max', 
                        'New_Min', 'New_Max']
    
    # Sort by percentage affected (descending)
    stats_df = stats_df.sort_values('Pct_Affected', ascending=False)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Summary
        summary_data = {
            'Metric': [
                'Total Observations',
                'Features Winsorized',
                'Lower Percentile',
                'Upper Percentile',
                '',
                'Total Values Capped',
                'Avg % Affected per Feature',
                'Max % Affected (Single Feature)',
                'Min % Affected (Single Feature)',
                '',
                'Features with >10% Affected',
                'Features with 5-10% Affected',
                'Features with <5% Affected'
            ],
            'Value': [
                f"{len(df_original):,}",
                f"{len(stats)}",
                "1st",
                "99th",
                '',
                f"{sum(s['n_affected'] for s in stats.values()):,}",
                f"{stats_df['Pct_Affected'].mean():.2f}%",
                f"{stats_df['Pct_Affected'].max():.2f}%",
                f"{stats_df['Pct_Affected'].min():.2f}%",
                '',
                f"{(stats_df['Pct_Affected'] > 10).sum()}",
                f"{((stats_df['Pct_Affected'] >= 5) & (stats_df['Pct_Affected'] <= 10)).sum()}",
                f"{(stats_df['Pct_Affected'] < 5).sum()}"
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Full Statistics
        stats_df.to_excel(writer, sheet_name='Winsorization_Stats', index=False)
        
        # Sheet 3: Top 20 Most Affected
        top_20 = stats_df.head(20).copy()
        top_20.to_excel(writer, sheet_name='Top_20_Most_Affected', index=False)
        
        # Sheet 4: Distribution Changes (sample of 10 features)
        sample_features = stats_df.head(10)['Feature'].tolist()
        dist_changes = []
        
        for feature in sample_features:
            dist_changes.append({
                'Feature': feature,
                'Original_Mean': df_original[feature].mean(),
                'New_Mean': df_winsorized[feature].mean(),
                'Original_Std': df_original[feature].std(),
                'New_Std': df_winsorized[feature].std(),
                'Original_Skew': df_original[feature].skew(),
                'New_Skew': df_winsorized[feature].skew(),
                'Original_Kurt': df_original[feature].kurt(),
                'New_Kurt': df_winsorized[feature].kurt()
            })
        
        pd.DataFrame(dist_changes).to_excel(writer, sheet_name='Distribution_Changes', index=False)
        
        # Sheet 5: Verification
        verification = {
            'Check': [
                'Observations Before',
                'Observations After',
                'Observations Lost',
                'Features Processed',
                'All Values Finite',
                'No New Missing Values'
            ],
            'Result': [
                f"{len(df_original):,}",
                f"{len(df_winsorized):,}",
                f"{len(df_original) - len(df_winsorized):,}",
                f"{len(stats)}/64",
                'Yes' if np.isfinite(df_winsorized.select_dtypes(include=[np.number])).all().all() else 'No',
                'Yes' if (df_winsorized.isnull().sum() == df_original.isnull().sum()).all() else 'No'
            ],
            'Status': [
                '✓',
                '✓',
                '✓' if len(df_original) == len(df_winsorized) else '✗',
                '✓' if len(stats) == 64 else '⚠',
                '✓' if np.isfinite(df_winsorized.select_dtypes(include=[np.number])).all().all() else '✗',
                '✓' if (df_winsorized.isnull().sum() == df_original.isnull().sum()).all() else '✗'
            ]
        }
        pd.DataFrame(verification).to_excel(writer, sheet_name='Verification', index=False)
    
    logger.info(f"Excel report saved: {excel_path}")
    
    # Create HTML report
    html_path = output_dir / "01b_outlier_treatment.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>01b: Outlier Treatment Report</title>
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
            .highlight {{ background-color: #fff3cd; }}
        </style>
    </head>
    <body>
        <h1>Phase 01 - Script 01b: Outlier Treatment (Winsorization)</h1>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="metric">
                <span class="metric-name">Total Observations:</span>
                <span class="metric-value">{len(df_original):,}</span>
            </div>
            <div class="metric">
                <span class="metric-name">Features Winsorized:</span>
                <span class="metric-value success">{len(stats)}/64</span>
            </div>
            <div class="metric">
                <span class="metric-name">Method:</span>
                <span class="metric-value">Winsorization at 1st/99th percentiles</span>
            </div>
            <div class="metric">
                <span class="metric-name">Total Values Capped:</span>
                <span class="metric-value warning">{sum(s['n_affected'] for s in stats.values()):,}</span>
            </div>
            <div class="metric">
                <span class="metric-name">Avg % Affected per Feature:</span>
                <span class="metric-value">{stats_df['Pct_Affected'].mean():.2f}%</span>
            </div>
        </div>
        
        <h2>Top 10 Most Affected Features</h2>
        {stats_df.head(10)[['Feature', 'N_Affected', 'Pct_Affected', 'Original_Min', 'Original_Max', 'New_Min', 'New_Max']].to_html(index=False, classes='table')}
        
        <h2>Verification</h2>
        <p><strong>✓ Sample size preserved:</strong> {len(df_winsorized):,} observations</p>
        <p><strong>✓ No observations removed</strong></p>
        <p><strong>✓ Missing values unchanged</strong></p>
        <p><strong>✓ Ready for imputation step</strong></p>
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved: {html_path}")


def main():
    """Main execution function."""
    
    # Setup logging
    logger = setup_logging('01b_outlier_treatment')
    
    print_header(logger, "PHASE 01 - SCRIPT 01b: OUTLIER TREATMENT (WINSORIZATION)")
    logger.info("Starting outlier treatment via winsorization...")
    
    # Setup paths
    data_dir = PROJECT_ROOT / "data" / "processed"
    output_dir = PROJECT_ROOT / "results" / "01_data_preparation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = data_dir / "poland_no_duplicates.parquet"
    output_file = data_dir / "poland_winsorized.parquet"
    
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    
    # Load data
    print_section(logger, "1. Loading Data")
    logger.info(f"Reading: {input_file}")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded: {len(df):,} observations, {len(df.columns)} columns")
    
    # Identify feature columns (A1-A64)
    feature_cols = [col for col in df.columns if col.startswith('A') and col[1:].isdigit()]
    logger.info(f"Identified {len(feature_cols)} feature columns: {feature_cols[:5]}...{feature_cols[-5:]}")
    
    # Check for missing features
    expected_features = [f'A{i}' for i in range(1, 65)]
    missing_features = set(expected_features) - set(feature_cols)
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    # Apply winsorization
    print_section(logger, "2. Applying Winsorization (1st/99th Percentiles)")
    df_winsorized, stats = winsorize_features(
        df, 
        feature_cols, 
        lower_percentile=1, 
        upper_percentile=99, 
        logger=logger
    )
    
    # Analyze results
    print_section(logger, "3. Analyzing Results")
    total_affected = sum(s['n_affected'] for s in stats.values())
    avg_pct = np.mean([s['pct_affected'] for s in stats.values()])
    
    logger.info(f"Total values capped: {total_affected:,}")
    logger.info(f"Average % affected per feature: {avg_pct:.2f}%")
    
    # Most affected features
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['pct_affected'], reverse=True)
    logger.info("\nTop 5 most affected features:")
    for feature, stat in sorted_stats[:5]:
        logger.info(f"  {feature}: {stat['pct_affected']:.2f}% ({stat['n_affected']:,} values)")
    
    # Verify
    print_section(logger, "4. Verification")
    logger.info(f"Observations before: {len(df):,}")
    logger.info(f"Observations after: {len(df_winsorized):,}")
    logger.info(f"Observations lost: {len(df) - len(df_winsorized)}")
    
    assert len(df) == len(df_winsorized), "Sample size changed!"
    logger.info("✓ VERIFIED: Sample size preserved")
    
    # Check missing values unchanged
    missing_before = df[feature_cols].isnull().sum().sum()
    missing_after = df_winsorized[feature_cols].isnull().sum().sum()
    logger.info(f"Missing values before: {missing_before:,}")
    logger.info(f"Missing values after: {missing_after:,}")
    assert missing_before == missing_after, "Missing values changed!"
    logger.info("✓ VERIFIED: Missing values unchanged")
    
    # Save winsorized data
    print_section(logger, "5. Saving Winsorized Data")
    logger.info(f"Saving to: {output_file}")
    df_winsorized.to_parquet(output_file, index=False, compression='snappy')
    logger.info(f"✓ Saved: {len(df_winsorized):,} observations")
    
    # Create reports
    print_section(logger, "6. Creating Reports")
    create_report(df, df_winsorized, stats, output_dir, logger)
    
    # Final summary
    print_section(logger, "COMPLETION SUMMARY")
    logger.info(f"✓ Observations: {len(df_winsorized):,} (no change)")
    logger.info(f"✓ Features winsorized: {len(stats)}/64")
    logger.info(f"✓ Values capped: {total_affected:,}")
    logger.info(f"✓ Average impact: {avg_pct:.2f}% per feature")
    logger.info(f"✓ Method: 1st/99th percentile winsorization")
    logger.info(f"✓ Ready for next step: 01c_missing_value_imputation.py")
    
    print("\n" + "="*80)
    print("SCRIPT 01b COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
