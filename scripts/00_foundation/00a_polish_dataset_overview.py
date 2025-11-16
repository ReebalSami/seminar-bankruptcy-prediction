#!/usr/bin/env python3
"""
Script 00a: Polish Dataset Overview and Feature Name Mapping
=============================================================

FOUNDATION PHASE - ANALYSIS FIRST, THEN CODE

Purpose:
--------
1. Load and understand raw Polish bankruptcy dataset
2. Map feature codes (A1-A64) to human-readable names from JSON metadata
3. Generate comprehensive dataset statistics
4. Create professional Excel report + HTML dashboard

This script DOES NOT preprocess or clean data - only describes it.

Output:
-------
- results/00_foundation/00a_polish_overview.xlsx (Excel with multiple sheets)
- results/00_foundation/00a_polish_overview.html (Professional HTML dashboard)
- results/00_foundation/00a_feature_mapping.csv (A1 → readable name mapping)
- logs/00a_polish_dataset_overview.log

Key Principle:
--------------
Work with READABLE feature names, not codes.
After this script: "net_profit_total_assets" NOT "A1"
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bankruptcy_prediction.utils.logging_setup import setup_logging, print_header, print_section
from src.bankruptcy_prediction.utils.config_loader import get_config


# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_feature_metadata(json_path: Path, logger) -> dict:
    """
    Load and parse Polish feature metadata from JSON.
    
    Returns
    -------
    dict with keys:
        'feature_map': A1 -> {name, short_name, category, formula, interpretation}
        'categories': Category -> list of features
        'readable_names': A1 -> "readable_column_name"
    """
    logger.info("Loading feature metadata...")
    
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    feature_map = metadata['features']
    categories = metadata['categories']
    
    # Create readable column names (snake_case for code)
    readable_names = {}
    for code, info in feature_map.items():
        # Use short_name converted to snake_case
        readable = info['short_name'].lower()
        readable = readable.replace(' / ', '_').replace('/', '_')
        readable = readable.replace(' ', '_').replace('-', '_')
        readable = readable.replace('(', '').replace(')', '')
        readable = readable.replace('.', '').replace(',', '')
        readable_names[code] = readable
    
    logger.info(f"✓ Loaded metadata for {len(feature_map)} features")
    logger.info(f"✓ {len(categories)} categories defined")
    logger.info("")
    
    return {
        'feature_map': feature_map,
        'categories': categories,
        'readable_names': readable_names
    }


def load_raw_data(data_path: Path, logger) -> pd.DataFrame:
    """Load raw Polish data with basic validation."""
    logger.info(f"Loading raw data: {data_path.name}")
    
    df = pd.read_parquet(data_path)
    
    logger.info(f"✓ Loaded successfully")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns[:5])}... (+{len(df.columns)-5} more)")
    logger.info("")
    
    return df


def create_feature_reference_table(metadata: dict, logger) -> pd.DataFrame:
    """
    Create comprehensive feature reference table.
    
    Columns:
    - Feature_Code: A1, A2, etc.
    - Readable_Name: net_profit_assets, etc.
    - Full_Name: Net Profit / Total Assets
    - Category: Profitability, Liquidity, etc.
    - Formula: net profit / total assets
    - Interpretation: What it means
    """
    logger.info("Creating feature reference table...")
    
    feature_map = metadata['feature_map']
    readable_names = metadata['readable_names']
    
    ref_data = []
    for code in sorted(feature_map.keys(), key=lambda x: int(x[1:])):
        info = feature_map[code]
        ref_data.append({
            'Feature_Code': code,
            'Readable_Name': readable_names[code],
            'Full_Name': info['name'],
            'Short_Name': info['short_name'],
            'Category': info['category'],
            'Formula': info['formula'],
            'Interpretation': info['interpretation']
        })
    
    ref_df = pd.DataFrame(ref_data)
    
    logger.info(f"✓ Created reference for {len(ref_df)} features")
    logger.info("")
    
    return ref_df


def analyze_dataset_structure(df: pd.DataFrame, metadata: dict, logger) -> dict:
    """
    Comprehensive dataset structure analysis.
    
    Returns statistics dict for reporting.
    """
    logger.info("Analyzing dataset structure...")
    
    stats = {}
    
    # Basic dimensions
    stats['total_observations'] = len(df)
    stats['total_features'] = len([col for col in df.columns if col.startswith('A')])
    stats['has_horizon'] = 'horizon' in df.columns
    stats['has_target'] = 'y' in df.columns or 'class' in df.columns
    
    # Target column
    target_col = 'y' if 'y' in df.columns else 'class'
    if target_col in df.columns:
        stats['target_column'] = target_col
        stats['bankruptcy_count'] = (df[target_col] == 1).sum()
        stats['healthy_count'] = (df[target_col] == 0).sum()
        stats['bankruptcy_rate'] = stats['bankruptcy_count'] / len(df) * 100
    
    # Horizons
    if 'horizon' in df.columns:
        stats['horizons'] = sorted(df['horizon'].unique().tolist())
        stats['horizon_count'] = len(stats['horizons'])
        
        # Per-horizon statistics
        horizon_stats = df.groupby('horizon').agg({
            target_col: ['count', 'sum', 'mean']
        }).round(4)
        horizon_stats.columns = ['Total_Obs', 'Bankruptcies', 'Bankruptcy_Rate']
        stats['horizon_distribution'] = horizon_stats
    
    # Feature categories
    category_counts = {}
    for cat, cat_info in metadata['categories'].items():
        category_counts[cat] = len(cat_info['features'])
    stats['category_counts'] = category_counts
    
    logger.info(f"✓ Dataset structure analysis complete")
    logger.info(f"  Total observations: {stats['total_observations']:,}")
    logger.info(f"  Features: {stats['total_features']}")
    if stats['has_horizon']:
        logger.info(f"  Horizons: {stats['horizons']}")
    logger.info(f"  Bankruptcy rate: {stats['bankruptcy_rate']:.2f}%")
    logger.info("")
    
    return stats


def create_excel_report(
    ref_df: pd.DataFrame,
    stats: dict,
    df: pd.DataFrame,
    output_path: Path,
    logger
):
    """
    Create comprehensive Excel report with multiple sheets.
    
    Sheets:
    1. Summary - High-level overview
    2. Feature_Reference - Full feature mapping table
    3. Category_Summary - Features by category
    4. Horizon_Distribution - Bankruptcy rates by horizon
    5. Sample_Data - First 100 rows with readable names
    """
    logger.info("Creating Excel report...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Summary
        summary_data = {
            'Metric': [
                'Total Observations',
                'Total Features',
                'Bankruptcies',
                'Healthy Companies',
                'Bankruptcy Rate (%)',
                'Horizons Available',
                'Categories'
            ],
            'Value': [
                f"{stats['total_observations']:,}",
                stats['total_features'],
                f"{stats['bankruptcy_count']:,}",
                f"{stats['healthy_count']:,}",
                f"{stats['bankruptcy_rate']:.2f}",
                ', '.join(map(str, stats.get('horizons', []))),
                ', '.join(stats['category_counts'].keys())
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Feature Reference
        ref_df.to_excel(writer, sheet_name='Feature_Reference', index=False)
        
        # Sheet 3: Category Summary
        cat_summary = pd.DataFrame([
            {'Category': cat, 'Feature_Count': count, 'Percentage': f"{count/64*100:.1f}%"}
            for cat, count in stats['category_counts'].items()
        ])
        cat_summary.to_excel(writer, sheet_name='Category_Summary', index=False)
        
        # Sheet 4: Horizon Distribution
        if 'horizon_distribution' in stats:
            horizon_dist = stats['horizon_distribution'].reset_index()
            horizon_dist['Bankruptcy_Rate'] = (horizon_dist['Bankruptcy_Rate'] * 100).round(2)
            horizon_dist.to_excel(writer, sheet_name='Horizon_Distribution', index=False)
        
        # Sheet 5: Sample Data (first 100 rows with readable names)
        # This shows how the data will look with readable column names
        sample = df.head(100).copy()
        # Note: We'll do actual renaming in data preparation phase
        sample.to_excel(writer, sheet_name='Sample_Data_Raw_Names', index=False)
    
    logger.info(f"✓ Excel report saved: {output_path}")
    logger.info("")


def create_html_dashboard(
    ref_df: pd.DataFrame,
    stats: dict,
    output_path: Path,
    logger
):
    """
    Create professional HTML dashboard with visualizations.
    
    NO assistant messages - final for third party viewing.
    """
    logger.info("Creating HTML dashboard...")
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polish Bankruptcy Dataset - Foundation Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-value {{
            font-size: 2.2em;
            font-weight: bold;
            color: #1e3c72;
        }}
        
        .section {{
            margin: 40px 0;
        }}
        
        .section-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #1e3c72;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: #1e3c72;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:hover {{
            background: #f5f7fa;
        }}
        
        .category-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            color: white;
        }}
        
        .profitability {{ background: #10b981; }}
        .liquidity {{ background: #3b82f6; }}
        .leverage {{ background: #ef4444; }}
        .activity {{ background: #f59e0b; }}
        .size {{ background: #8b5cf6; }}
        .other {{ background: #6b7280; }}
        
        .footer {{
            background: #f5f7fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        
        .horizon-table {{
            max-width: 800px;
            margin: 20px auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Polish Companies Bankruptcy Prediction</h1>
            <p>Foundation Phase - Dataset Overview & Feature Mapping</p>
            <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">FH Wedel | Seminar Project 2024/2025</p>
        </header>
        
        <div class="content">
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Observations</div>
                    <div class="metric-value">{stats['total_observations']:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Features</div>
                    <div class="metric-value">{stats['total_features']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Bankruptcies</div>
                    <div class="metric-value">{stats['bankruptcy_count']:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Bankruptcy Rate</div>
                    <div class="metric-value">{stats['bankruptcy_rate']:.2f}%</div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Feature Categories Distribution</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Feature Count</th>
                            <th>Percentage</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Add category rows
    for cat, count in stats['category_counts'].items():
        cat_class = cat.lower()
        pct = count / 64 * 100
        desc = "Profitability and return metrics" if cat == "Profitability" else \
               "Short-term obligation coverage" if cat == "Liquidity" else \
               "Debt and capital structure" if cat == "Leverage" else \
               "Operational efficiency ratios" if cat == "Activity" else \
               "Company size indicator" if cat == "Size" else "Other metrics"
        
        html += f"""
                        <tr>
                            <td><span class="category-badge {cat_class}">{cat}</span></td>
                            <td>{count}</td>
                            <td>{pct:.1f}%</td>
                            <td>{desc}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
"""
    
    # Horizon distribution
    if 'horizon_distribution' in stats:
        html += """
            <div class="section">
                <h2 class="section-title">Bankruptcy Rates by Prediction Horizon</h2>
                <table class="horizon-table">
                    <thead>
                        <tr>
                            <th>Horizon (Years)</th>
                            <th>Total Observations</th>
                            <th>Bankruptcies</th>
                            <th>Rate (%)</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for idx, row in stats['horizon_distribution'].iterrows():
            html += f"""
                        <tr>
                            <td>H{idx}</td>
                            <td>{int(row['Total_Obs']):,}</td>
                            <td>{int(row['Bankruptcies']):,}</td>
                            <td>{row['Bankruptcy_Rate']*100:.2f}%</td>
                        </tr>
"""
        
        html += """
                    </tbody>
                </table>
            </div>
"""
    
    # Feature reference sample
    html += """
            <div class="section">
                <h2 class="section-title">Feature Reference (Sample)</h2>
                <p style="margin-bottom: 20px; color: #666;">Showing first 10 features. Full reference available in Excel file.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Code</th>
                            <th>Readable Name</th>
                            <th>Full Name</th>
                            <th>Category</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for _, row in ref_df.head(10).iterrows():
        cat_class = row['Category'].lower()
        html += f"""
                        <tr>
                            <td><strong>{row['Feature_Code']}</strong></td>
                            <td><code>{row['Readable_Name']}</code></td>
                            <td>{row['Full_Name']}</td>
                            <td><span class="category-badge {cat_class}">{row['Category']}</span></td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Foundation Phase - Script 00a</strong></p>
            <p>This analysis establishes the foundation for all subsequent work.</p>
            <p style="margin-top: 10px;">Next: Data Preparation (01_data_preparation)</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"✓ HTML dashboard saved: {output_path}")
    logger.info("")


def main():
    """Execute Polish dataset overview analysis."""
    
    # Setup
    logger = setup_logging('00a_polish_dataset_overview')
    config = get_config()
    output_dir = PROJECT_ROOT / 'results' / '00_foundation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header(logger, "SCRIPT 00a: POLISH DATASET OVERVIEW")
    logger.info("Foundation Phase - Analysis First, Then Code")
    logger.info("Goal: Understand dataset, map feature names, create reference")
    logger.info("")
    
    # =====================================================
    # STEP 1: Load Feature Metadata
    # =====================================================
    print_section(logger, "STEP 1: Load Feature Metadata from JSON")
    
    metadata_path = PROJECT_ROOT / 'data' / 'polish-companies-bankruptcy' / 'feature_descriptions.json'
    metadata = load_feature_metadata(metadata_path, logger)
    
    # =====================================================
    # STEP 2: Load Raw Data
    # =====================================================
    print_section(logger, "STEP 2: Load Raw Polish Data")
    
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'poland_clean_full.parquet'
    df = load_raw_data(data_path, logger)
    
    # =====================================================
    # STEP 3: Create Feature Reference Table
    # =====================================================
    print_section(logger, "STEP 3: Create Feature Reference Table")
    
    ref_df = create_feature_reference_table(metadata, logger)
    
    # Save as CSV for easy reference
    ref_csv_path = output_dir / '00a_feature_mapping.csv'
    ref_df.to_csv(ref_csv_path, index=False)
    logger.info(f"✓ Feature mapping CSV saved: {ref_csv_path}")
    logger.info("")
    
    # =====================================================
    # STEP 4: Analyze Dataset Structure
    # =====================================================
    print_section(logger, "STEP 4: Analyze Dataset Structure")
    
    stats = analyze_dataset_structure(df, metadata, logger)
    
    # =====================================================
    # STEP 5: Create Excel Report
    # =====================================================
    print_section(logger, "STEP 5: Create Excel Report")
    
    excel_path = output_dir / '00a_polish_overview.xlsx'
    create_excel_report(ref_df, stats, df, excel_path, logger)
    
    # =====================================================
    # STEP 6: Create HTML Dashboard
    # =====================================================
    print_section(logger, "STEP 6: Create HTML Dashboard")
    
    html_path = output_dir / '00a_polish_overview.html'
    create_html_dashboard(ref_df, stats, html_path, logger)
    
    # =====================================================
    # FINAL SUMMARY
    # =====================================================
    print_header(logger, "SUMMARY")
    
    logger.info("✅ Polish dataset overview complete!")
    logger.info("")
    logger.info("Key Findings:")
    logger.info(f"  • {stats['total_observations']:,} observations across {stats['horizon_count']} horizons")
    logger.info(f"  • {stats['total_features']} financial ratios in {len(stats['category_counts'])} categories")
    logger.info(f"  • Severe class imbalance: {stats['bankruptcy_rate']:.2f}% bankruptcy rate")
    logger.info("")
    logger.info("Output Files:")
    logger.info(f"  • {excel_path}")
    logger.info(f"  • {html_path}")
    logger.info(f"  • {ref_csv_path}")
    logger.info("")
    logger.info("Next Steps:")
    logger.info("  1. Review HTML dashboard")
    logger.info("  2. Verify feature mappings")
    logger.info("  3. Run 00b_polish_feature_analysis.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
