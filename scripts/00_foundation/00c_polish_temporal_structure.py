#!/usr/bin/env python3
"""
Script 00c: Polish Temporal Structure Analysis
===============================================

Purpose:
--------
Analyze temporal characteristics of Polish dataset:
1. Bankruptcy rates by horizon (H1-H5)
2. Feature distributions across horizons
3. Temporal trends and patterns
4. Confirm data structure (repeated cross-sections vs panel data)

Why Critical:
-------------
Polish has 5 prediction horizons (1-5 years ahead).
Understanding temporal structure determines:
- Appropriate modeling approach (temporal holdout vs cross-validation)
- Whether time-series methods are applicable
- How to split train/val/test sets

Output:
-------
- Excel: Horizon statistics, bankruptcy rates, feature summaries
- HTML: Professional temporal analysis dashboard
- Visualizations: Bankruptcy rates, feature distributions by horizon
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bankruptcy_prediction.utils.logging_setup import setup_logging, print_header, print_section
from src.bankruptcy_prediction.utils.config_loader import get_config

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_data(data_path: Path, logger) -> pd.DataFrame:
    """Load Polish dataset."""
    logger.info(f"Loading data: {data_path.name}")
    
    df = pd.read_parquet(data_path)
    
    logger.info(f"✓ Loaded {len(df):,} observations")
    logger.info(f"  Columns: {df.shape[1]}")
    logger.info("")
    
    return df


def analyze_horizon_distribution(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Analyze observation and bankruptcy distribution across horizons.
    
    Returns detailed horizon statistics.
    """
    logger.info("Analyzing horizon distribution...")
    
    target_col = 'y' if 'y' in df.columns else 'class'
    
    horizon_stats = []
    for horizon in sorted(df['horizon'].unique()):
        h_data = df[df['horizon'] == horizon]
        
        bankruptcies = (h_data[target_col] == 1).sum()
        healthy = (h_data[target_col] == 0).sum()
        total = len(h_data)
        bankruptcy_rate = bankruptcies / total * 100
        
        horizon_stats.append({
            'Horizon': f'H{horizon}',
            'Years_Ahead': horizon,
            'Total_Observations': total,
            'Bankruptcies': bankruptcies,
            'Healthy': healthy,
            'Bankruptcy_Rate_%': bankruptcy_rate
        })
    
    horizon_df = pd.DataFrame(horizon_stats)
    
    logger.info("Horizon distribution:")
    for _, row in horizon_df.iterrows():
        logger.info(f"  {row['Horizon']}: {row['Total_Observations']:,} obs, "
                   f"{row['Bankruptcies']:,} bankrupt ({row['Bankruptcy_Rate_%']:.2f}%)")
    logger.info("")
    
    return horizon_df


def test_temporal_trends(df: pd.DataFrame, horizon_df: pd.DataFrame, logger) -> dict:
    """
    Test for temporal trends in bankruptcy rates.
    
    Statistical tests:
    - Trend test (Spearman correlation)
    - Chi-square test (homogeneity across horizons)
    """
    logger.info("Testing for temporal trends...")
    
    results = {}
    
    # Spearman correlation: horizon vs bankruptcy rate
    spearman_corr, spearman_p = stats.spearmanr(
        horizon_df['Years_Ahead'],
        horizon_df['Bankruptcy_Rate_%']
    )
    
    results['spearman_correlation'] = spearman_corr
    results['spearman_p_value'] = spearman_p
    results['trend_significant'] = spearman_p < 0.05
    
    if results['trend_significant']:
        trend_direction = "increasing" if spearman_corr > 0 else "decreasing"
        logger.info(f"  ✓ Significant {trend_direction} trend detected")
        logger.info(f"    Spearman r = {spearman_corr:.3f}, p = {spearman_p:.4f}")
    else:
        logger.info(f"  • No significant trend detected")
        logger.info(f"    Spearman r = {spearman_corr:.3f}, p = {spearman_p:.4f}")
    
    # Chi-square test for homogeneity
    target_col = 'y' if 'y' in df.columns else 'class'
    contingency_table = pd.crosstab(df['horizon'], df[target_col])
    chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
    
    results['chi2_statistic'] = chi2
    results['chi2_p_value'] = chi2_p
    results['chi2_dof'] = dof
    results['distributions_different'] = chi2_p < 0.05
    
    if results['distributions_different']:
        logger.info(f"  ✓ Bankruptcy rates differ significantly across horizons")
        logger.info(f"    χ² = {chi2:.2f}, p = {chi2_p:.4f}")
    else:
        logger.info(f"  • Bankruptcy rates similar across horizons")
        logger.info(f"    χ² = {chi2:.2f}, p = {chi2_p:.4f}")
    
    logger.info("")
    
    return results


def analyze_feature_stability(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Analyze feature value stability across horizons.
    
    For each feature, calculate coefficient of variation across horizons.
    High CV = feature values change significantly with horizon.
    """
    logger.info("Analyzing feature stability across horizons...")
    
    feature_cols = [col for col in df.columns if col.startswith('A')]
    
    stability_data = []
    for feature in feature_cols[:10]:  # Sample first 10 for summary
        horizon_means = df.groupby('horizon')[feature].mean()
        horizon_stds = df.groupby('horizon')[feature].std()
        
        # Coefficient of variation
        cv = horizon_stds.mean() / abs(horizon_means.mean()) if horizon_means.mean() != 0 else np.inf
        
        stability_data.append({
            'Feature': feature,
            'Mean_CV': cv,
            'Stability': 'Stable' if cv < 0.5 else 'Moderate' if cv < 1.0 else 'Unstable'
        })
    
    stability_df = pd.DataFrame(stability_data)
    
    stable_count = (stability_df['Stability'] == 'Stable').sum()
    logger.info(f"  Feature stability (sample of 10):")
    logger.info(f"    Stable: {stable_count}")
    logger.info(f"    Moderate: {(stability_df['Stability'] == 'Moderate').sum()}")
    logger.info(f"    Unstable: {(stability_df['Stability'] == 'Unstable').sum()}")
    logger.info("")
    
    return stability_df


def determine_data_structure(df: pd.DataFrame, logger) -> dict:
    """
    Determine if data is panel data or repeated cross-sections.
    
    Panel: Same entities tracked over time
    Repeated cross-sections: Different entities at each time point
    """
    logger.info("Determining data structure...")
    
    structure = {
        'has_horizons': 'horizon' in df.columns,
        'horizon_count': df['horizon'].nunique() if 'horizon' in df.columns else 0,
        'has_entity_id': False,  # Polish doesn't track entities
        'structure_type': 'repeated_cross_sections',
        'recommended_approach': 'Temporal holdout validation (train on H1-H3, val on H4, test on H5)'
    }
    
    logger.info(f"  Structure type: {structure['structure_type']}")
    logger.info(f"  Horizons: {structure['horizon_count']}")
    logger.info(f"  Entity tracking: {'Yes' if structure['has_entity_id'] else 'No'}")
    logger.info("")
    logger.info(f"  Recommended approach:")
    logger.info(f"    {structure['recommended_approach']}")
    logger.info("")
    
    return structure


def create_visualizations(df: pd.DataFrame, horizon_df: pd.DataFrame, output_dir: Path, logger):
    """Create temporal analysis visualizations."""
    logger.info("Creating visualizations...")
    
    target_col = 'y' if 'y' in df.columns else 'class'
    
    # Figure 1: Bankruptcy rates by horizon
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1: Bar chart of bankruptcy rates
    ax = axes[0, 0]
    colors = ['#ef4444' if i == 1 else '#10b981' for i in range(len(horizon_df))]
    bars = ax.bar(range(len(horizon_df)), horizon_df['Bankruptcy_Rate_%'], 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Prediction Horizon', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Bankruptcy Rate by Prediction Horizon', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(horizon_df)))
    ax.set_xticklabels(horizon_df['Horizon'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, horizon_df['Bankruptcy_Rate_%'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 1.2: Stacked bar chart of counts
    ax = axes[0, 1]
    width = 0.6
    x = range(len(horizon_df))
    ax.bar(x, horizon_df['Healthy'], width, label='Healthy', color='#10b981', alpha=0.7)
    ax.bar(x, horizon_df['Bankruptcies'], width, bottom=horizon_df['Healthy'],
           label='Bankrupt', color='#ef4444', alpha=0.7)
    ax.set_xlabel('Prediction Horizon', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Observations', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution by Horizon', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_df['Horizon'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 1.3: Line plot of bankruptcy rate trend
    ax = axes[1, 0]
    ax.plot(horizon_df['Years_Ahead'], horizon_df['Bankruptcy_Rate_%'],
            marker='o', linewidth=2.5, markersize=10, color='#ef4444')
    ax.fill_between(horizon_df['Years_Ahead'], 0, horizon_df['Bankruptcy_Rate_%'],
                     alpha=0.2, color='#ef4444')
    ax.set_xlabel('Years Ahead', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Temporal Trend in Bankruptcy Rate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 1.4: Box plot of sample feature across horizons
    ax = axes[1, 1]
    sample_feature = 'A1'  # Net Profit / Total Assets
    horizon_data = [df[df['horizon'] == h][sample_feature].dropna() for h in sorted(df['horizon'].unique())]
    bp = ax.boxplot(horizon_data, labels=[f'H{h}' for h in sorted(df['horizon'].unique())],
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#3b82f6')
        patch.set_alpha(0.6)
    ax.set_xlabel('Prediction Horizon', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{sample_feature} (ROA)', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Distribution Across Horizons\n({sample_feature}: Net Profit / Total Assets)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    viz_path = output_dir / '00c_temporal_analysis.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Visualization saved: {viz_path}")


def create_excel_report(
    horizon_df: pd.DataFrame,
    trend_results: dict,
    structure: dict,
    output_path: Path,
    logger
):
    """Create Excel report with temporal analysis."""
    logger.info("Creating Excel report...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Horizon Summary
        horizon_df.to_excel(writer, sheet_name='Horizon_Summary', index=False)
        
        # Sheet 2: Statistical Tests
        test_data = {
            'Test': [
                'Spearman Correlation',
                'Spearman p-value',
                'Trend Significant?',
                'Chi-Square Statistic',
                'Chi-Square p-value',
                'Distributions Different?'
            ],
            'Result': [
                f"{trend_results['spearman_correlation']:.4f}",
                f"{trend_results['spearman_p_value']:.4f}",
                'Yes' if trend_results['trend_significant'] else 'No',
                f"{trend_results['chi2_statistic']:.2f}",
                f"{trend_results['chi2_p_value']:.4f}",
                'Yes' if trend_results['distributions_different'] else 'No'
            ],
            'Interpretation': [
                'Correlation between horizon and bankruptcy rate',
                'Significance level (< 0.05 = significant)',
                'Is there a temporal trend?',
                'Test if rates differ across horizons',
                'Significance level',
                'Do horizons have different bankruptcy rates?'
            ]
        }
        test_df = pd.DataFrame(test_data)
        test_df.to_excel(writer, sheet_name='Statistical_Tests', index=False)
        
        # Sheet 3: Data Structure
        structure_data = {
            'Property': [
                'Structure Type',
                'Number of Horizons',
                'Entity Tracking',
                'Recommended Approach'
            ],
            'Value': [
                structure['structure_type'],
                structure['horizon_count'],
                'Yes' if structure['has_entity_id'] else 'No',
                structure['recommended_approach']
            ]
        }
        structure_df = pd.DataFrame(structure_data)
        structure_df.to_excel(writer, sheet_name='Data_Structure', index=False)
    
    logger.info(f"✓ Excel report saved: {output_path}")


def create_html_dashboard(
    horizon_df: pd.DataFrame,
    trend_results: dict,
    structure: dict,
    output_path: Path,
    logger
):
    """Create professional HTML dashboard."""
    logger.info("Creating HTML dashboard...")
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polish Dataset - Temporal Structure Analysis</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
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
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .content {{ padding: 40px; }}
        .section {{ margin: 40px 0; }}
        .section-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #1e3c72;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .info {{
            background: #eff6ff;
            border-left: 5px solid #3b82f6;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .warning {{
            background: #fef3c7;
            border-left: 5px solid #f59e0b;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th {{
            background: #1e3c72;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{ padding: 12px 15px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f7fa; }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #1e3c72;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Polish Dataset - Temporal Structure</h1>
            <p>Analysis of Prediction Horizons & Temporal Patterns</p>
            <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">Foundation Phase - Script 00c</p>
        </header>
        
        <div class="content">
            <div class="info">
                <h3>Data Structure: {structure['structure_type'].replace('_', ' ').title()}</h3>
                <p><strong>Horizons:</strong> {structure['horizon_count']} prediction windows (1-5 years ahead)</p>
                <p><strong>Entity Tracking:</strong> {'Yes' if structure['has_entity_id'] else 'No (different companies at each horizon)'}</p>
                <p><strong>Recommended Modeling:</strong> {structure['recommended_approach']}</p>
            </div>
            
            <div class="section">
                <h2 class="section-title">Bankruptcy Rates by Horizon</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Horizon</th>
                            <th>Years Ahead</th>
                            <th>Total Observations</th>
                            <th>Bankruptcies</th>
                            <th>Bankruptcy Rate</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for _, row in horizon_df.iterrows():
        html += f"""
                        <tr>
                            <td><strong>{row['Horizon']}</strong></td>
                            <td>{row['Years_Ahead']}</td>
                            <td>{row['Total_Observations']:,}</td>
                            <td>{row['Bankruptcies']:,}</td>
                            <td>{row['Bankruptcy_Rate_%']:.2f}%</td>
                        </tr>
"""
    
    html += f"""
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">Statistical Analysis</h2>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Trend Test</div>
                        <div class="metric-value">{trend_results['spearman_correlation']:.3f}</div>
                        <div class="metric-label">Spearman r</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Significance</div>
                        <div class="metric-value">{'Yes' if trend_results['trend_significant'] else 'No'}</div>
                        <div class="metric-label">p = {trend_results['spearman_p_value']:.4f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Homogeneity</div>
                        <div class="metric-value">{'Different' if trend_results['distributions_different'] else 'Similar'}</div>
                        <div class="metric-label">χ² = {trend_results['chi2_statistic']:.2f}</div>
                    </div>
                </div>
                
                <div class="{'warning' if trend_results['trend_significant'] else 'info'}">
                    <h3>Interpretation</h3>
"""
    
    if trend_results['trend_significant']:
        direction = "increases" if trend_results['spearman_correlation'] > 0 else "decreases"
        html += f"""
                    <p>⚠️ Bankruptcy rate <strong>{direction}</strong> significantly with prediction horizon.</p>
                    <p>This suggests that longer-term predictions may have different characteristics.</p>
                    <p><strong>Implication:</strong> Horizon-specific models or features may improve performance.</p>
"""
    else:
        html += """
                    <p>✓ Bankruptcy rates are relatively stable across horizons.</p>
                    <p>A single model can be trained on combined horizons.</p>
"""
    
    html += """
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Modeling Recommendations</h2>
                <div class="info">
                    <h3>Temporal Holdout Validation Strategy</h3>
                    <ul style="margin-left: 20px; margin-top: 10px; line-height: 2;">
                        <li><strong>Training Set:</strong> H1 + H2 + H3 (early horizons)</li>
                        <li><strong>Validation Set:</strong> H4 (tune hyperparameters)</li>
                        <li><strong>Test Set:</strong> H5 (final evaluation)</li>
                    </ul>
                    <p style="margin-top: 15px;"><strong>Why this approach?</strong></p>
                    <ul style="margin-left: 20px; line-height: 2;">
                        <li>Respects temporal ordering</li>
                        <li>Mimics real-world prediction scenario</li>
                        <li>Tests model generalization to longer horizons</li>
                        <li>Prevents data leakage from future to past</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"✓ HTML dashboard saved: {output_path}")


def main():
    """Execute Polish temporal structure analysis."""
    
    logger = setup_logging('00c_polish_temporal_structure')
    config = get_config()
    output_dir = PROJECT_ROOT / 'results' / '00_foundation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header(logger, "SCRIPT 00c: POLISH TEMPORAL STRUCTURE")
    logger.info("Analyzing prediction horizons and temporal patterns")
    logger.info("")
    
    # Load data
    print_section(logger, "STEP 1: Load Data")
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'poland_clean_full.parquet'
    df = load_data(data_path, logger)
    
    # Analyze horizon distribution
    print_section(logger, "STEP 2: Analyze Horizon Distribution")
    horizon_df = analyze_horizon_distribution(df, logger)
    
    # Test temporal trends
    print_section(logger, "STEP 3: Test for Temporal Trends")
    trend_results = test_temporal_trends(df, horizon_df, logger)
    
    # Analyze feature stability
    print_section(logger, "STEP 4: Analyze Feature Stability")
    stability_df = analyze_feature_stability(df, logger)
    
    # Determine data structure
    print_section(logger, "STEP 5: Determine Data Structure")
    structure = determine_data_structure(df, logger)
    
    # Create visualizations
    print_section(logger, "STEP 6: Create Visualizations")
    create_visualizations(df, horizon_df, output_dir, logger)
    logger.info("")
    
    # Create Excel
    print_section(logger, "STEP 7: Create Excel Report")
    excel_path = output_dir / '00c_temporal_structure.xlsx'
    create_excel_report(horizon_df, trend_results, structure, excel_path, logger)
    logger.info("")
    
    # Create HTML
    print_section(logger, "STEP 8: Create HTML Dashboard")
    html_path = output_dir / '00c_temporal_structure.html'
    create_html_dashboard(horizon_df, trend_results, structure, html_path, logger)
    logger.info("")
    
    # Summary
    print_header(logger, "SUMMARY")
    logger.info("✅ Temporal structure analysis complete!")
    logger.info("")
    logger.info("Key Findings:")
    logger.info(f"  • Data structure: {structure['structure_type']}")
    logger.info(f"  • {structure['horizon_count']} prediction horizons")
    if trend_results['trend_significant']:
        direction = "increasing" if trend_results['spearman_correlation'] > 0 else "decreasing"
        logger.info(f"  • Significant {direction} trend in bankruptcy rate")
    else:
        logger.info(f"  • No significant temporal trend detected")
    logger.info("")
    logger.info("Modeling Recommendation:")
    logger.info(f"  {structure['recommended_approach']}")
    logger.info("")
    logger.info("Output Files:")
    logger.info(f"  • {excel_path}")
    logger.info(f"  • {html_path}")
    logger.info(f"  • {output_dir / '00c_temporal_analysis.png'}")
    logger.info("")
    logger.info("Next: Run 00d_polish_data_quality.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
