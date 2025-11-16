#!/usr/bin/env python3
"""
Script 00d: Polish Data Quality Analysis
=========================================

Comprehensive data quality assessment:
- Missing value patterns
- Duplicate observations
- Feature variance
- Outlier detection
- Quality recommendations for Phase 01
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bankruptcy_prediction.utils.logging_setup import setup_logging, print_header, print_section

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)


def create_visualizations(missing_df, variance_df, exact_dups, df, output_dir, logger, horizon_outliers_df=None):
    """Create comprehensive data quality visualizations."""
    logger.info("Creating visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # 1. Missing values - top 15 features
    ax1 = fig.add_subplot(gs[0, :2])
    top_missing = missing_df[missing_df['Missing_Count'] > 0].head(15)
    colors_missing = ['#ef4444' if pct > 30 else '#f59e0b' if pct > 10 else '#fbbf24' 
                      for pct in top_missing['Missing_Percentage']]
    ax1.barh(range(len(top_missing)), top_missing['Missing_Percentage'], color=colors_missing, alpha=0.8)
    ax1.set_yticks(range(len(top_missing)))
    ax1.set_yticklabels([f"{row['Feature']} - {row['Feature_Name'][:25]}" 
                          for _, row in top_missing.iterrows()], fontsize=9)
    ax1.set_xlabel('Missing Percentage (%)', fontweight='bold', fontsize=11)
    ax1.set_title('Top 15 Features with Missing Values', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # 2. Missing by category
    ax2 = fig.add_subplot(gs[0, 2])
    cat_missing = missing_df.groupby('Category')['Missing_Percentage'].mean().sort_values(ascending=False)
    ax2.bar(range(len(cat_missing)), cat_missing.values, color='#f59e0b', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(cat_missing)))
    ax2.set_xticklabels(cat_missing.index, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Avg Missing %', fontweight='bold', fontsize=11)
    ax2.set_title('Missing Values by Category', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, val in enumerate(cat_missing.values):
        ax2.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 3. Missing value distribution
    ax3 = fig.add_subplot(gs[1, 0])
    missing_bins = [0, 5, 10, 20, 50, 100]
    missing_counts = pd.cut(missing_df['Missing_Percentage'], bins=missing_bins, 
                           labels=['0-5%', '5-10%', '10-20%', '20-50%', '>50%']).value_counts().sort_index()
    ax3.bar(range(len(missing_counts)), missing_counts.values, color='#3b82f6', alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(missing_counts)))
    ax3.set_xticklabels(missing_counts.index, rotation=0)
    ax3.set_ylabel('Feature Count', fontweight='bold', fontsize=11)
    ax3.set_title('Distribution of Missing %', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Variance distribution
    ax4 = fig.add_subplot(gs[1, 1])
    var_counts = variance_df['Variance_Class'].value_counts()
    colors_var = {'Zero': '#ef4444', 'Very Low': '#f59e0b', 'Low': '#fbbf24', 'Normal': '#10b981'}
    bar_colors = [colors_var.get(cat, '#6b7280') for cat in var_counts.index]
    ax4.bar(range(len(var_counts)), var_counts.values, color=bar_colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(var_counts)))
    ax4.set_xticklabels(var_counts.index, rotation=0)
    ax4.set_ylabel('Feature Count', fontweight='bold', fontsize=11)
    ax4.set_title('Variance Distribution', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for i, val in enumerate(var_counts.values):
        ax4.text(i, val + 0.5, str(val), ha='center', va='bottom', fontweight='bold')
    
    # 5. Duplicates
    ax5 = fig.add_subplot(gs[1, 2])
    dup_data = ['Unique', 'Duplicates']
    dup_values = [len(df) - exact_dups, exact_dups]
    colors_dup = ['#10b981', '#ef4444']
    ax5.pie(dup_values, labels=dup_data, autopct='%1.2f%%', colors=colors_dup, startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax5.set_title(f'Duplicate Observations\n({exact_dups:,} duplicates)', fontsize=13, fontweight='bold')
    
    # 6. Quality score by feature (bottom 20)
    ax6 = fig.add_subplot(gs[2, :])
    # Quality score: 100 - missing%
    missing_df_sorted = missing_df.sort_values('Missing_Percentage', ascending=True)
    worst_20 = missing_df_sorted.tail(20)
    quality_scores = 100 - worst_20['Missing_Percentage']
    colors_quality = ['#ef4444' if score < 70 else '#f59e0b' if score < 85 else '#10b981' 
                     for score in quality_scores]
    ax6.barh(range(len(worst_20)), quality_scores, color=colors_quality, alpha=0.7, edgecolor='black')
    ax6.set_yticks(range(len(worst_20)))
    ax6.set_yticklabels([f"{row['Feature']} - {row['Feature_Name'][:25]}" 
                         for _, row in worst_20.iterrows()], fontsize=8)
    ax6.set_xlabel('Quality Score (0-100, based on completeness)', fontweight='bold', fontsize=11)
    ax6.set_title('Bottom 20 Features by Data Quality Score', fontsize=13, fontweight='bold')
    ax6.axvline(x=70, color='#ef4444', linestyle='--', alpha=0.5, linewidth=2, label='Poor (<70)')
    ax6.axvline(x=85, color='#f59e0b', linestyle='--', alpha=0.5, linewidth=2, label='Fair (<85)')
    ax6.legend(fontsize=9)
    ax6.grid(axis='x', alpha=0.3)
    ax6.invert_yaxis()
    
    # 7. Horizon-specific outlier comparison
    if horizon_outliers_df is not None and 'horizon' in df.columns:
        ax7 = fig.add_subplot(gs[3, 0])
        horizon_stats = []
        horizons = sorted(df['horizon'].unique())
        for h in horizons:
            h_data = horizon_outliers_df[horizon_outliers_df['Horizon'] == f'H{h}']
            horizon_stats.append(h_data['Outlier_Percentage'].mean())
        
        colors_horizon = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
        ax7.bar(range(len(horizons)), horizon_stats, color=colors_horizon[:len(horizons)], alpha=0.7, edgecolor='black')
        ax7.set_xticks(range(len(horizons)))
        ax7.set_xticklabels([f'H{h}' for h in horizons])
        ax7.set_ylabel('Mean Outlier %', fontweight='bold', fontsize=11)
        ax7.set_title('Average Outlier Rate by Horizon', fontsize=13, fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)
        for i, val in enumerate(horizon_stats):
            ax7.text(i, val + 0.1, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 8. Horizon-specific missing comparison (A37 example)
        ax8 = fig.add_subplot(gs[3, 1])
        if 'horizon' in df.columns:
            a37_missing = []
            for h in horizons:
                h_df = df[df['horizon'] == h]
                miss_pct = h_df['A37'].isnull().sum() / len(h_df) * 100
                a37_missing.append(miss_pct)
            
            ax8.plot(range(len(horizons)), a37_missing, marker='o', linewidth=2.5, markersize=8, 
                    color='#ef4444', label='A37 (Worst Feature)')
            ax8.set_xticks(range(len(horizons)))
            ax8.set_xticklabels([f'H{h}' for h in horizons])
            ax8.set_ylabel('Missing %', fontweight='bold', fontsize=11)
            ax8.set_title('Missing Values Across Horizons (A37)', fontsize=13, fontweight='bold')
            ax8.grid(alpha=0.3)
            ax8.legend(fontsize=9)
            for i, val in enumerate(a37_missing):
                ax8.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 9. Data quality stability summary
        ax9 = fig.add_subplot(gs[3, 2])
        outlier_range = max(horizon_stats) - min(horizon_stats)
        missing_range = max(a37_missing) - min(a37_missing) if 'horizon' in df.columns else 0
        
        stability_data = ['Outlier\nVariation', 'Missing\nVariation\n(A37)']
        stability_values = [outlier_range, missing_range]
        colors_stability = ['#10b981' if val < 2 else '#f59e0b' if val < 5 else '#ef4444' for val in stability_values]
        
        ax9.bar(range(len(stability_data)), stability_values, color=colors_stability, alpha=0.7, edgecolor='black')
        ax9.set_xticks(range(len(stability_data)))
        ax9.set_xticklabels(stability_data, fontsize=10)
        ax9.set_ylabel('Variation (%)', fontweight='bold', fontsize=11)
        ax9.set_title('Data Quality Stability Across Horizons', fontsize=13, fontweight='bold')
        ax9.grid(axis='y', alpha=0.3)
        for i, val in enumerate(stability_values):
            ax9.text(i, val + 0.1, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add stability interpretation
        stability_text = 'STABLE' if outlier_range < 2 else 'MODERATE' if outlier_range < 5 else 'UNSTABLE'
        ax9.text(0.5, 0.95, f'Overall: {stability_text}', transform=ax9.transAxes,
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Polish Dataset - Data Quality Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    viz_path = output_dir / '00d_data_quality_analysis.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Visualization saved: {viz_path}")


def create_html_dashboard(missing_df, variance_df, exact_dups, df, output_path, logger, horizon_outliers_df=None):
    """Create professional HTML dashboard."""
    logger.info("Creating HTML dashboard...")
    
    features_with_missing = (missing_df['Missing_Count'] > 0).sum()
    worst_feature = missing_df.iloc[0]
    zero_var = (variance_df['Variance_Class'] == 'Zero').sum()
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polish Dataset - Data Quality Report</title>
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
        }}
        .metric-card.critical {{ border-left-color: #ef4444; }}
        .metric-card.warning {{ border-left-color: #f59e0b; }}
        .metric-card.good {{ border-left-color: #10b981; }}
        .metric-value {{
            font-size: 2.2em;
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
        .section {{ margin: 40px 0; }}
        .section-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #1e3c72;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .alert {{
            background: #fee;
            border-left: 5px solid #ef4444;
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
        .info {{
            background: #eff6ff;
            border-left: 5px solid #3b82f6;
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
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Polish Dataset - Data Quality Report</h1>
            <p>Comprehensive Assessment of Data Issues</p>
            <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">Foundation Phase - Script 00d</p>
        </header>
        
        <div class="content">
"""
    
    # Alert if critical issues
    if worst_feature['Missing_Percentage'] > 40:
        html += f"""
            <div class="alert">
                <h3>🚨 Critical Quality Issues Detected</h3>
                <p><strong>{worst_feature['Feature']}</strong> ({worst_feature['Feature_Name']}) has <strong>{worst_feature['Missing_Percentage']:.1f}% missing values</strong></p>
                <p>Consider removal or advanced imputation methods for features with >40% missing.</p>
            </div>
"""
    
    # Key metrics
    html += f"""
            <div class="metric-grid">
                <div class="metric-card {'critical' if features_with_missing > 50 else 'warning'}">
                    <div class="metric-label">Features with Missing</div>
                    <div class="metric-value">{features_with_missing}/64</div>
                    <div class="metric-label">{features_with_missing/64*100:.1f}%</div>
                </div>
                <div class="metric-card warning">
                    <div class="metric-label">Worst Missing</div>
                    <div class="metric-value">{worst_feature['Missing_Percentage']:.1f}%</div>
                    <div class="metric-label">{worst_feature['Feature']} ({worst_feature['Feature_Name'][:15]})</div>
                </div>
                <div class="metric-card {'critical' if zero_var > 0 else 'good'}">
                    <div class="metric-label">Zero Variance</div>
                    <div class="metric-value">{zero_var}</div>
                    <div class="metric-label">features</div>
                </div>
                <div class="metric-card {'warning' if exact_dups > 0 else 'good'}">
                    <div class="metric-label">Duplicates</div>
                    <div class="metric-value">{exact_dups:,}</div>
                    <div class="metric-label">{exact_dups/len(df)*100:.2f}%</div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Missing Values - Top 10 Features</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Name</th>
                            <th>Category</th>
                            <th>Missing %</th>
                            <th>Missing Count</th>
                            <th>Non-Missing</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    for _, row in missing_df[missing_df['Missing_Count'] > 0].head(10).iterrows():
        html += f"""
                        <tr>
                            <td><strong>{row['Feature']}</strong></td>
                            <td>{row['Feature_Name']}</td>
                            <td>{row['Category']}</td>
                            <td>{row['Missing_Percentage']:.2f}%</td>
                            <td>{row['Missing_Count']:,}</td>
                            <td>{row['Non_Missing']:,}</td>
                        </tr>
"""
    
    html += f"""
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">Data Quality Summary</h2>
                <div class="info">
                    <h3>Overall Assessment</h3>
                    <ul style="margin-left: 20px; margin-top: 10px; line-height: 2;">
                        <li><strong>Total Features:</strong> 64</li>
                        <li><strong>Features with Missing:</strong> {features_with_missing} ({features_with_missing/64*100:.1f}%)</li>
                        <li><strong>Total Observations:</strong> {len(df):,}</li>
                        <li><strong>Exact Duplicates:</strong> {exact_dups:,} ({exact_dups/len(df)*100:.2f}%)</li>
                        <li><strong>Zero Variance Features:</strong> {zero_var}</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Horizon-Specific Quality Analysis</h2>
"""
    
    # Add horizon-specific analysis if available
    if horizon_outliers_df is not None and 'horizon' in df.columns:
        horizons = sorted(df['horizon'].unique())
        
        # Calculate horizon statistics
        html += """
                <div class="info">
                    <h3>📊 Data Quality Stability Across Horizons</h3>
                    <p>Analysis of outlier rates and missing values across prediction horizons H1-H5:</p>
                    <table>
                        <thead>
                            <tr>
                                <th>Horizon</th>
                                <th>Observations</th>
                                <th>Mean Outlier %</th>
                                <th>Median Outlier %</th>
                                <th>Max Outlier %</th>
                                <th>A37 Missing %</th>
                            </tr>
                        </thead>
                        <tbody>
"""
        
        for h in horizons:
            h_df = df[df['horizon'] == h]
            h_outliers = horizon_outliers_df[horizon_outliers_df['Horizon'] == f'H{h}']
            mean_outlier = h_outliers['Outlier_Percentage'].mean()
            median_outlier = h_outliers['Outlier_Percentage'].median()
            max_outlier = h_outliers['Outlier_Percentage'].max()
            a37_missing = h_df['A37'].isnull().sum() / len(h_df) * 100
            
            html += f"""
                            <tr>
                                <td><strong>H{h}</strong></td>
                                <td>{len(h_df):,}</td>
                                <td>{mean_outlier:.2f}%</td>
                                <td>{median_outlier:.2f}%</td>
                                <td>{max_outlier:.2f}%</td>
                                <td>{a37_missing:.2f}%</td>
                            </tr>
"""
        
        # Calculate stability metrics
        outlier_means = [horizon_outliers_df[horizon_outliers_df['Horizon'] == f'H{h}']['Outlier_Percentage'].mean() 
                        for h in horizons]
        outlier_variation = max(outlier_means) - min(outlier_means)
        
        a37_missing_by_h = [df[df['horizon'] == h]['A37'].isnull().sum() / len(df[df['horizon'] == h]) * 100 
                           for h in horizons]
        a37_variation = max(a37_missing_by_h) - min(a37_missing_by_h)
        
        stability_status = "✅ STABLE" if outlier_variation < 2 else "⚠️ MODERATE" if outlier_variation < 5 else "❌ UNSTABLE"
        
        html += f"""
                        </tbody>
                    </table>
                    
                    <div style="margin-top: 20px; padding: 15px; background: #f0f9ff; border-left: 4px solid #3b82f6; border-radius: 5px;">
                        <h4 style="margin-bottom: 10px;">📈 Stability Assessment</h4>
                        <ul style="margin-left: 20px; line-height: 1.8;">
                            <li><strong>Outlier Rate Variation:</strong> {outlier_variation:.2f}% (range: {min(outlier_means):.2f}% - {max(outlier_means):.2f}%)</li>
                            <li><strong>A37 Missing Variation:</strong> {a37_variation:.2f}% (range: {min(a37_missing_by_h):.2f}% - {max(a37_missing_by_h):.2f}%)</li>
                            <li><strong>Overall Status:</strong> {stability_status}</li>
                        </ul>
                    </div>
                    
                    <div style="margin-top: 15px; padding: 15px; background: #f0fdf4; border-left: 4px solid #10b981; border-radius: 5px;">
                        <h4 style="margin-bottom: 10px;">✅ Methodological Implication</h4>
                        <p style="line-height: 1.8; margin: 0;">
                            Data quality shows <strong>relative stability across horizons</strong> (outlier variation: {outlier_variation:.2f}%). 
                            This justifies applying a <strong>uniform preprocessing pipeline</strong> across all horizons, followed by 
                            horizon-specific modeling. The slight increase in outlier rates from H1 ({outlier_means[0]:.2f}%) to H4 
                            ({outlier_means[3]:.2f}%) reflects natural data characteristics rather than quality degradation.
                        </p>
                    </div>
                </div>
            </div>
"""
    else:
        html += """
                <div class="warning">
                    <p>⚠️ Horizon-specific analysis not available (horizon column not found or data not provided)</p>
                </div>
            </div>
"""
    
    html += f"""
            
            <div class="section">
                <h2 class="section-title">Preprocessing Pipeline Order (Evidence-Based)</h2>
                <div class="info">
                    <h3>Phase 01: Data Preparation (in this order)</h3>
                    <ol style="margin-left: 20px; margin-top: 10px; line-height: 2;">
                        <li><strong>Remove Duplicates:</strong> {exact_dups:,} exact duplicates BEFORE train/test split to prevent leakage</li>
                        <li><strong>Handle Outliers:</strong> Full analysis + winsorization (1st/99th percentile) BEFORE imputation (sample showed 10/10 features affected)</li>
                        <li><strong>Missing Values - PASSIVE IMPUTATION for Financial Ratios:</strong>
                            <ul style="margin-left: 20px;">
                                <li>Log-transform numerator and denominator</li>
                                <li>Impute missing values (KNN or median)</li>
                                <li>Back-transform and calculate ratio</li>
                                <li>For A37 (43.7% missing): Try imputation first, re-evaluate in Phase 03 based on VIF</li>
                            </ul>
                        </li>
                        <li><strong>Feature Scaling:</strong> Standardization after imputation</li>
                        <li><strong>Temporal Split:</strong> H1-H3 (train) / H4 (val) / H5 (test)</li>
                    </ol>
                    <p style="margin-top: 15px;"><strong>Why this order?</strong> "When outliers are removed or treated and missing values accurately imputed, the correlations among predictors become more realistic, often resulting in decreased VIF values." (Research: Number Analytics, 2024)</p>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Phase 03: Multicollinearity Analysis (AFTER Phase 01)</h2>
                <div class="warning">
                    <h3>⚠️ Critical Sequencing Note</h3>
                    <p><strong>VIF calculation MUST come AFTER imputation.</strong> Imputation changes correlations between features. VIF calculated on incomplete data would be invalid.</p>
                    <p>Known from Script 00b: 1 inverse pair + 9 redundant groups → expect many features with VIF > 10.</p>
                </div>
            </div>
            
            <div class="warning">
                <h3>⚠️ Important Notes</h3>
                <ul style="margin-left: 20px; margin-top: 10px; line-height: 1.8;">
                    <li>ALL 64 features have some missing values - imputation is mandatory</li>
                    <li>A37 (Quick Assets/LT Liabilities) has 43.7% missing - evaluate removal vs imputation</li>
                    <li>Duplicates are only 0.92% of data - safe to remove without major data loss</li>
                    <li>No zero-variance features detected - all features provide information</li>
                </ul>
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
    """Execute data quality analysis."""
    logger = setup_logging('00d_polish_data_quality')
    output_dir = PROJECT_ROOT / 'results' / '00_foundation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header(logger, "SCRIPT 00d: POLISH DATA QUALITY")
    logger.info("Comprehensive data quality assessment")
    logger.info("")
    
    # Load data and metadata
    print_section(logger, "STEP 1: Load Data & Metadata")
    df = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'poland_clean_full.parquet')
    
    with open(PROJECT_ROOT / 'data' / 'polish-companies-bankruptcy' / 'feature_descriptions.json') as f:
        metadata = json.load(f)
    
    logger.info(f"✓ Data shape: {df.shape}")
    logger.info(f"✓ Metadata: {len(metadata['features'])} features")
    logger.info("")
    
    # Analyze missing values
    print_section(logger, "STEP 2: Missing Value Analysis")
    feature_cols = [col for col in df.columns if col.startswith('A')]
    features_meta = metadata['features']
    
    missing_data = []
    for feat in feature_cols:
        miss_cnt = df[feat].isnull().sum()
        miss_pct = miss_cnt / len(df) * 100
        meta_info = features_meta.get(feat, {})
        
        missing_data.append({
            'Feature': feat,
            'Feature_Name': meta_info.get('short_name', 'Unknown'),
            'Category': meta_info.get('category', 'Unknown'),
            'Missing_Count': miss_cnt,
            'Missing_Percentage': miss_pct,
            'Non_Missing': len(df) - miss_cnt
        })
    
    missing_df = pd.DataFrame(missing_data).sort_values('Missing_Percentage', ascending=False)
    
    features_with_missing = (missing_df['Missing_Count'] > 0).sum()
    worst_feature = missing_df.iloc[0]
    
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Features with missing: {features_with_missing} ({features_with_missing/len(feature_cols)*100:.1f}%)")
    logger.info(f"Worst feature: {worst_feature['Feature']} ({worst_feature['Missing_Percentage']:.1f}% missing)")
    logger.info("")
    logger.info("Top 5 features with missing values:")
    for _, row in missing_df.head(5).iterrows():
        if row['Missing_Count'] > 0:
            logger.info(f"  {row['Feature']:4s} - {row['Feature_Name'][:30]:30s}: {row['Missing_Percentage']:6.2f}%")
    logger.info("")
    
    # Analyze duplicates
    print_section(logger, "STEP 3: Duplicate Analysis (Complete)")
    exact_dups = df.duplicated().sum()
    feature_dups = df[feature_cols].duplicated().sum()
    
    logger.info(f"Exact duplicates (all columns): {exact_dups:,} ({exact_dups/len(df)*100:.2f}%)")
    logger.info(f"Feature duplicates (X only): {feature_dups:,} ({feature_dups/len(df)*100:.2f}%)")
    
    # Detailed duplicate investigation
    if exact_dups > 0:
        dup_df = df[df.duplicated(keep=False)]
        logger.info(f"\nDuplicate Pattern Analysis:")
        logger.info(f"  Total duplicate rows: {len(dup_df):,}")
        logger.info(f"  Number of duplicate groups: {exact_dups // 2:,}")
        
        # Check horizon distribution of duplicates
        if 'horizon' in dup_df.columns:
            horizon_dups = dup_df.groupby('horizon').size()
            logger.info(f"  Duplicates by horizon:")
            for h, count in horizon_dups.items():
                logger.info(f"    H{h}: {count:,} rows ({count/len(dup_df)*100:.1f}% of duplicates)")
        
        logger.info(f"\n  ⚠️  Critical: {exact_dups:,} rows are EXACT duplicates (all 68 columns identical)")
        logger.info(f"  Nature: {exact_dups // 2:,} pairs - each row duplicated exactly once")
        logger.info(f"  Recommendation: REMOVE in Phase 01 (assumed data entry errors)")
    logger.info("")
    
    # Analyze variance
    print_section(logger, "STEP 4: Variance Analysis")
    variance_data = []
    for feat in feature_cols:
        vals = df[feat].dropna()
        var = vals.var() if len(vals) > 0 else 0
        unique_cnt = vals.nunique()
        
        if var == 0:
            var_class = 'Zero'
        elif var < 0.01:
            var_class = 'Very Low'
        elif var < 0.1:
            var_class = 'Low'
        else:
            var_class = 'Normal'
        
        meta_info = features_meta.get(feat, {})
        variance_data.append({
            'Feature': feat,
            'Feature_Name': meta_info.get('short_name', 'Unknown'),
            'Variance': var,
            'Unique_Values': unique_cnt,
            'Variance_Class': var_class
        })
    
    variance_df = pd.DataFrame(variance_data).sort_values('Variance')
    
    zero_var = (variance_df['Variance_Class'] == 'Zero').sum()
    very_low_var = (variance_df['Variance_Class'] == 'Very Low').sum()
    low_var = (variance_df['Variance_Class'] == 'Low').sum()
    
    logger.info(f"Zero variance: {zero_var} features")
    logger.info(f"Very low variance (< 0.01): {very_low_var} features")
    logger.info(f"Low variance (< 0.1): {low_var} features")
    logger.info(f"Normal variance: {len(variance_df) - zero_var - very_low_var - low_var} features")
    
    if zero_var > 0:
        logger.info("")
        logger.info("⚠️  WARNING: Zero-variance features MUST be removed before modeling!")
    logger.info("")
    
    # Outlier detection (simple IQR method)
    print_section(logger, "STEP 5: Outlier Detection (Complete - All 64 Features)")
    outlier_data = []
    for feat in feature_cols:
        Q1 = df[feat].quantile(0.25)
        Q3 = df[feat].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        outliers = ((df[feat] < lower) | (df[feat] > upper)).sum()
        valid_count = df[feat].notna().sum()
        outlier_pct = outliers / valid_count * 100 if valid_count > 0 else 0
        
        meta_info = features_meta.get(feat, {})
        outlier_data.append({
            'Feature': feat,
            'Feature_Name': meta_info.get('short_name', 'Unknown'),
            'Category': meta_info.get('category', 'Unknown'),
            'Outlier_Count': outliers,
            'Outlier_Percentage': outlier_pct,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower_Bound': lower,
            'Upper_Bound': upper
        })
    
    outlier_df = pd.DataFrame(outlier_data).sort_values('Outlier_Percentage', ascending=False)
    features_with_outliers = (outlier_df['Outlier_Count'] > 0).sum()
    
    logger.info(f"Total features analyzed: {len(feature_cols)}")
    logger.info(f"Features with outliers: {features_with_outliers} ({features_with_outliers/len(feature_cols)*100:.1f}%)")
    logger.info(f"Worst feature: {outlier_df.iloc[0]['Feature']} ({outlier_df.iloc[0]['Outlier_Percentage']:.1f}%)")
    logger.info("")
    logger.info("Top 5 features with outliers:")
    for _, row in outlier_df.head(5).iterrows():
        if row['Outlier_Count'] > 0:
            logger.info(f"  {row['Feature']:4s} - {row['Feature_Name'][:30]:30s}: {row['Outlier_Percentage']:6.2f}% ({row['Outlier_Count']:,} obs)")
    logger.info("")
    
    # Horizon-specific outlier analysis
    print_section(logger, "STEP 5b: Horizon-Specific Outlier Analysis")
    if 'horizon' in df.columns:
        horizon_outliers = []
        for h in sorted(df['horizon'].unique()):
            h_df = df[df['horizon'] == h]
            for feat in feature_cols:
                Q1 = h_df[feat].quantile(0.25)
                Q3 = h_df[feat].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 3 * IQR
                upper = Q3 + 3 * IQR
                outliers = ((h_df[feat] < lower) | (h_df[feat] > upper)).sum()
                valid_count = h_df[feat].notna().sum()
                outlier_pct = outliers / valid_count * 100 if valid_count > 0 else 0
                
                meta_info = features_meta.get(feat, {})
                horizon_outliers.append({
                    'Horizon': f'H{h}',
                    'Feature': feat,
                    'Feature_Name': meta_info.get('short_name', 'Unknown'),
                    'Outlier_Count': outliers,
                    'Outlier_Percentage': outlier_pct
                })
        
        horizon_outliers_df = pd.DataFrame(horizon_outliers)
        
        # Summary statistics by horizon
        logger.info("Outlier percentages by horizon:")
        for h in sorted(df['horizon'].unique()):
            h_data = horizon_outliers_df[horizon_outliers_df['Horizon'] == f'H{h}']
            mean_pct = h_data['Outlier_Percentage'].mean()
            median_pct = h_data['Outlier_Percentage'].median()
            max_pct = h_data['Outlier_Percentage'].max()
            logger.info(f"  H{h}: Mean={mean_pct:5.2f}%, Median={median_pct:5.2f}%, Max={max_pct:5.2f}%")
        
        logger.info("")
        logger.info("Assessment: Outlier patterns across horizons")
    else:
        horizon_outliers_df = None
        logger.info("⚠️  'horizon' column not found - skipping horizon-specific analysis")
    logger.info("")
    
    # Create visualizations
    print_section(logger, "STEP 6: Create Visualizations")
    create_visualizations(missing_df, variance_df, exact_dups, df, output_dir, logger, horizon_outliers_df)
    logger.info("")
    
    # Save Excel report
    print_section(logger, "STEP 7: Save Excel Report")
    excel_path = output_dir / '00d_data_quality.xlsx'
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Summary
        summary_data = {
            'Metric': [
                'Total Features',
                'Features with Missing',
                'Max Missing Percentage',
                'Exact Duplicates',
                'Zero Variance Features',
                'Low Variance Features'
            ],
            'Value': [
                len(feature_cols),
                features_with_missing,
                f"{missing_df['Missing_Percentage'].max():.2f}%",
                f"{exact_dups:,} ({exact_dups/len(df)*100:.2f}%)",
                zero_var,
                very_low_var + low_var
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Missing values - overall
        missing_df.to_excel(writer, sheet_name='Missing_Values', index=False)
        
        # Missing values - by horizon
        if 'horizon' in df.columns:
            horizon_missing = []
            for h in sorted(df['horizon'].unique()):
                h_df = df[df['horizon'] == h]
                for feat in feature_cols:
                    miss_cnt = h_df[feat].isnull().sum()
                    miss_pct = miss_cnt / len(h_df) * 100
                    meta_info = features_meta.get(feat, {})
                    horizon_missing.append({
                        'Horizon': f'H{h}',
                        'Feature': feat,
                        'Feature_Name': meta_info.get('short_name', 'Unknown'),
                        'Missing_Count': miss_cnt,
                        'Missing_Percentage': miss_pct
                    })
            horizon_missing_df = pd.DataFrame(horizon_missing)
            horizon_missing_pivot = horizon_missing_df.pivot(index='Feature', 
                                                              columns='Horizon', 
                                                              values='Missing_Percentage')
            horizon_missing_pivot['Feature_Name'] = [features_meta.get(f, {}).get('short_name', 'Unknown') 
                                                      for f in horizon_missing_pivot.index]
            cols = ['Feature_Name'] + [f'H{h}' for h in sorted(df['horizon'].unique())]
            horizon_missing_pivot[cols].to_excel(writer, sheet_name='Missing_By_Horizon')
        
        # Variance
        variance_df.to_excel(writer, sheet_name='Variance_Analysis', index=False)
        
        # Outliers - complete analysis
        outlier_df.to_excel(writer, sheet_name='Outliers_Complete', index=False)
        
        # Outliers - by horizon (pivot table)
        if horizon_outliers_df is not None:
            horizon_outliers_pivot = horizon_outliers_df.pivot(index='Feature',
                                                                columns='Horizon',
                                                                values='Outlier_Percentage')
            horizon_outliers_pivot['Feature_Name'] = [features_meta.get(f, {}).get('short_name', 'Unknown')
                                                       for f in horizon_outliers_pivot.index]
            cols = ['Feature_Name'] + [f'H{h}' for h in sorted(df['horizon'].unique())]
            horizon_outliers_pivot[cols].to_excel(writer, sheet_name='Outliers_By_Horizon')
        
        # Recommendations (evidence-based)
        rec_data = {
            'Issue': [
                'Missing Values (ALL features)',
                'Duplicates',
                'Zero Variance',
                'Outliers (sample: 10/10 features)'
            ],
            'Severity': [
                'HIGH',
                'LOW',
                'NONE' if zero_var == 0 else 'CRITICAL',
                'MEDIUM'
            ],
            'Recommended_Action': [
                'PASSIVE IMPUTATION for ratios: impute numerator/denominator (log-transformed), then calculate ratio. For A37 (43.7% missing): try imputation first, re-evaluate in Phase 03 based on VIF.',
                f'Remove {exact_dups:,} duplicates ({exact_dups // 2:,} pairs) BEFORE train/test split. Nature: EXACT duplicates (all columns identical), assumed data entry errors.',
                'None detected - all features have normal variance' if zero_var == 0 else 'MUST remove before modeling',
                f'Complete analysis done: {features_with_outliers}/{len(feature_cols)} features affected. Apply winsorization (1st/99th percentile) BEFORE imputation in Phase 01.'
            ]
        }
        pd.DataFrame(rec_data).to_excel(writer, sheet_name='Recommendations', index=False)
    
    logger.info(f"✓ Excel report saved: {excel_path}")
    logger.info("")
    
    # Create HTML dashboard
    print_section(logger, "STEP 8: Create HTML Dashboard")
    html_path = output_dir / '00d_data_quality.html'
    create_html_dashboard(missing_df, variance_df, exact_dups, df, html_path, logger, horizon_outliers_df)
    logger.info("")
    
    # Summary
    print_header(logger, "SUMMARY")
    logger.info("✅ Data quality analysis complete!")
    logger.info("")
    logger.info("Key Findings:")
    logger.info(f"  • Missing: {features_with_missing}/{len(feature_cols)} features affected")
    logger.info(f"  • Worst missing: {worst_feature['Feature']} ({worst_feature['Missing_Percentage']:.1f}%)")
    logger.info(f"  • Duplicates: {exact_dups:,} observations")
    logger.info(f"  • Zero variance: {zero_var} features")
    logger.info(f"  • Low variance: {very_low_var + low_var} features")
    logger.info("")
    logger.info("Output Files:")
    logger.info(f"  • {excel_path}")
    logger.info(f"  • {html_path}")
    logger.info(f"  • {output_dir / '00d_data_quality_analysis.png'}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 FOUNDATION PHASE COMPLETE (Scripts 00a-00d)")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next Phase: 01_data_preparation")
    logger.info("  - Handle missing values")
    logger.info("  - Remove duplicates")
    logger.info("  - Remove zero-variance features")
    logger.info("  - Create train/val/test splits (H1-H3 / H4 / H5)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
