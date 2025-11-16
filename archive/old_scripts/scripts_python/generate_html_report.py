#!/usr/bin/env python3
"""Generate HTML Report from all script outputs"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
output_dir = project_root / 'results' / 'script_outputs'
html_file = project_root / 'results' / 'ANALYSIS_REPORT.html'

print("Generating HTML Report...")

html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Bankruptcy Prediction - Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            background-color: #ecf0f1;
            padding: 10px;
            border-left: 4px solid #3498db;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            min-width: 200px;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .timestamp {{
            color: #95a5a6;
            font-style: italic;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>

<h1>🎯 Bankruptcy Prediction Analysis Report</h1>
<p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

"""

# Script 01: Data Understanding
script01_dir = output_dir / '01_data_understanding'
if script01_dir.exists():
    html += """
<div class="section">
    <h2>📊 01. Data Understanding</h2>
"""
    
    # Dataset info
    info_file = script01_dir / 'dataset_info.csv'
    if info_file.exists():
        info = pd.read_csv(info_file).iloc[0]
        html += f"""
    <div class="metric">
        <div class="metric-label">Total Samples</div>
        <div class="metric-value">{int(info['n_samples']):,}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Features</div>
        <div class="metric-value">{int(info['n_features'])}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Bankruptcy Rate</div>
        <div class="metric-value">{info['bankruptcy_rate']*100:.2f}%</div>
    </div>
"""
    
    # Horizon stats
    horizon_file = script01_dir / 'horizon_statistics.csv'
    if horizon_file.exists():
        horizons = pd.read_csv(horizon_file)
        html += "<h3>Class Distribution by Horizon</h3>"
        html += horizons.to_html(index=False, classes='')
    
    # Figures
    fig1 = script01_dir / 'figures' / 'class_distribution_by_horizon.png'
    fig2 = script01_dir / 'figures' / 'features_by_category.png'
    if fig1.exists():
        html += f'<h3>Visualizations</h3><div class="grid"><img src="script_outputs/01_data_understanding/figures/class_distribution_by_horizon.png">'
    if fig2.exists():
        html += f'<img src="script_outputs/01_data_understanding/figures/features_by_category.png"></div>'
    
    html += "</div>"

# Script 02: Exploratory Analysis
script02_dir = output_dir / '02_exploratory_analysis'
if script02_dir.exists():
    html += """
<div class="section">
    <h2>🔍 02. Exploratory Analysis</h2>
"""
    
    # Discriminative power
    disc_file = script02_dir / 'discriminative_power.csv'
    if disc_file.exists():
        disc = pd.read_csv(disc_file).head(10)
        html += "<h3>Top 10 Most Discriminative Features</h3>"
        html += disc[['readable_name', 'category', 'correlation', 'p_value']].to_html(index=False, classes='')
    
    # Figures
    fig1 = script02_dir / 'figures' / 'correlation_heatmap.png'
    fig2 = script02_dir / 'figures' / 'discriminative_power.png'
    if fig1.exists() or fig2.exists():
        html += '<h3>Visualizations</h3><div class="grid">'
        if fig1.exists():
            html += '<img src="script_outputs/02_exploratory_analysis/figures/correlation_heatmap.png">'
        if fig2.exists():
            html += '<img src="script_outputs/02_exploratory_analysis/figures/discriminative_power.png">'
        html += '</div>'
    
    html += "</div>"

# Script 04: Baseline Models
script04_dir = output_dir / '04_baseline_models'
if script04_dir.exists():
    html += """
<div class="section">
    <h2>🤖 04. Baseline Models</h2>
"""
    
    results_file = script04_dir / 'baseline_results.csv'
    if results_file.exists():
        results = pd.read_csv(results_file)
        html += "<h3>Model Performance</h3>"
        
        for _, row in results.iterrows():
            html += f"""
    <div class="metric">
        <div class="metric-label">{row['model_name']} - ROC-AUC</div>
        <div class="metric-value">{row['roc_auc']:.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">{row['model_name']} - Recall@1%FPR</div>
        <div class="metric-value">{row['recall_1pct_fpr']*100:.1f}%</div>
    </div>
"""
        
        html += "<h3>Detailed Results</h3>"
        html += results.to_html(index=False, classes='', float_format='%.4f')
    
    html += "</div>"

# Script 06: Model Calibration
script06_dir = output_dir / '06_model_calibration'
if script06_dir.exists():
    html += """
<div class="section">
    <h2>🎯 06. Model Calibration</h2>
"""
    
    cal_summary_file = script06_dir / 'calibration_summary.json'
    if cal_summary_file.exists():
        with open(cal_summary_file) as f:
            cal_summary = json.load(f)
        
        html += "<h3>Calibration Improvements</h3>"
        html += f"""
    <div class="metric">
        <div class="metric-label">RF - Before Calibration</div>
        <div class="metric-value">{cal_summary['rf_brier_before']:.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">RF - After Calibration</div>
        <div class="metric-value">{cal_summary['rf_brier_after']:.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Logistic - Before Calibration</div>
        <div class="metric-value">{cal_summary['logit_brier_before']:.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Logistic - After Calibration</div>
        <div class="metric-value">{cal_summary['logit_brier_after']:.4f}</div>
    </div>
"""
    
    # Figure
    cal_fig = script06_dir / 'figures' / 'calibration_comparison.png'
    if cal_fig.exists():
        html += '<h3>Calibration Curves</h3>'
        html += '<img src="script_outputs/06_model_calibration/figures/calibration_comparison.png" style="max-width: 100%;">'
    
    html += "</div>"

# Script 07: Robustness Analysis
script07_dir = output_dir / '07_robustness_analysis'
if script07_dir.exists():
    html += """
<div class="section">
    <h2>🔄 07. Cross-Horizon Robustness</h2>
"""
    
    rob_summary_file = script07_dir / 'robustness_summary.json'
    if rob_summary_file.exists():
        with open(rob_summary_file) as f:
            rob_summary = json.load(f)
        
        html += "<h3>Performance Summary</h3>"
        html += f"""
    <div class="metric">
        <div class="metric-label">Same-Horizon Avg AUC</div>
        <div class="metric-value">{rob_summary['avg_same_horizon_auc']:.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Cross-Horizon Avg AUC</div>
        <div class="metric-value">{rob_summary['avg_cross_horizon_auc']:.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Avg Performance Drop</div>
        <div class="metric-value">{rob_summary['avg_degradation_pct']:.2f}%</div>
    </div>
    <div class="metric">
        <div class="metric-label">Horizons Tested</div>
        <div class="metric-value">{rob_summary['horizons_tested']}</div>
    </div>
"""
    
    # Cross-horizon results table
    cross_results_file = script07_dir / 'cross_horizon_results.csv'
    if cross_results_file.exists():
        cross_results = pd.read_csv(cross_results_file)
        # Create pivot table for better visualization
        pivot = cross_results.pivot(index='train_horizon', columns='test_horizon', values='roc_auc')
        html += "<h3>Cross-Horizon Performance Matrix (ROC-AUC)</h3>"
        html += pivot.to_html(classes='', float_format='%.4f')
    
    # Heatmap
    rob_fig = script07_dir / 'figures' / 'cross_horizon_heatmap.png'
    if rob_fig.exists():
        html += '<h3>Performance Heatmap</h3>'
        html += '<img src="script_outputs/07_robustness_analysis/figures/cross_horizon_heatmap.png" style="max-width: 100%;">'
    
    html += "</div>"

# Script 08: Econometric Analysis
script08_dir = output_dir / '08_econometric_analysis'
if script08_dir.exists():
    html += """
<div class="section">
    <h2>📊 08. Econometric Analysis & Interpretability</h2>
"""
    
    econ_summary_file = script08_dir / 'econometric_summary.json'
    if econ_summary_file.exists():
        with open(econ_summary_file) as f:
            econ_summary = json.load(f)
        
        html += "<h3>Statistical Diagnostics</h3>"
        html += f"""
    <div class="metric">
        <div class="metric-label">VIF - High Multicollinearity</div>
        <div class="metric-value">{econ_summary['vif_high_count']} features</div>
    </div>
    <div class="metric">
        <div class="metric-label">VIF - Low Multicollinearity</div>
        <div class="metric-value">{econ_summary['vif_low_count']} features</div>
    </div>
    <div class="metric">
        <div class="metric-label">Top Protective Factor</div>
        <div class="metric-value">{econ_summary['top_protective_feature']}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Top SHAP Feature</div>
        <div class="metric-value">{econ_summary['top_shap_feature']}</div>
    </div>
"""
    
    # VIF analysis
    vif_file = script08_dir / 'vif_analysis.csv'
    if vif_file.exists():
        vif_df = pd.read_csv(vif_file).head(10)
        html += "<h3>Top 10 Features by VIF (Multicollinearity)</h3>"
        html += vif_df[['readable_name', 'category', 'vif']].to_html(index=False, classes='', float_format='%.2f')
    
    # Logistic coefficients
    coef_file = script08_dir / 'logistic_coefficients.csv'
    if coef_file.exists():
        coef_df = pd.read_csv(coef_file).head(10)
        html += "<h3>Top 10 Logistic Regression Coefficients</h3>"
        html += coef_df[['readable_name', 'category', 'coefficient', 'direction']].to_html(index=False, classes='', float_format='%.4f')
    
    # SHAP importance
    shap_file = script08_dir / 'shap_importance.csv'
    if shap_file.exists():
        shap_df = pd.read_csv(shap_file).head(10)
        html += "<h3>Top 10 SHAP Feature Importance</h3>"
        html += shap_df[['readable_name', 'mean_abs_shap']].to_html(index=False, classes='', float_format='%.4f')
    
    # Figures
    vif_fig = script08_dir / 'figures' / 'vif_analysis.png'
    coef_fig = script08_dir / 'figures' / 'logistic_coefficients.png'
    shap_fig = script08_dir / 'figures' / 'shap_importance.png'
    
    if vif_fig.exists() or coef_fig.exists() or shap_fig.exists():
        html += '<h3>Visualizations</h3><div class="grid">'
        if vif_fig.exists():
            html += '<img src="script_outputs/08_econometric_analysis/figures/vif_analysis.png" style="max-width: 100%;">'
        if coef_fig.exists():
            html += '<img src="script_outputs/08_econometric_analysis/figures/logistic_coefficients.png" style="max-width: 100%;">'
        if shap_fig.exists():
            html += '<img src="script_outputs/08_econometric_analysis/figures/shap_importance.png" style="max-width: 100%;">'
        html += '</div>'
    
    html += "</div>"

# All results from ResultsCollector
all_results_file = project_root / 'results' / 'models' / 'all_results.csv'
if all_results_file.exists():
    html += """
<div class="section">
    <h2>📈 All Model Results</h2>
"""
    all_results = pd.read_csv(all_results_file)
    html += all_results.to_html(index=False, classes='', float_format='%.4f')
    html += "</div>"

html += """
<div class="section">
    <h2>✅ Summary</h2>
    <p class="success">✓ Analysis scripts executed successfully</p>
    <p class="success">✓ Models trained and evaluated</p>
    <p class="success">✓ Results saved and visualized</p>
    
    <h3>Next Steps:</h3>
    <ul>
        <li>Review model performance metrics</li>
        <li>Examine feature importance</li>
        <li>Analyze cross-horizon robustness</li>
        <li>Prepare thesis chapters</li>
    </ul>
</div>

</body>
</html>
"""

# Write HTML file
html_file.write_text(html)
print(f"✓ HTML Report generated: {html_file}")
print(f"\nOpen in browser: file://{html_file}")
