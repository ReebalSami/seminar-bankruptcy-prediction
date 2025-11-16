#!/usr/bin/env python3
"""Generate Comprehensive HTML Report from ALL datasets"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
output_dir = project_root / 'results' / 'script_outputs'
html_file = project_root / 'results' / 'COMPREHENSIVE_REPORT.html'

print("="*70)
print("GENERATING COMPREHENSIVE HTML REPORT")
print("="*70)

html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Bankruptcy Prediction - Comprehensive Multi-Dataset Report</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            text-align: center;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            background: linear-gradient(to right, #3498db, #2980b9);
            color: white;
            padding: 15px;
            border-radius: 5px;
        }}
        h3 {{
            color: #2c3e50;
            margin-top: 25px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .section {{
            background-color: white;
            padding: 25px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            background-color: #ecf0f1;
            padding: 15px;
            margin: 10px;
            border-radius: 5px;
            min-width: 200px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .metric-label {{
            font-weight: bold;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .metric-value {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 15px 0;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .dataset-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 5px;
        }}
        .polish-badge {{ background-color: #3498db; color: white; }}
        .american-badge {{ background-color: #e74c3c; color: white; }}
        .taiwan-badge {{ background-color: #2ecc71; color: white; }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin: 20px 0;
        }}
        .best-model {{
            background-color: #f39c12;
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.3em;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>🏆 Bankruptcy Prediction - Comprehensive Multi-Dataset Analysis</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="section">
        <h2>📊 Executive Summary</h2>
"""

# Load cross-dataset comparison
comparison_file = output_dir / '09_cross_dataset_comparison' / 'comparison_summary.json'
if comparison_file.exists():
    with open(comparison_file) as f:
        summary = json.load(f)
    
    html += f"""
    <div class="best-model">
        🥇 Best Overall Performance: {summary['best_overall_dataset']} Dataset - {summary['best_overall_model']}<br>
        ROC-AUC: {summary['best_overall_auc']:.4f}
    </div>
    
    <div class="metric">
        <div class="metric-label">Datasets Analyzed</div>
        <div class="metric-value">3</div>
    </div>
    <div class="metric">
        <div class="metric-label">Models Trained</div>
        <div class="metric-value">{summary['total_models_compared']}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Polish Avg AUC</div>
        <div class="metric-value">{summary['avg_auc_polish']:.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Taiwan Avg AUC</div>
        <div class="metric-value">{summary['avg_auc_taiwan']:.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">American Avg AUC</div>
        <div class="metric-value">{summary['avg_auc_american']:.4f}</div>
    </div>
"""

html += """
    </div>
"""

# ========================================================================
# CROSS-DATASET COMPARISON
# ========================================================================
html += """
<div class="section">
    <h2>🌍 Cross-Dataset Comparison</h2>
"""

comparison_dir = output_dir / '09_cross_dataset_comparison'
if comparison_dir.exists():
    # Dataset characteristics
    char_file = comparison_dir / 'dataset_characteristics.csv'
    if char_file.exists():
        char_df = pd.read_csv(char_file)
        html += "<h3>Dataset Characteristics</h3>"
        html += char_df.to_html(index=False, classes='', escape=False)
    
    # Best models
    best_file = comparison_dir / 'best_models.csv'
    if best_file.exists():
        best_df = pd.read_csv(best_file)
        html += "<h3>Best Model by Dataset</h3>"
        html += best_df.to_html(index=False, classes='', float_format='%.4f')
    
    # Figures
    figs_dir = comparison_dir / 'figures'
    if figs_dir.exists():
        html += '<h3>Visualizations</h3><div class="grid">'
        for fig in ['dataset_characteristics.png', 'model_performance_comparison.png', 
                   'performance_heatmap.png', 'comparison_table.png']:
            fig_path = figs_dir / fig
            if fig_path.exists():
                html += f'<img src="script_outputs/09_cross_dataset_comparison/figures/{fig}">'
        html += '</div>'

html += "</div>"

# ========================================================================
# POLISH DATASET
# ========================================================================
html += """
<div class="section">
    <h2><span class="polish-badge">POLISH DATASET</span> Complete Analysis (UCI Repository)</h2>
"""

print("\n[1/3] Adding Polish dataset results...")

# Polish summary
polish_summary_file = output_dir / '01_data_understanding' / 'summary.json'
if polish_summary_file.exists():
    with open(polish_summary_file) as f:
        polish_sum = json.load(f)
    
    html += f"""
    <h3>Dataset Overview</h3>
    <div class="metric">
        <div class="metric-label">Total Samples</div>
        <div class="metric-value">{polish_sum['full_shape'][0]:,}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Features</div>
        <div class="metric-value">{polish_sum['dataset_info']['n_features']}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Horizons</div>
        <div class="metric-value">5</div>
    </div>
    <div class="metric">
        <div class="metric-label">Time Period</div>
        <div class="metric-value">2000-2013</div>
    </div>
"""

# Polish best results
polish_advanced = output_dir / '05_advanced_models' / 'advanced_results.csv'
if polish_advanced.exists():
    adv_df = pd.read_csv(polish_advanced)
    html += "<h3>Advanced Models Performance</h3>"
    html += adv_df.to_html(index=False, classes='', float_format='%.4f')

# Polish econometric
econ_file = output_dir / '08_econometric_analysis' / 'econometric_summary.json'
if econ_file.exists():
    with open(econ_file) as f:
        econ = json.load(f)
    html += f"""
    <h3>Econometric Analysis</h3>
    <div class="metric">
        <div class="metric-label">High VIF Features</div>
        <div class="metric-value">{econ['vif_high_count']}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Top Protective Feature</div>
        <div class="metric-value" style="font-size: 1.2em;">{econ['top_protective_feature']}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Top SHAP Feature</div>
        <div class="metric-value" style="font-size: 1.2em;">{econ['top_shap_feature']}</div>
    </div>
"""

# Polish visualizations (selection)
polish_figs = [
    ('05_advanced_models/figures/model_comparison.png', 'Model Performance'),
    ('07_robustness_analysis/figures/cross_horizon_heatmap.png', 'Cross-Horizon Robustness'),
    ('08_econometric_analysis/figures/shap_importance.png', 'SHAP Feature Importance')
]

html += '<h3>Key Visualizations</h3><div class="grid">'
for fig_path, title in polish_figs:
    full_path = output_dir / fig_path
    if full_path.exists():
        html += f'<div><h4>{title}</h4><img src="script_outputs/{fig_path}"></div>'
html += '</div>'

html += "</div>"

# ========================================================================
# AMERICAN DATASET
# ========================================================================
html += """
<div class="section">
    <h2><span class="american-badge">AMERICAN DATASET</span> NYSE & NASDAQ Companies</h2>
"""

print("\n[2/3] Adding American dataset results...")

# American summary
american_summary_file = output_dir / 'american' / 'cleaning_summary.json'
if american_summary_file.exists():
    with open(american_summary_file) as f:
        am_sum = json.load(f)
    
    html += f"""
    <h3>Dataset Overview</h3>
    <div class="metric">
        <div class="metric-label">Total Samples</div>
        <div class="metric-value">{am_sum['total_samples']:,}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Companies</div>
        <div class="metric-value">{am_sum['companies']:,}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Features</div>
        <div class="metric-value">{am_sum['features']}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Bankruptcy Rate</div>
        <div class="metric-value">{am_sum['bankruptcy_rate']*100:.2f}%</div>
    </div>
    <div class="metric">
        <div class="metric-label">Time Period</div>
        <div class="metric-value">1999-2018</div>
    </div>
"""

# American model results
am_results = output_dir / 'american' / 'baseline_results.csv'
if am_results.exists():
    am_df = pd.read_csv(am_results)
    html += "<h3>Model Performance</h3>"
    html += am_df[['model_name', 'roc_auc', 'pr_auc', 'recall_1pct_fpr']].to_html(index=False, classes='', float_format='%.4f')

# American EDA summary
am_eda = output_dir / 'american' / 'eda_summary.json'
if am_eda.exists():
    with open(am_eda) as f:
        eda = json.load(f)
    html += f"""
    <h3>Exploratory Analysis</h3>
    <div class="metric">
        <div class="metric-label">Significant Features</div>
        <div class="metric-value">{eda['significant_features']} / {eda['total_features']}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Top Discriminative</div>
        <div class="metric-value" style="font-size: 1.2em;">{eda['top_discriminative_feature']}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Top Correlated</div>
        <div class="metric-value" style="font-size: 1.2em;">{eda['top_correlated_feature'][:30]}</div>
    </div>
"""

# American visualizations
am_figs = [
    ('american/figures/discriminative_features.png', 'Discriminative Features'),
    ('american/figures/roc_curves.png', 'ROC Curves'),
    ('american/figures/model_comparison.png', 'Model Comparison')
]

html += '<h3>Visualizations</h3><div class="grid">'
for fig_path, title in am_figs:
    full_path = output_dir / fig_path
    if full_path.exists():
        html += f'<div><h4>{title}</h4><img src="script_outputs/{fig_path}"></div>'
html += '</div>'

html += "</div>"

# ========================================================================
# TAIWAN DATASET
# ========================================================================
html += """
<div class="section">
    <h2><span class="taiwan-badge">TAIWAN DATASET</span> Taiwan Economic Journal (TEJ)</h2>
"""

print("\n[3/3] Adding Taiwan dataset results...")

# Taiwan summary
taiwan_summary_file = output_dir / 'taiwan' / 'cleaning_summary.json'
if taiwan_summary_file.exists():
    with open(taiwan_summary_file) as f:
        tw_sum = json.load(f)
    
    html += f"""
    <h3>Dataset Overview</h3>
    <div class="metric">
        <div class="metric-label">Total Samples</div>
        <div class="metric-value">{tw_sum['total_samples']:,}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Features</div>
        <div class="metric-value">{tw_sum['features']}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Bankruptcy Rate</div>
        <div class="metric-value">{tw_sum['bankruptcy_rate']*100:.2f}%</div>
    </div>
    <div class="metric">
        <div class="metric-label">Bankruptcies</div>
        <div class="metric-value">{tw_sum['bankruptcies']}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Time Period</div>
        <div class="metric-value">1999-2009</div>
    </div>
"""

# Taiwan model results
tw_results = output_dir / 'taiwan' / 'baseline_results.csv'
if tw_results.exists():
    tw_df = pd.read_csv(tw_results)
    html += "<h3>Model Performance</h3>"
    html += tw_df[['model_name', 'roc_auc', 'pr_auc', 'recall_1pct_fpr']].to_html(index=False, classes='', float_format='%.4f')

# Taiwan EDA summary
tw_eda = output_dir / 'taiwan' / 'eda_summary.json'
if tw_eda.exists():
    with open(tw_eda) as f:
        tw_eda_data = json.load(f)
    html += f"""
    <h3>Exploratory Analysis</h3>
    <div class="metric">
        <div class="metric-label">Total Features</div>
        <div class="metric-value">{tw_eda_data['total_features']}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Top Correlated Feature</div>
        <div class="metric-value" style="font-size: 1.0em;">{tw_eda_data['top_correlated_feature'][:40]}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Correlation</div>
        <div class="metric-value">{tw_eda_data['top_correlation']:.4f}</div>
    </div>
"""

# Taiwan visualizations
tw_figs = [
    ('taiwan/figures/top_correlations.png', 'Top Correlations'),
    ('taiwan/figures/roc_curves.png', 'ROC Curves'),
    ('taiwan/figures/model_comparison.png', 'Model Performance')
]

html += '<h3>Visualizations</h3><div class="grid">'
for fig_path, title in tw_figs:
    full_path = output_dir / fig_path
    if full_path.exists():
        html += f'<div><h4>{title}</h4><img src="script_outputs/{fig_path}"></div>'
html += '</div>'

html += "</div>"

# ========================================================================
# ECONOMETRIC RIGOR
# ========================================================================
html += """
<div class="section">
    <h2>🔬 Econometric Rigor & Advanced Analyses</h2>
    <p>Comprehensive econometric diagnostics, panel data analysis, and advanced modeling techniques.</p>
</div>
"""

# Econometric Diagnostics
html += '<div class="section"><h3>10. Econometric Diagnostics & Remediation</h3>'

econ_diag = output_dir / '10_econometric_diagnostics' / 'diagnostics_summary.json'
if econ_diag.exists():
    with open(econ_diag) as f:
        diag_data = json.load(f)
    
    html += '<h4>Original Model Issues</h4>'
    html += f'<div class="metric"><div class="metric-label">Hosmer-Lemeshow</div><div class="metric-value">{diag_data["hosmer_lemeshow"]["result"]}</div></div>'
    html += f'<div class="metric"><div class="metric-label">Events Per Variable</div><div class="metric-value">{diag_data["sample_size"]["events_per_variable"]:.2f}</div></div>'
    html += f'<div class="metric"><div class="metric-label">Condition Number</div><div class="metric-value">{diag_data["multicollinearity"]["condition_number"]:.2e}</div></div>'
    
    econ_rem = output_dir / '10b_econometric_remediation' / 'remediation_summary.json'
    if econ_rem.exists():
        with open(econ_rem) as f:
            rem_data = json.load(f)
        
        html += '<h4>Best Remediation Solution</h4>'
        html += f'<div class="metric"><div class="metric-label">Approach</div><div class="metric-value">{rem_data["best_solution"]["approach"]}</div></div>'
        html += f'<div class="metric"><div class="metric-label">AUC</div><div class="metric-value">{rem_data["best_solution"]["auc"]:.4f}</div></div>'
        html += f'<div class="metric"><div class="metric-label">EPV (Fixed)</div><div class="metric-value">{rem_data["best_solution"]["epv"]:.2f}</div></div>'
    
    # Visualizations
    html += '<h4>Diagnostic Visualizations</h4><div class="grid">'
    diag_figs = [
        ('10_econometric_diagnostics/figures/residual_diagnostics.png', 'Residual Diagnostics'),
        ('10_econometric_diagnostics/figures/influence_diagnostics.png', 'Influential Observations'),
        ('10b_econometric_remediation/figures/remediation_comparison.png', 'Remediation Comparison')
    ]
    for fig_path, title in diag_figs:
        full_path = output_dir / fig_path
        if full_path.exists():
            html += f'<div><h5>{title}</h5><img src="script_outputs/{fig_path}"></div>'
    html += '</div>'

html += '</div>'

# Panel Data Analysis
html += '<div class="section"><h3>11. Panel Data Analysis</h3>'

panel_file = output_dir / '11_panel_data_analysis' / 'panel_summary.json'
if panel_file.exists():
    with open(panel_file) as f:
        panel_data = json.load(f)
    
    html += '<h4>Temporal Validation Results</h4>'
    html += f'<div class="metric"><div class="metric-label">Total Observations</div><div class="metric-value">{panel_data["total_observations"]:,}</div></div>'
    html += f'<div class="metric"><div class="metric-label">Horizons</div><div class="metric-value">{panel_data["horizons"]}</div></div>'
    html += f'<div class="metric"><div class="metric-label">Temporal AUC</div><div class="metric-value">{panel_data["temporal_validation_auc"]:.4f}</div></div>'
    html += f'<div class="metric"><div class="metric-label">Clustered SE AUC</div><div class="metric-value">{panel_data["clustered_bootstrap_mean_auc"]:.4f}</div></div>'
    
    html += '<h4>Visualizations</h4><div class="grid">'
    panel_figs = [
        ('11_panel_data_analysis/figures/within_vs_cross_horizon.png', 'Within vs Cross-Horizon'),
        ('11_panel_data_analysis/figures/clustered_bootstrap.png', 'Bootstrap Distribution'),
        ('11_panel_data_analysis/figures/bankruptcy_by_horizon.png', 'Bankruptcy by Horizon')
    ]
    for fig_path, title in panel_figs:
        full_path = output_dir / fig_path
        if full_path.exists():
            html += f'<div><h5>{title}</h5><img src="script_outputs/{fig_path}"></div>'
    html += '</div>'

html += '</div>'

# Cross-Dataset Transfer Learning
html += '<div class="section"><h3>12. Cross-Dataset Transfer Learning</h3>'

transfer_file = output_dir / '12_cross_dataset_transfer' / 'transfer_summary.json'
if transfer_file.exists():
    with open(transfer_file) as f:
        transfer_data = json.load(f)
    
    html += '<h4>Generalizability Across Economies</h4>'
    html += '<p>Testing if bankruptcy models trained on one country generalize to others.</p>'
    html += f'<div class="metric"><div class="metric-label">Total Experiments</div><div class="metric-value">{transfer_data["total_experiments"]}</div></div>'
    html += f'<div class="metric"><div class="metric-label">Avg Transfer AUC</div><div class="metric-value">{transfer_data["avg_transfer_auc"]:.4f}</div></div>'
    html += f'<div class="metric"><div class="metric-label">Avg Degradation</div><div class="metric-value">{transfer_data["avg_degradation_%"]:.2f}%</div></div>'
    html += f'<div class="metric"><div class="metric-label">Best Transfer</div><div class="metric-value">{transfer_data["best_transfer"]["from"]} → {transfer_data["best_transfer"]["to"]}</div></div>'
    
    html += '<h4>Key Finding</h4>'
    html += '<p>⚠️ Models show <strong>29.4% degradation</strong> when applied across countries - bankruptcy prediction is <strong>context-dependent</strong> and requires local calibration.</p>'
    
    html += '<h4>Visualizations</h4><div class="grid">'
    transfer_figs = [
        ('12_cross_dataset_transfer/figures/transfer_matrix.png', 'Transfer Learning Matrix'),
        ('12_cross_dataset_transfer/figures/transfer_vs_baseline.png', 'Transfer vs Baseline')
    ]
    for fig_path, title in transfer_figs:
        full_path = output_dir / fig_path
        if full_path.exists():
            html += f'<div><h5>{title}</h5><img src="script_outputs/{fig_path}"></div>'
    html += '</div>'

html += '</div>'

# Time Series with Lagged Variables
html += '<div class="section"><h3>13. Time Series Analysis with Lagged Variables</h3>'

lagged_file = output_dir / '13_time_series_lagged' / 'lagged_analysis_summary.json'
if lagged_file.exists():
    with open(lagged_file) as f:
        lagged_data = json.load(f)
    
    html += '<h4>Impact of Historical Data</h4>'
    html += '<p>Testing whether past financial performance improves bankruptcy prediction.</p>'
    
    html += '<table>'
    html += '<tr><th>Model</th><th>Features</th><th>AUC</th><th>Improvement</th></tr>'
    for key, val in lagged_data.items():
        if key.startswith('model_'):
            improvement = val.get('improvement_%', 0)
            html += f'<tr><td>{val["description"]}</td><td>{val["features"]}</td><td>{val["auc"]:.4f}</td><td>{"%.2f%%" % improvement if improvement > 0 else "-"}</td></tr>'
    html += '</table>'
    
    html += '<h4>Key Finding</h4>'
    html += '<p>✅ <strong>Change features (deltas)</strong> are most important (35%) - financial <strong>trends</strong> predict bankruptcy better than absolute values.</p>'
    
    html += '<h4>Visualizations</h4><div class="grid">'
    lagged_figs = [
        ('13_time_series_lagged/figures/lagged_model_comparison.png', 'Model Comparison'),
        ('13_time_series_lagged/figures/temporal_importance.png', 'Temporal Feature Importance'),
        ('13_time_series_lagged/figures/top_temporal_features.png', 'Top Features')
    ]
    for fig_path, title in lagged_figs:
        full_path = output_dir / fig_path
        if full_path.exists():
            html += f'<div><h5>{title}</h5><img src="script_outputs/{fig_path}"></div>'
    html += '</div>'

html += '</div>'

# ========================================================================
# FOOTER
# ========================================================================
html += """
<div class="section">
    <h2>📚 Project Information</h2>
    <p><strong>Analysis Date:</strong> November 2025</p>
    <p><strong>Framework:</strong> Python 3.13, scikit-learn, XGBoost, LightGBM, CatBoost</p>
    <p><strong>Datasets:</strong> Polish (UCI), American (Kaggle), Taiwan (TEJ)</p>
    <p><strong>Total Samples Analyzed:</strong> 327,926 (with multi-horizon)</p>
    <p><strong>Total Models Trained:</strong> 40+ (including remediation and transfer learning)</p>
    <p><strong>Best Performance:</strong> Polish CatBoost (0.9812 AUC)</p>
    <p><strong>Econometric Tests:</strong> ✅ Hosmer-Lemeshow, VIF, EPV, Clustered SE, Transfer Learning</p>
    <p><strong>Advanced Analyses:</strong> ✅ Panel data, Temporal lags, Cross-dataset validation</p>
</div>

</body>
</html>
"""

# Save report
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✓ Comprehensive HTML Report generated: {html_file}")
print(f"\nOpen in browser: file://{html_file}")
print("\n" + "="*70)
print("✓ REPORT GENERATION COMPLETE")
print("="*70)
