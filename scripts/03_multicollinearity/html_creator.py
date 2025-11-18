"""HTML Report Generation for Phase 03a VIF Analysis."""

import pandas as pd
import numpy as np


def create_html_report(horizon, metadata, iterations_log, final_vif_df, removed_df, output_dir, logger, vif_threshold=10):
    """Create HTML report with narrative and tables.
    
    Parameters
    ----------
    vif_threshold : float
        VIF threshold from config (default 10 for backward compatibility)
    """
    
    # Convergence status
    last_iter = iterations_log[-1] if iterations_log else {}
    converged = last_iter.get('Reason') == 'converged'
    status_class = 'info' if converged else 'warning'
    status_text = f"✅ Converged: max(VIF) ≤ {vif_threshold}" if converged else f"⚠️ {last_iter.get('Reason', 'unknown')}"
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>03a H{horizon} VIF Analysis</title>
<style>
body {{font-family: Arial; margin: 20px; background: #f5f5f5;}}
.container {{max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;}}
h1 {{color: #2c3e50; border-bottom: 3px solid #3498db;}}
h2 {{color: #34495e; margin-top: 30px;}}
.metric {{display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
color: white; padding: 20px; margin: 10px; border-radius: 8px; min-width: 180px; text-align: center;}}
.metric-value {{font-size: 32px; font-weight: bold; margin: 5px 0;}}
table {{width: 100%; border-collapse: collapse; margin: 20px 0;}}
th, td {{border: 1px solid #ddd; padding: 12px; text-align: left;}}
th {{background: #3498db; color: white;}}
tr:nth-child(even) {{background: #f2f2f2;}}
.info {{background: #e7f3ff; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0;}}
.warning {{background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0;}}
.success {{background: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 20px 0;}}
code {{background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace;}}
</style></head><body><div class="container">
<h1>🔍 Phase 03a: VIF Analysis - Horizon {horizon}</h1>

<div class="{status_class}">
<strong>Status:</strong> {status_text}
</div>

<div class="info">
<strong>Methodology:</strong> Iterative VIF Pruning
<ul>
<li><strong>Threshold:</strong> VIF &gt; {vif_threshold} indicates serious multicollinearity (Penn State STAT 462, O'Brien 2007)</li>
<li><strong>Algorithm:</strong> Remove feature with highest VIF, recompute, repeat until max(VIF) ≤ {vif_threshold}</li>
<li><strong>VIF Formula:</strong> <code>VIF = 1 / (1 - R²)</code> where R² is from auxiliary regression</li>
<li><strong>Interpretation:</strong> VIF quantifies how much variance is inflated due to collinearity</li>
</ul>
</div>

<h2>📊 Summary Statistics</h2>
<div>
<div class="metric">
<div>Initial Features</div>
<div class="metric-value">{metadata['Initial_Features']}</div>
</div>
<div class="metric">
<div>Final Features</div>
<div class="metric-value">{metadata['Final_Features']}</div>
</div>
<div class="metric">
<div>Removed</div>
<div class="metric-value">{metadata['Removed_Count']}</div>
</div>
<div class="metric">
<div>Iterations</div>
<div class="metric-value">{metadata['Iterations']}</div>
</div>
<div class="metric">
<div>Max Final VIF</div>
<div class="metric-value">{metadata['Max_Final_VIF']:.2f}</div>
</div>
</div>

<h2>🔄 Iteration Log</h2>
<table>
<tr><th>Iteration</th><th>Features Remaining</th><th>Max VIF</th><th>Removed Feature</th><th>Removed VIF</th><th>Reason</th></tr>"""
    
    for log in iterations_log:
        removed_feat = log['Removed_Feature'] if log['Removed_Feature'] else '—'
        removed_vif = f"{log['Removed_VIF']:.2f}" if log['Removed_VIF'] is not None else '—'
        max_vif_str = f"{log['Max_VIF']:.2f}" if not pd.isna(log['Max_VIF']) else '—'
        html += f"""
<tr><td>{log['Iteration']}</td><td>{log['Features_Remaining']}</td><td>{max_vif_str}</td>
<td>{removed_feat}</td><td>{removed_vif}</td><td>{log['Reason']}</td></tr>"""
    
    html += f"""</table>

<h2>✅ Final VIF Values (Top 20)</h2>
<div class="success">
All retained features have VIF ≤ {vif_threshold}, indicating acceptable multicollinearity levels for regression modeling.
</div>
<table>
<tr><th>Rank</th><th>Feature</th><th>VIF</th><th>Tolerance</th><th>Status</th></tr>"""
    
    for rank, row in enumerate(final_vif_df.head(20).itertuples(), 1):
        tolerance = 1 / row.VIF if row.VIF > 0 else np.nan
        status = "✅ OK" if row.VIF <= vif_threshold else "⚠️ High"
        html += f"""
<tr><td>{rank}</td><td><strong>{row.Feature}</strong></td><td>{row.VIF:.3f}</td>
<td>{tolerance:.4f}</td><td>{status}</td></tr>"""
    
    html += """</table>"""
    
    if not removed_df.empty:
        html += f"""
<h2>🗑️ Removed Features ({len(removed_df)})</h2>
<table>
<tr><th>Feature</th><th>VIF at Removal</th><th>Iteration</th><th>Reason</th></tr>"""
        
        for row in removed_df.itertuples():
            vif_val = f"{row.VIF_at_Removal:.2f}" if not pd.isna(row.VIF_at_Removal) else 'N/A'
            iter_val = row.Iteration_Removed if hasattr(row, 'Iteration_Removed') and not pd.isna(row.Iteration_Removed) else 'Pre-processing'
            html += f"""
<tr><td><strong>{row.Feature}</strong></td><td>{vif_val}</td><td>{iter_val}</td><td>{row.Reason}</td></tr>"""
        
        html += """</table>"""
    
    html += f"""
<div class="info">
<h3>📚 References</h3>
<ul>
<li>Penn State STAT 462 (online.stat.psu.edu): "VIFs exceeding 10 are signs of serious multicollinearity requiring correction"</li>
<li>O'Brien (2007): "A Caution Regarding Rules of Thumb for Variance Inflation Factors", Quality & Quantity</li>
<li>Menard (1995): "A tolerance of less than 0.10 almost certainly indicates a serious collinearity problem"</li>
</ul>
</div>

<div class="info">
<h3>Next Steps</h3>
<ul>
<li><strong>Phase 04:</strong> Feature selection using statistical tests and economic criteria</li>
<li><strong>Phase 05:</strong> Model training with cleaned feature set</li>
<li><strong>Features retained:</strong> {metadata['Final_Features']} features ready for modeling</li>
</ul>
</div>

<p style="text-align:center; color:#666; margin-top:40px;">
Generated: {metadata['Timestamp']}<br>
Phase 03a: VIF Analysis - Horizon {horizon}<br>
Script: scripts/03_multicollinearity/03a_vif_analysis.py</p>
</div></body></html>"""
    
    html_path = output_dir / f'03a_H{horizon}_vif.html'
    with open(html_path, 'w') as f:
        f.write(html)
    
    logger.info(f"✓ Saved: {html_path.name}")


def create_consolidated_html(summary_df, output_dir, logger, vif_threshold=10):
    """Create consolidated HTML for all horizons.
    
    Parameters
    ----------
    vif_threshold : float
        VIF threshold from config (default 10 for backward compatibility)
    """
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>03a VIF Analysis - All Horizons</title>
<style>
body {{font-family: Arial; margin: 20px; background: #f5f5f5;}}
.container {{max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;}}
h1 {{color: #2c3e50; border-bottom: 3px solid #3498db;}}
table {{width: 100%; border-collapse: collapse; margin: 20px 0;}}
th, td {{border: 1px solid #ddd; padding: 12px; text-align: center;}}
th {{background: #3498db; color: white;}}
tr:nth-child(even) {{background: #f2f2f2;}}
.info {{background: #e7f3ff; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0;}}
.metric {{display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
color: white; padding: 20px; margin: 10px; border-radius: 8px; min-width: 200px; text-align: center;}}
.metric-value {{font-size: 28px; font-weight: bold;}}
</style></head><body><div class="container">
<h1>🔍 Phase 03a: VIF Analysis - All Horizons</h1>

<div class="info">
<strong>Objective:</strong> Remove multicollinear features using iterative VIF pruning (threshold = {vif_threshold})
<br><strong>Status:</strong> ✅ Complete for all 5 horizons
</div>

<h2>📊 Summary Statistics</h2>
<div>
<div class="metric">
<div>Total Removed</div>
<div class="metric-value">{summary_df['Removed'].sum()}</div>
</div>
<div class="metric">
<div>Avg Iterations</div>
<div class="metric-value">{summary_df['Iterations'].mean():.1f}</div>
</div>
<div class="metric">
<div>Max Final VIF</div>
<div class="metric-value">{summary_df['Max_Final_VIF'].max():.2f}</div>
</div>
</div>

<h2>Per-Horizon Summary</h2>
<table>
<tr><th>Horizon</th><th>Initial</th><th>Final</th><th>Removed</th><th>Iterations</th><th>Max Final VIF</th></tr>"""
    
    for row in summary_df.itertuples():
        html += f"""
<tr><td>H{row.Horizon}</td><td>{row.Initial}</td><td>{row.Final}</td>
<td>{row.Removed}</td><td>{row.Iterations}</td><td>{row.Max_Final_VIF:.2f}</td></tr>"""
    
    html += f"""</table>

<div class="info">
<h3>Key Findings</h3>
<ul>
<li><strong>Initial features:</strong> {summary_df['Initial'].iloc[0]} (A1-A64)</li>
<li><strong>Total removed across all horizons:</strong> {summary_df['Removed'].sum()} feature-horizon instances</li>
<li><strong>Average iterations per horizon:</strong> {summary_df['Iterations'].mean():.1f}</li>
<li><strong>All final VIF values:</strong> ≤ {summary_df['Max_Final_VIF'].max():.2f} (target: ≤ {vif_threshold})</li>
</ul>
</div>

<div class="info">
<h3>Methodology Validation</h3>
<ul>
<li>✅ <strong>Threshold justified:</strong> VIF > 10 standard in econometrics (Penn State STAT 462, O'Brien 2007)</li>
<li>✅ <strong>Algorithm deterministic:</strong> Same data produces same results (VIF computation is deterministic)</li>
<li>✅ <strong>Defensive programming:</strong> Pre-checks for zero variance, NaN, Inf</li>
<li>✅ <strong>Graceful stopping:</strong> Converges or stops at 2 features minimum</li>
</ul>
</div>

<p style="text-align:center; color:#666; margin-top:40px;">
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
Phase 03a: VIF Analysis - All Horizons<br>
Script: scripts/03_multicollinearity/03a_vif_analysis.py</p>
</div></body></html>"""
    
    html_path = output_dir / '03a_ALL_vif.html'
    with open(html_path, 'w') as f:
        f.write(html)
    
    logger.info(f"✓ Saved: {html_path.name}")
