"""
Phase 02c: Correlation Analysis & Economic Plausibility Validation

Comprehensive analysis:
1. Pearson correlation matrix with hierarchical clustering
2. Identify high correlations (|r| > threshold from config, standard: 0.8) → multicollinearity candidates
3. Economic plausibility checks (expected direction of bankruptcy correlation)
4. Visualization with clustered heatmaps

Note: Correlation threshold of 0.8 is the established standard (PMC Epidemiology 2016)

OUTPUT: 02c_H[1-5]_correlation.xlsx/html/png + 02c_ALL_correlation.xlsx/html
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import logging

# Import project utilities
from src.bankruptcy_prediction.utils.target_utils import get_canonical_target
from src.bankruptcy_prediction.utils.metadata_loader import get_metadata
from src.bankruptcy_prediction.utils.logging_setup import setup_logging, print_header, print_section
from src.bankruptcy_prediction.utils.config_loader import get_config


def load_data(logger):
    """Load data with canonical target."""
    df = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'poland_imputed.parquet')
    logger.info(f"✓ Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Canonicalize target variable
    df = get_canonical_target(df, drop_duplicates=True)
    
    return df


def get_feature_columns(df, logger):
    """Extract feature columns."""
    cols = [c for c in df.columns if c.startswith('A') and c[1:].isdigit()]
    logger.info(f"✓ Found {len(cols)} features (A1-A64)")
    return sorted(cols, key=lambda x: int(x[1:]))


def analyze_correlations(df_h, features, horizon, output_dir, logger, corr_threshold=0.8):
    """
    Analyze correlations for one horizon.
    Creates clustered heatmap and identifies high correlations.
    
    Parameters
    ----------
    corr_threshold : float
        Correlation threshold from config (standard: 0.8)
    """
    logger.info("Analyzing correlations...")
    
    # Calculate Pearson correlation
    corr_matrix = df_h[features].corr(method='pearson')
    
    # Hierarchical clustering
    dist = 1 - abs(corr_matrix)
    linkage = hierarchy.linkage(squareform(dist), method='average')
    dendro = hierarchy.dendrogram(linkage, no_plot=True)
    order = dendro['leaves']
    corr_reordered = corr_matrix.iloc[order, order]
    
    # Identify high correlations
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > corr_threshold:
                high_corr.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': r,
                    'Abs_Correlation': abs(r)
                })
    
    high_corr_df = pd.DataFrame(high_corr).sort_values('Abs_Correlation', ascending=False)
    logger.info(f"  Found {len(high_corr_df)} high correlations (|r| > {corr_threshold})")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_reordered, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0, cbar_kws={"shrink": 0.8},
                xticklabels=True, yticklabels=True, ax=ax)
    ax.set_title(f'H{horizon}: Feature Correlation Matrix (Hierarchical Clustering)', 
                 fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / f'02c_H{horizon}_correlation_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved 02c_H{horizon}_correlation_heatmap.png")
    
    return corr_matrix, high_corr_df


def analyze_economic_plausibility(df_h, features, metadata, logger):
    """
    Check economic plausibility of feature-bankruptcy relationships.
    Uses metadata-driven category lookup (no hard-coding).
    """
    logger.info("Validating economic plausibility...")
    
    results = []
    for feat in features:
        # Correlation with bankruptcy
        corr, pval = stats.pearsonr(df_h[feat], df_h['y'])  # Using canonical 'y'
        
        # Get category from metadata
        category = metadata.get_category(feat)
        
        # Expected direction based on category
        if category in ['Profitability', 'Liquidity', 'Activity']:
            expected = 'negative'  # Higher profitability/liquidity should reduce bankruptcy
        elif category in ['Leverage']:
            expected = 'positive'  # Higher leverage should increase bankruptcy
        else:
            expected = 'unknown'
        
        # Check if observed matches expected
        observed = 'positive' if corr > 0 else 'negative'
        plausible = (expected == observed) if expected != 'unknown' else True
        
        results.append({
            'Feature': feat,
            'Category': category,
            'Correlation_with_Bankruptcy': corr,
            'P_Value': pval,
            'Significant': pval < 0.05,
            'Expected_Direction': expected,
            'Observed_Direction': observed,
            'Economically_Plausible': plausible
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Correlation_with_Bankruptcy', key=abs, ascending=False)
    
    implausible = (~results_df['Economically_Plausible']).sum()
    logger.info(f"  Economically implausible: {implausible}/{len(results_df)}")
    
    return results_df


def analyze_horizon(df, horizon, features, metadata, output_dir, logger, corr_threshold=0.8):
    """Complete analysis for one horizon.
    
    Parameters
    ----------
    corr_threshold : float
        Correlation threshold from config (standard: 0.8)
    """
    print_section(logger, f"HORIZON {horizon} (H{horizon})", width=60)
    
    df_h = df[df['horizon'] == horizon]
    
    # Correlation analysis
    corr_matrix, high_corr_df = analyze_correlations(df_h, features, horizon, output_dir, logger, corr_threshold=corr_threshold)
    
    # Economic plausibility
    econ_df = analyze_economic_plausibility(df_h, features, metadata, logger)
    
    # Save Excel
    excel_path = output_dir / f'02c_H{horizon}_correlation.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        high_corr_df.to_excel(writer, sheet_name='High_Correlations', index=False)
        econ_df.to_excel(writer, sheet_name='Economic_Validation', index=False)
        corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
        
        meta = pd.DataFrame([{
            'Horizon': horizon,
            'Total_Features': len(features),
            'High_Correlations': len(high_corr_df),
            'Economically_Implausible': (~econ_df['Economically_Plausible']).sum()
        }])
        meta.to_excel(writer, sheet_name='Metadata', index=False)
    
    logger.info(f"✓ Saved {excel_path.name}")
    
    # Create HTML
    create_html(high_corr_df, econ_df, horizon, output_dir, logger, corr_threshold=corr_threshold)
    
    logger.info(f"✅ H{horizon} Complete")
    return high_corr_df, econ_df


def create_html(high_corr_df, econ_df, horizon, output_dir, logger, corr_threshold=0.8):
    """Create HTML report.
    
    Parameters
    ----------
    corr_threshold : float
        Correlation threshold from config
    """
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>02c H{horizon} Correlation Analysis</title>
<style>
body {{font-family: Arial; margin: 20px; background: #f5f5f5;}}
.container {{max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;}}
h1 {{color: #2c3e50; border-bottom: 3px solid #9b59b6;}}
.metric {{display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
color: white; padding: 20px; margin: 10px; border-radius: 8px; min-width: 200px; text-align: center;}}
table {{width: 100%; border-collapse: collapse; margin: 20px 0;}}
th, td {{border: 1px solid #ddd; padding: 12px; text-align: left;}}
th {{background: #9b59b6; color: white;}}
tr:nth-child(even) {{background: #f2f2f2;}}
.info {{background: #e7f3ff; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0;}}
.warning {{background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0;}}
img {{max-width: 100%; margin: 20px 0; border: 1px solid #ddd;}}
</style></head><body><div class="container">
<h1>📊 Phase 02c: Correlation Analysis - Horizon {horizon}</h1>

<div class="info">
<strong>Purpose:</strong> Identify multicollinearity patterns and validate economic plausibility
<ul><li>High correlations (|r| > {corr_threshold}) → Candidates for removal in Phase 03</li>
<li>Economic validation → Ensure features behave as expected</li></ul>
</div>

<div>
<div class="metric"><div>High Correlations</div><div style="font-size:28px; font-weight:bold;">{len(high_corr_df)}</div></div>
<div class="metric"><div>Economically Plausible</div><div style="font-size:28px; font-weight:bold;">{econ_df['Economically_Plausible'].sum()}/64</div></div>
</div>

<h2>Correlation Heatmap</h2>
<img src="02c_H{horizon}_correlation_heatmap.png" alt="Correlation Heatmap">

<h2>High Correlations (|r| > {corr_threshold}) - Top 20</h2>
<div class="warning">
<strong>Multicollinearity Alert:</strong> These feature pairs are highly correlated and may cause issues in modeling.
Phase 03 will use VIF analysis to systematically remove redundant features.
</div>
<table><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation</th></tr>"""
    
    for row in high_corr_df.head(20).itertuples():
        html += f"""<tr><td>{row.Feature_1}</td><td>{row.Feature_2}</td><td>{row.Correlation:.3f}</td></tr>"""
    
    html += f"""</table>

<h2>Economic Plausibility - Top 10 by Correlation with Bankruptcy</h2>
<table><tr><th>Feature</th><th>Category</th><th>Correlation</th><th>Expected</th><th>Observed</th><th>Plausible?</th></tr>"""
    
    for row in econ_df.head(10).itertuples():
        plausible_icon = "✅" if row.Economically_Plausible else "❌"
        html += f"""<tr><td><strong>{row.Feature}</strong></td><td>{row.Category}</td>
<td>{row.Correlation_with_Bankruptcy:.3f}</td><td>{row.Expected_Direction}</td>
<td>{row.Observed_Direction}</td><td>{plausible_icon}</td></tr>"""
    
    html += f"""</table>

<div class="info">
<h3>Next Steps</h3>
<ul><li><strong>Phase 03:</strong> VIF analysis to remove multicollinear features</li>
<li><strong>Target:</strong> VIF < 10 for all retained features</li>
<li><strong>Method:</strong> Iterative removal starting with highest VIF</li></ul>
</div>

<p style="text-align:center; color:#666; margin-top:40px;">
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
Phase 02c: Correlation Analysis - Horizon {horizon}</p>
</div></body></html>"""
    
    html_path = output_dir / f'02c_H{horizon}_correlation.html'
    with open(html_path, 'w') as f:
        f.write(html)
    logger.info(f"✓ Saved {html_path.name}")


def create_consolidated(all_high_corr, all_econ, output_dir, logger, corr_threshold=0.8):
    """Create consolidated reports.
    
    Parameters
    ----------
    corr_threshold : float
        Correlation threshold from config
    """
    print_section(logger, "CREATING CONSOLIDATED REPORTS", width=60)
    
    # Excel
    excel_path = output_dir / '02c_ALL_correlation.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        overview = []
        for h in range(1, 6):
            if f'H{h}' in all_high_corr:
                high_corr = all_high_corr[f'H{h}']
                econ = all_econ[f'H{h}']
                overview.append({
                    'Horizon': h,
                    'High_Correlations': len(high_corr),
                    'Economically_Plausible': econ['Economically_Plausible'].sum(),
                    'Economically_Implausible': (~econ['Economically_Plausible']).sum()
                })
                high_corr.to_excel(writer, sheet_name=f'H{h}_HighCorr', index=False)
                econ.to_excel(writer, sheet_name=f'H{h}_Economic', index=False)
        
        pd.DataFrame(overview).to_excel(writer, sheet_name='Overview', index=False)
    
    logger.info(f"✓ Saved {excel_path.name}")
    
    # HTML
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>02c Correlation Analysis - All Horizons</title>
<style>
body {{font-family: Arial; margin: 20px; background: #f5f5f5;}}
.container {{max-width: 1600px; margin: 0 auto; background: white; padding: 30px;}}
h1 {{color: #2c3e50; border-bottom: 3px solid #9b59b6;}}
table {{width: 100%; border-collapse: collapse; margin: 20px 0;}}
th, td {{border: 1px solid #ddd; padding: 12px;}}
th {{background: #9b59b6; color: white;}}
</style></head><body><div class="container">
<h1>📊 Phase 02c: Correlation Analysis - All Horizons</h1>
<h2>Overview</h2>
<table><tr><th>Horizon</th><th>High Correlations (|r|>{corr_threshold})</th><th>Economically Plausible</th><th>Implausible</th></tr>"""
    
    for item in overview:
        html += f"""<tr><td>H{item['Horizon']}</td><td>{item['High_Correlations']}</td>
<td>{item['Economically_Plausible']}</td><td>{item['Economically_Implausible']}</td></tr>"""
    
    html += """</table>
<h2>Individual Reports</h2>
<ul>""" + "\n".join([f"<li><a href='02c_H{h}_correlation.html'>H{h} Report</a></li>" for h in range(1,6)]) + f"""</ul>
<p style="text-align:center; color:#666; margin-top:40px;">
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div></body></html>"""
    
    with open(output_dir / '02c_ALL_correlation.html', 'w') as f:
        f.write(html)
    logger.info("✓ Saved 02c_ALL_correlation.html")


def main():
    """Main execution."""
    # Setup logging
    logger = setup_logging('02c_correlation_economic')
    
    # Load configuration
    config = get_config()
    corr_threshold = config.get('analysis', 'correlation_threshold')
    logger.info(f"Configuration: Correlation threshold = {corr_threshold} (standard for high correlation)")
    
    print_header(logger, "PHASE 02c: CORRELATION & ECONOMIC VALIDATION")
    logger.info("Starting correlation analysis and economic validation...")
    
    try:
        df = load_data(logger)
        features = get_feature_columns(df, logger)
        
        # Load metadata for category lookup
        metadata = get_metadata()
        logger.info(f"✓ Loaded metadata: {metadata.summary()}")
        
        output_dir = PROJECT_ROOT / 'results' / '02_exploratory_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_high_corr = {}
        all_econ = {}
        
        for h in [1, 2, 3, 4, 5]:
            high_corr_df, econ_df = analyze_horizon(df, h, features, metadata, output_dir, logger, corr_threshold=corr_threshold)
            all_high_corr[f'H{h}'] = high_corr_df
            all_econ[f'H{h}'] = econ_df
        
        create_consolidated(all_high_corr, all_econ, output_dir, logger, corr_threshold=corr_threshold)
        
        logger.info("\n" + "="*60)
        logger.info("✅ PHASE 02c COMPLETE")
        logger.info("="*60)
        logger.info("Generated: 5 Excel + 5 HTML + 5 PNG per-horizon + 2 consolidated = 17 files")
        
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
