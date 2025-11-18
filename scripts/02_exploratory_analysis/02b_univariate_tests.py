"""
Phase 02b: Univariate Statistical Tests with A+ Methodology

Rigorous statistical approach for bankruptcy prediction:

1. **Normality Assessment** (D'Agostino-Pearson K² test)
   - More appropriate for large samples (n > 5000) than Shapiro-Wilk
   - Combined skewness + kurtosis test
   - Fallback: |skewness| > 2 OR |kurtosis| > 5 → non-normal

2. **Test Selection**:
   - Both normal + equal variance → Student's t-test
   - Both normal + unequal variance → Welch's t-test
   - Non-normal → Mann-Whitney U test (non-parametric)

3. **Effect Sizes**:
   - Cohen's d (parametric tests)
   - Rank-biserial correlation (Mann-Whitney U)
   - Point-biserial correlation (all tests)

4. **Multiple Testing Correction**:
   - Benjamini-Hochberg FDR procedure  
   - Controls false discovery rate PER HORIZON (64 features each, 5 horizons total)
   - Separate FDR control for each horizon appropriate for independent models
   - Reports both p-values and q-values (FDR-adjusted)

5. **Canonical Target**: Uses 'y' column consistently

OUTPUT: 02b_H[1-5]_univariate_tests.xlsx/html + 02b_ALL_univariate_tests.xlsx/html
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import logging

# Import project utilities
from src.bankruptcy_prediction.utils.target_utils import get_canonical_target
from src.bankruptcy_prediction.utils.logging_setup import setup_logging, print_header, print_section


def load_data(logger):
    """Load imputed dataset with canonical target."""
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


def cohen_d(x, y):
    """Calculate Cohen's d effect size."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)


def rank_biserial(u, n1, n2):
    """Calculate rank-biserial correlation from Mann-Whitney U."""
    return 1 - (2*u) / (n1 * n2)


def interpret_effect(size, effect_type='d'):
    """Interpret effect size magnitude."""
    size = abs(size)
    if effect_type == 'd':
        if size < 0.2: return 'negligible'
        elif size < 0.5: return 'small'
        elif size < 0.8: return 'medium'
        else: return 'large'
    else:  # rank-biserial
        if size < 0.1: return 'negligible'
        elif size < 0.3: return 'small'
        elif size < 0.5: return 'medium'
        else: return 'large'


def test_normality_large_sample(data, feature):
    """
    Test normality using D'Agostino-Pearson K² test.
    More appropriate for large samples than Shapiro-Wilk.
    
    Combines skewness and kurtosis into omnibus test.
    """
    try:
        # D'Agostino-Pearson test (scipy.stats.normaltest)
        stat, p_value = stats.normaltest(data[feature])
        
        # Also check distribution shape heuristics
        skewness = stats.skew(data[feature])
        kurtosis = stats.kurtosis(data[feature])  # Excess kurtosis
        
        # Consider non-normal if:
        # 1. K² test rejects (p < 0.05), OR
        # 2. Extreme skewness (|skew| > 2), OR
        # 3. Extreme kurtosis (|kurt| > 5)
        is_normal = (p_value > 0.05) and (abs(skewness) <= 2) and (abs(kurtosis) <= 5)
        
        return is_normal, p_value, skewness, kurtosis
    except:
        # If test fails (e.g., constant column), assume non-normal
        return False, 0.0, np.nan, np.nan


def test_feature(feat, bankrupt, non_bankrupt):
    """
    Test one feature with proper assumption validation.
    Returns dict with test results.
    """
    # Test normality using D'Agostino K² (appropriate for large n)
    norm_b, p_norm_b, skew_b, kurt_b = test_normality_large_sample(bankrupt, feat)
    norm_nb, p_norm_nb, skew_nb, kurt_nb = test_normality_large_sample(non_bankrupt, feat)
    both_normal = norm_b and norm_nb
    
    if both_normal:
        # Test variance equality
        _, p_levene = stats.levene(bankrupt[feat], non_bankrupt[feat])
        equal_var = p_levene > 0.05
        
        if equal_var:
            # Student's t-test
            stat, pval = stats.ttest_ind(bankrupt[feat], non_bankrupt[feat])
            test_used = "t-test"
        else:
            # Welch's t-test
            stat, pval = stats.ttest_ind(bankrupt[feat], non_bankrupt[feat], equal_var=False)
            test_used = "Welch's t-test"
        
        # Cohen's d
        effect = cohen_d(bankrupt[feat].values, non_bankrupt[feat].values)
        effect_type = "Cohen's d"
    else:
        # Mann-Whitney U test
        u_stat, pval = stats.mannwhitneyu(bankrupt[feat], non_bankrupt[feat], alternative='two-sided')
        test_used = "Mann-Whitney U"
        effect = rank_biserial(u_stat, len(bankrupt), len(non_bankrupt))
        effect_type = "Rank-biserial r"
    
    # Point-biserial correlation
    y = np.concatenate([np.ones(len(bankrupt)), np.zeros(len(non_bankrupt))])
    x = np.concatenate([bankrupt[feat].values, non_bankrupt[feat].values])
    pb_corr, _ = stats.pearsonr(x, y)
    
    return {
        'Feature': feat,
        'Test': test_used,
        'P_Value': pval,
        'Significant_p05': pval < 0.05,  # Will be updated with FDR
        'Effect_Size': effect,
        'Effect_Type': effect_type,
        'Effect_Interp': interpret_effect(effect, 'd' if 'Cohen' in effect_type else 'r'),
        'Point_Biserial_R': pb_corr,
        'Normal_Bankrupt': norm_b,
        'Normal_NonBankrupt': norm_nb,
        'Skewness_Bankrupt': skew_b,
        'Kurtosis_Bankrupt': kurt_b
    }


def analyze_horizon(df, horizon, features, output_dir, logger):
    """Analyze all features for one horizon."""
    print_section(logger, f"HORIZON {horizon} (H{horizon})", width=60)
    
    df_h = df[df['horizon'] == horizon]
    bankrupt = df_h[df_h['y'] == 1]  # Using canonical target 'y'
    non_bankrupt = df_h[df_h['y'] == 0]
    
    logger.info(f"Bankrupt: {len(bankrupt)} | Non-bankrupt: {len(non_bankrupt)}")
    
    # Test all features
    results = [test_feature(f, bankrupt, non_bankrupt) for f in features]
    results_df = pd.DataFrame(results)
    
    # Apply Benjamini-Hochberg FDR correction
    logger.info("Applying Benjamini-Hochberg FDR correction...")
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(
        results_df['P_Value'].values,
        alpha=0.05,
        method='fdr_bh'  # Benjamini-Hochberg
    )
    
    results_df['Q_Value_FDR'] = pvals_corrected
    results_df['Significant_FDR_q05'] = reject  # FDR-corrected significance
    
    # Sort by effect size
    results_df = results_df.sort_values('Effect_Size', key=abs, ascending=False).reset_index(drop=True)
    
    # Summary
    sig_p = results_df['Significant_p05'].sum()
    sig_q = results_df['Significant_FDR_q05'].sum()
    parametric = results_df['Test'].str.contains('t-test').sum()
    
    logger.info(f"Significant (p<0.05): {sig_p}/{len(results_df)} ({sig_p/len(results_df)*100:.1f}%)")
    logger.info(f"Significant (FDR q<0.05): {sig_q}/{len(results_df)} ({sig_q/len(results_df)*100:.1f}%)")
    logger.info(f"Parametric: {parametric} | Non-parametric: {len(results_df)-parametric}")
    
    # Save Excel
    excel_path = output_dir / f'02b_H{horizon}_univariate_tests.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='All_Results', index=False)
        results_df[results_df['Significant_FDR_q05']].to_excel(writer, sheet_name='Significant_FDR', index=False)
        results_df[results_df['Effect_Interp'].isin(['medium', 'large'])].to_excel(writer, sheet_name='Large_Effects', index=False)
        
        # Comparison: p vs q significance
        comparison = pd.DataFrame({
            'Feature': results_df['Feature'],
            'P_Value': results_df['P_Value'],
            'Q_Value_FDR': results_df['Q_Value_FDR'],
            'Sig_p05': results_df['Significant_p05'],
            'Sig_FDR_q05': results_df['Significant_FDR_q05'],
            'Lost_after_FDR': results_df['Significant_p05'] & ~results_df['Significant_FDR_q05']
        })
        comparison.to_excel(writer, sheet_name='FDR_Comparison', index=False)
        
        meta = pd.DataFrame([{
            'Horizon': horizon,
            'Total_Features': len(results_df),
            'Significant_p05': sig_p,
            'Significant_FDR_q05': sig_q,
            'Lost_after_FDR': sig_p - sig_q,
            'Parametric_Tests': parametric,
            'NonParametric_Tests': len(results_df) - parametric
        }])
        meta.to_excel(writer, sheet_name='Metadata', index=False)
    
    logger.info(f"✓ Saved {excel_path.name}")
    
    # Create HTML
    create_html(results_df, horizon, len(bankrupt), len(non_bankrupt), output_dir, logger)
    
    logger.info(f"✅ H{horizon} Complete")
    return results_df


def create_html(df, horizon, n_b, n_nb, output_dir, logger):
    """Create HTML report."""
    sig_p = df['Significant_p05'].sum()
    sig_q = df['Significant_FDR_q05'].sum()
    param = df['Test'].str.contains('t-test').sum()
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>02b H{horizon} Univariate Tests</title>
<style>
body {{font-family: Arial; margin: 20px; background: #f5f5f5;}}
.container {{max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;}}
h1 {{color: #2c3e50; border-bottom: 3px solid #3498db;}}
.metric {{display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
color: white; padding: 20px; margin: 10px; border-radius: 8px; min-width: 200px; text-align: center;}}
table {{width: 100%; border-collapse: collapse; margin: 20px 0;}}
th, td {{border: 1px solid #ddd; padding: 12px; text-align: left;}}
th {{background: #3498db; color: white;}}
tr:nth-child(even) {{background: #f2f2f2;}}
.info {{background: #e7f3ff; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0;}}
</style></head><body><div class="container">
<h1>📊 Phase 02b: Univariate Tests - Horizon {horizon}</h1>
<div class="info">
<strong>Methodology:</strong> A+ Statistical Rigor
<ul><li>Step 1: Test normality (D'Agostino K² - appropriate for large n)</li>
<li>Step 2: Test selection based on normality + variance equality</li>
<li>Step 3: Effect sizes (Cohen's d or rank-biserial)</li>
<li>Step 4: FDR correction (Benjamini-Hochberg) to control false discoveries</li></ul>
</div>
<div>
<div class="metric"><div>Total Features</div><div style="font-size:28px; font-weight:bold;">{len(df)}</div></div>
<div class="metric"><div>Significant (p&lt;0.05)</div><div style="font-size:28px; font-weight:bold;">{sig_p}</div></div>
<div class="metric"><div>Significant (FDR q&lt;0.05)</div><div style="font-size:28px; font-weight:bold;">{sig_q}</div></div>
<div class="metric"><div>Parametric Tests</div><div style="font-size:28px; font-weight:bold;">{param}</div></div>
</div>
<h2>Top 10 Features by Effect Size</h2>
<p><strong>Note:</strong> Significance based on FDR-corrected q-values (Benjamini-Hochberg procedure)</p>
<table><tr><th>Rank</th><th>Feature</th><th>Test</th><th>P-Value</th><th>Q-Value (FDR)</th><th>Effect Size</th><th>Sig?</th></tr>"""
    
    for i, row in enumerate(df.head(10).itertuples(), 1):
        sig_marker = "✅" if row.Significant_FDR_q05 else "❌"
        html += f"""<tr><td>{i}</td><td><strong>{row.Feature}</strong></td><td>{row.Test}</td>
<td>{row.P_Value:.2e}</td><td>{row.Q_Value_FDR:.2e}</td><td>{row.Effect_Size:.3f} ({row.Effect_Type})</td><td>{sig_marker}</td></tr>"""
    
    html += f"""</table>
<p style="text-align:center; color:#666; margin-top:40px;">
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
Phase 02b: Univariate Statistical Tests - Horizon {horizon}</p>
</div></body></html>"""
    
    html_path = output_dir / f'02b_H{horizon}_univariate_tests.html'
    with open(html_path, 'w') as f:
        f.write(html)
    logger.info(f"✓ Saved {html_path.name}")


def create_consolidated(all_results, output_dir, logger):
    """Create consolidated reports."""
    print_section(logger, "CREATING CONSOLIDATED REPORTS", width=60)
    
    # Excel
    excel_path = output_dir / '02b_ALL_univariate_tests.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        overview = []
        for h in range(1, 6):
            if f'H{h}' in all_results:
                df = all_results[f'H{h}']
                sig_p = df['Significant_p05'].sum()
                sig_q = df['Significant_FDR_q05'].sum()
                param = df['Test'].str.contains('t-test').sum()
                overview.append({
                    'Horizon': h,
                    'Total': len(df),
                    'Significant_p05': sig_p,
                    'Significant_FDR_q05': sig_q,
                    'Lost_after_FDR': sig_p - sig_q,
                    'Pct_Significant_FDR': f"{sig_q/len(df)*100:.1f}%",
                    'Parametric': param,
                    'NonParametric': len(df) - param
                })
                df.to_excel(writer, sheet_name=f'H{h}_All', index=False)
        
        pd.DataFrame(overview).to_excel(writer, sheet_name='Overview', index=False)
        
        if 'H5' in all_results:
            all_results['H5'].head(20).to_excel(writer, sheet_name='Top_Features_H5', index=False)
    
    logger.info(f"✓ Saved {excel_path.name}")
    
    # HTML
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>02b Univariate Tests - All Horizons</title>
<style>
body {{font-family: Arial; margin: 20px; background: #f5f5f5;}}
.container {{max-width: 1600px; margin: 0 auto; background: white; padding: 30px;}}
h1 {{color: #2c3e50; border-bottom: 3px solid #3498db;}}
table {{width: 100%; border-collapse: collapse; margin: 20px 0;}}
th, td {{border: 1px solid #ddd; padding: 12px;}}
th {{background: #3498db; color: white;}}
</style></head><body><div class="container">
<h1>📊 Phase 02b: Univariate Tests - All Horizons</h1>
<h2>Overview</h2>
<table><tr><th>Horizon</th><th>Total</th><th>Sig (p&lt;0.05)</th><th>Sig (FDR q&lt;0.05)</th><th>Lost after FDR</th><th>Parametric</th><th>Non-Parametric</th></tr>"""
    
    for item in overview:
        html += f"""<tr><td>H{item['Horizon']}</td><td>{item['Total']}</td><td>{item['Significant_p05']}</td>
<td>{item['Significant_FDR_q05']}</td><td>{item['Lost_after_FDR']}</td>
<td>{item['Parametric']}</td><td>{item['NonParametric']}</td></tr>"""
    
    html += f"""</table>
<h2>Individual Reports</h2>
<ul>""" + "\n".join([f"<li><a href='02b_H{h}_univariate_tests.html'>H{h} Report</a></li>" for h in range(1,6)]) + f"""</ul>
<p style="text-align:center; color:#666; margin-top:40px;">
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div></body></html>"""
    
    with open(output_dir / '02b_ALL_univariate_tests.html', 'w') as f:
        f.write(html)
    logger.info("✓ Saved 02b_ALL_univariate_tests.html")


def main():
    """Main execution."""
    # Setup logging
    logger = setup_logging('02b_univariate_tests')
    
    print_header(logger, "PHASE 02b: UNIVARIATE STATISTICAL TESTS")
    logger.info("Starting univariate tests with FDR correction...")
    
    try:
        df = load_data(logger)
        features = get_feature_columns(df, logger)
        output_dir = PROJECT_ROOT / 'results' / '02_exploratory_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        for h in [1, 2, 3, 4, 5]:
            results_df = analyze_horizon(df, h, features, output_dir, logger)
            all_results[f'H{h}'] = results_df
        
        create_consolidated(all_results, output_dir, logger)
        
        logger.info("\n" + "="*60)
        logger.info("✅ PHASE 02b COMPLETE")
        logger.info("="*60)
        logger.info("Generated: 5 Excel + 5 HTML per-horizon + 2 consolidated = 12 files")
        
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
