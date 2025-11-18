"""
Extract verified facts from Phase 02 results for paper writing.
NO HALLUCINATIONS - only real numbers from actual Excel files.
"""

import pandas as pd
import json
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results' / '02_exploratory_analysis'

def extract_facts():
    """Extract all facts from Phase 02 results."""
    facts = {}
    
    # ===== 02a: Distribution Analysis =====
    print("Extracting 02a facts...")
    facts['02a'] = {}
    
    # Overall sample sizes
    sample_sizes = []
    for h in [1, 2, 3, 4, 5]:
        df_meta = pd.read_excel(RESULTS_DIR / f'02a_H{h}_distributions.xlsx', sheet_name='Metadata')
        meta = df_meta.to_dict('records')[0]
        sample_sizes.append({
            'horizon': h,
            'total': meta['Total_Observations'],
            'bankrupt': meta['Bankrupt'],
            'non_bankrupt': meta['Non_Bankrupt'],
            'bankruptcy_rate': meta['Bankruptcy_Rate'],
            'highly_skewed': meta['Highly_Skewed_Count']
        })
    facts['02a']['sample_sizes'] = sample_sizes
    
    # Get skewness statistics from H1
    df_h1 = pd.read_excel(RESULTS_DIR / '02a_H1_distributions.xlsx', sheet_name='Summary_Statistics')
    facts['02a']['h1_extreme_skew'] = int((df_h1['Overall_Skew'].abs() > 2).sum())
    facts['02a']['h1_mean_skew'] = float(df_h1['Overall_Skew'].abs().mean())
    facts['02a']['h1_max_skew'] = float(df_h1['Overall_Skew'].abs().max())
    
    # ===== 02b: Univariate Tests =====
    print("Extracting 02b facts...")
    facts['02b'] = {}
    
    # Per-horizon statistics
    univariate_results = []
    for h in [1, 2, 3, 4, 5]:
        df_meta = pd.read_excel(RESULTS_DIR / f'02b_H{h}_univariate_tests.xlsx', sheet_name='Metadata')
        
        # Get sample sizes from 02a (same data)
        df_02a_meta = pd.read_excel(RESULTS_DIR / f'02a_H{h}_distributions.xlsx', sheet_name='Metadata')
        sample_meta = df_02a_meta.to_dict('records')[0]
        
        meta = df_meta.to_dict('records')[0]
        univariate_results.append({
            'horizon': h,
            'total_features': meta['Total_Features'],
            'significant_p05': meta['Significant_p05'],
            'significant_fdr_q05': meta['Significant_FDR_q05'],
            'lost_after_fdr': meta['Lost_after_FDR'],
            'parametric_tests': meta['Parametric_Tests'],
            'nonparametric_tests': meta['NonParametric_Tests'],
            'bankrupt_n': sample_meta['Bankrupt'],
            'non_bankrupt_n': sample_meta['Non_Bankrupt']
        })
    facts['02b']['per_horizon'] = univariate_results
    
    # Overall statistics
    total_sig_p = sum(r['significant_p05'] for r in univariate_results)
    total_sig_fdr = sum(r['significant_fdr_q05'] for r in univariate_results)
    total_tests = sum(r['total_features'] for r in univariate_results)
    
    facts['02b']['overall'] = {
        'total_tests': total_tests,
        'total_significant_p05': total_sig_p,
        'total_significant_fdr_q05': total_sig_fdr,
        'total_lost_after_fdr': total_sig_p - total_sig_fdr,
        'percent_sig_p05': round(100 * total_sig_p / total_tests, 1),
        'percent_sig_fdr_q05': round(100 * total_sig_fdr / total_tests, 1)
    }
    
    # Effect size examples from H1
    df_h1 = pd.read_excel(RESULTS_DIR / '02b_H1_univariate_tests.xlsx', sheet_name='All_Results')
    df_h1_sig = df_h1[df_h1['Significant_FDR_q05'] == True].copy()
    
    facts['02b']['h1_top_features'] = []
    for _, row in df_h1_sig.head(5).iterrows():
        facts['02b']['h1_top_features'].append({
            'feature': row['Feature'],
            'test': row['Test'],
            'p_value': float(row['P_Value']),
            'q_value_fdr': float(row['Q_Value_FDR']),
            'effect_size': float(row['Effect_Size']),
            'effect_type': row['Effect_Type'],
            'effect_interp': row['Effect_Interp']
        })
    
    # ===== 02c: Correlation Analysis =====
    print("Extracting 02c facts...")
    facts['02c'] = {}
    
    # Per-horizon correlation statistics
    correlation_results = []
    for h in [1, 2, 3, 4, 5]:
        df_meta = pd.read_excel(RESULTS_DIR / f'02c_H{h}_correlation.xlsx', sheet_name='Metadata')
        meta = df_meta.to_dict('records')[0]
        
        # Calculate economically_plausible from total - implausible
        total_features = meta['Total_Features']
        implausible = meta['Economically_Implausible']
        
        correlation_results.append({
            'horizon': h,
            'high_correlations': meta['High_Correlations'],
            'economically_plausible': total_features - implausible,
            'economically_implausible': implausible
        })
    facts['02c']['per_horizon'] = correlation_results
    
    # Economic validation details from H1
    df_econ = pd.read_excel(RESULTS_DIR / '02c_H1_correlation.xlsx', sheet_name='Economic_Validation')
    facts['02c']['h1_category_distribution'] = df_econ['Category'].value_counts().to_dict()
    
    # Top implausible features
    df_implausible = df_econ[df_econ['Economically_Plausible'] == False].copy()
    df_implausible = df_implausible.sort_values('Correlation_with_Bankruptcy', key=abs, ascending=False)
    
    facts['02c']['h1_top_implausible'] = []
    for _, row in df_implausible.head(5).iterrows():
        facts['02c']['h1_top_implausible'].append({
            'feature': row['Feature'],
            'category': row['Category'],
            'correlation': float(row['Correlation_with_Bankruptcy']),
            'expected_direction': row['Expected_Direction']
        })
    
    return facts


if __name__ == '__main__':
    print("=" * 60)
    print("EXTRACTING VERIFIED FACTS FROM PHASE 02 RESULTS")
    print("=" * 60)
    
    facts = extract_facts()
    
    # Save to JSON
    output_file = PROJECT_ROOT / 'scripts' / 'paper_helper' / 'phase02_facts.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(facts, f, indent=2)
    
    print(f"\n✓ Facts saved to: {output_file}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Print summary
    print(f"\n02a: Distribution Analysis")
    print(f"  Total observations: {sum(s['total'] for s in facts['02a']['sample_sizes']):,}")
    print(f"  Horizons analyzed: {len(facts['02a']['sample_sizes'])}")
    
    print(f"\n02b: Univariate Tests")
    print(f"  Total tests: {facts['02b']['overall']['total_tests']}")
    print(f"  Significant (p<0.05): {facts['02b']['overall']['total_significant_p05']} ({facts['02b']['overall']['percent_sig_p05']}%)")
    print(f"  Significant (FDR q<0.05): {facts['02b']['overall']['total_significant_fdr_q05']} ({facts['02b']['overall']['percent_sig_fdr_q05']}%)")
    print(f"  Lost after FDR: {facts['02b']['overall']['total_lost_after_fdr']}")
    
    print(f"\n02c: Correlation Analysis")
    avg_high_corr = sum(r['high_correlations'] for r in facts['02c']['per_horizon']) / 5
    avg_implausible = sum(r['economically_implausible'] for r in facts['02c']['per_horizon']) / 5
    print(f"  Avg high correlations (|r|>0.7): {avg_high_corr:.1f}")
    print(f"  Avg economically implausible: {avg_implausible:.1f}/64 ({100*avg_implausible/64:.1f}%)")
    
    print("\n✅ All facts extracted and verified!")
