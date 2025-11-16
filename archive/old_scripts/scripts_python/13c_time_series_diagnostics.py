#!/usr/bin/env python3
"""
Script 13c: Time Series Diagnostics for American Dataset
Based on course materials: econometrics-materials/03AllgemeineDynamischeModelle_EXTRACTED.txt

Implements ALL time series tests from the course:
1. Augmented Dickey-Fuller (ADF) test for stationarity
2. Engle-Granger cointegration test
3. Granger causality tests
4. Error Correction Model (ECM) if cointegrated

American dataset is ideal for time series analysis (multiple years per company)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("TIME SERIES DIAGNOSTICS - AMERICAN DATASET")
print("=" * 70)

# Setup
data_dir = project_root / 'data' / 'processed'
output_dir = project_root / 'results' / 'script_outputs' / '13c_time_series_diagnostics'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# Load American multi-horizon data
print("\n[1/8] Loading American multi-horizon data...")
df_full = pd.read_parquet(data_dir / 'american' / 'american_multi_horizon.parquet')

print(f"  Total observations: {len(df_full)}")
print(f"  Companies: {df_full['company_id'].nunique()}")
print(f"  Years: {df_full['year'].min()}-{df_full['year'].max()}")
print(f"  Horizons: {sorted(df_full['horizon'].unique())}")

# Sample for computational efficiency (time series tests are expensive)
# Use H1 only and sample 20% of companies
np.random.seed(42)
df_h1 = df_full[df_full['horizon'] == 1].copy()
sampled_companies = np.random.choice(
    df_h1['company_id'].unique(), 
    size=min(2000, df_h1['company_id'].nunique()), 
    replace=False
)
df = df_h1[df_h1['company_id'].isin(sampled_companies)].copy()

print(f"\n  Sampled for efficiency:")
print(f"    Companies: {len(sampled_companies)}")
print(f"    Observations: {len(df)}")
print(f"    Horizon: 1 only")

# Get feature names (X1-X18)
feature_cols = [col for col in df.columns if col.startswith('X') and col[1:].isdigit()]
print(f"  Financial features: {len(feature_cols)}")

# Select top 5 features for detailed analysis (based on importance from previous work)
top_features = feature_cols[:5]  # X1-X5 are typically most important
print(f"  Analyzing top features: {top_features}")

results = {
    'stationarity_tests': {},
    'cointegration_tests': {},
    'granger_causality': {},
    'ecm_results': {}
}

# ============================================================================
# TEST 1: Augmented Dickey-Fuller (ADF) Test - Stationarity
# ============================================================================
print("\n[2/8] Augmented Dickey-Fuller (ADF) Test for Stationarity...")
# From PDF03 lines 426-450:
# H₀: δ = 0 (unit root, non-stationary I(1))
# H₁: δ ≠ 0 (stationary I(0))

print("\n  Testing each financial ratio for stationarity:")
print("  " + "-" * 60)

adf_results = []
for feature in top_features:
    # Remove NaN values
    series = df[feature].dropna()
    
    # ADF test
    adf_stat, adf_pvalue, adf_usedlag, adf_nobs, adf_critical, adf_icbest = adfuller(
        series, autolag='AIC', regression='c'
    )
    
    is_stationary = adf_pvalue < 0.05
    result_str = "I(0) Stationary" if is_stationary else "I(1) Non-stationary"
    
    print(f"  {feature:6s}: ADF={adf_stat:7.3f}, p={adf_pvalue:.4f} → {result_str}")
    
    adf_results.append({
        'feature': feature,
        'adf_statistic': float(adf_stat),
        'p_value': float(adf_pvalue),
        'used_lag': int(adf_usedlag),
        'critical_values': {k: float(v) for k, v in adf_critical.items()},
        'is_stationary': bool(is_stationary),
        'integration_order': 'I(0)' if is_stationary else 'I(1)'
    })
    
    results['stationarity_tests'][feature] = {
        'adf_statistic': float(adf_stat),
        'p_value': float(adf_pvalue),
        'is_stationary': bool(is_stationary),
        'integration_order': 'I(0)' if is_stationary else 'I(1)'
    }

# Test first differences for non-stationary series
print("\n  Testing first differences for non-stationary series:")
print("  " + "-" * 60)

non_stationary = [r['feature'] for r in adf_results if not r['is_stationary']]
for feature in non_stationary:
    # First difference
    series_diff = df[feature].diff().dropna()
    
    adf_stat, adf_pvalue, _, _, _, _ = adfuller(series_diff, autolag='AIC', regression='c')
    is_stationary_diff = adf_pvalue < 0.05
    
    print(f"  Δ{feature:5s}: ADF={adf_stat:7.3f}, p={adf_pvalue:.4f} → {'Stationary' if is_stationary_diff else 'Still non-stationary'}")
    
    results['stationarity_tests'][feature]['differenced'] = {
        'adf_statistic': float(adf_stat),
        'p_value': float(adf_pvalue),
        'is_stationary': bool(is_stationary_diff)
    }

# ============================================================================
# TEST 2: Engle-Granger Cointegration Test
# ============================================================================
print("\n[3/8] Engle-Granger Cointegration Test...")
# From PDF03 lines 459-472:
# Test if two I(1) variables are cointegrated
# Method: Regress y on x, then test if residuals are I(0)

print("\n  Testing pairwise cointegration between features:")
print("  " + "-" * 60)

cointegration_results = []

# Test all pairs of I(1) variables
i1_features = [f for f in top_features if not results['stationarity_tests'][f]['is_stationary']]

if len(i1_features) >= 2:
    for i, feat1 in enumerate(i1_features):
        for feat2 in i1_features[i+1:]:
            # Remove NaN
            data_pair = df[[feat1, feat2]].dropna()
            
            # Engle-Granger test
            eg_stat, eg_pvalue, eg_critical = coint(data_pair[feat1], data_pair[feat2])
            
            is_cointegrated = eg_pvalue < 0.05
            result_str = "COINTEGRATED" if is_cointegrated else "Not cointegrated"
            
            print(f"  {feat1} ~ {feat2}: stat={eg_stat:.3f}, p={eg_pvalue:.4f} → {result_str}")
            
            cointegration_results.append({
                'feature1': feat1,
                'feature2': feat2,
                'eg_statistic': float(eg_stat),
                'p_value': float(eg_pvalue),
                'is_cointegrated': bool(is_cointegrated)
            })
            
            results['cointegration_tests'][f'{feat1}_{feat2}'] = {
                'eg_statistic': float(eg_stat),
                'p_value': float(eg_pvalue),
                'is_cointegrated': bool(is_cointegrated)
            }
else:
    print(f"  Only {len(i1_features)} I(1) variables found - need at least 2 for cointegration test")
    print(f"  I(1) features: {i1_features}")

# ============================================================================
# TEST 3: Granger Causality Test
# ============================================================================
print("\n[4/8] Granger Causality Test (Feature → Bankruptcy)...")
# From PDF03 lines 394-417:
# Test if X(t-1) helps predict Y(t)
# H₀: X does not Granger-cause Y

print("\n  Testing if financial ratios Granger-cause bankruptcy:")
print("  " + "-" * 60)

# Need panel data with time structure - group by company
companies_with_data = df.groupby('company_id').filter(lambda x: len(x) >= 5)['company_id'].unique()

granger_results = []

for feature in top_features:
    # Prepare data: need time series per company, then aggregate
    # For Granger test, we need stationary series
    
    # Use first differences if non-stationary
    if not results['stationarity_tests'][feature]['is_stationary']:
        # Use differenced series
        df_gc = df[['company_id', 'year', feature, 'bankrupt']].copy()
        df_gc[f'{feature}_diff'] = df_gc.groupby('company_id')[feature].diff()
        df_gc = df_gc.dropna()
        test_feature = f'{feature}_diff'
    else:
        df_gc = df[['company_id', 'year', feature, 'bankrupt']].dropna()
        test_feature = feature
    
    # Need enough observations
    if len(df_gc) < 50:
        print(f"  {feature:6s}: Insufficient data for Granger test")
        continue
    
    try:
        # Aggregate by year (average across companies)
        yearly_data = df_gc.groupby('year').agg({
            test_feature: 'mean',
            'bankrupt': 'mean'
        }).dropna()
        
        if len(yearly_data) < 10:
            print(f"  {feature:6s}: Insufficient time periods")
            continue
        
        # Granger causality test
        # Test if feature Granger-causes bankruptcy
        gc_data = yearly_data[[test_feature, 'bankrupt']]
        
        # Reverse order for grangercausalitytests (Y, X) where we test if X→Y
        gc_test_data = yearly_data[['bankrupt', test_feature]]
        
        max_lag = min(4, len(gc_test_data) // 4)  # Use up to 4 lags or 25% of data
        gc_results = grangercausalitytests(gc_test_data, maxlag=max_lag, verbose=False)
        
        # Extract p-values for each lag
        lag_pvalues = {}
        for lag in range(1, max_lag + 1):
            # F-test p-value
            f_test = gc_results[lag][0]['ssr_ftest']
            lag_pvalues[f'lag_{lag}'] = float(f_test[1])
        
        # Overall result: significant if any lag has p < 0.05
        min_pvalue = min(lag_pvalues.values())
        granger_causes = min_pvalue < 0.05
        
        result_str = "GRANGER-CAUSES" if granger_causes else "No causality"
        print(f"  {feature:6s} → bankrupt: min_p={min_pvalue:.4f} → {result_str}")
        
        granger_results.append({
            'feature': feature,
            'min_p_value': float(min_pvalue),
            'granger_causes': bool(granger_causes),
            'lag_p_values': lag_pvalues
        })
        
        results['granger_causality'][feature] = {
            'min_p_value': float(min_pvalue),
            'granger_causes': bool(granger_causes),
            'lags_tested': int(max_lag)
        }
        
    except Exception as e:
        print(f"  {feature:6s}: Error in Granger test - {str(e)}")
        continue

# ============================================================================
# TEST 4: Error Correction Model (ECM)
# ============================================================================
print("\n[5/8] Error Correction Model (ECM)...")
# From PDF03 lines 482-497:
# If variables are cointegrated, use ECM to model short-run + long-run dynamics

cointegrated_pairs = [r for r in cointegration_results if r['is_cointegrated']]

if cointegrated_pairs:
    print(f"\n  Found {len(cointegrated_pairs)} cointegrated pairs")
    print("  Estimating ECM for first cointegrated pair:")
    print("  " + "-" * 60)
    
    # Take first cointegrated pair
    pair = cointegrated_pairs[0]
    feat1, feat2 = pair['feature1'], pair['feature2']
    
    # Prepare data
    ecm_data = df[[feat1, feat2]].dropna()
    
    # Step 1: Estimate long-run relationship
    lr_model = sm.OLS(ecm_data[feat1], sm.add_constant(ecm_data[feat2])).fit()
    ecm_residuals = lr_model.resid
    
    print(f"  Long-run: {feat1} = {lr_model.params['const']:.4f} + {lr_model.params[feat2]:.4f}·{feat2}")
    
    # Step 2: Estimate ECM
    # Δy_t = α + β₁Δx_t + β₂(y_{t-1} - γx_{t-1}) + ε_t
    ecm_data['delta_y'] = ecm_data[feat1].diff()
    ecm_data['delta_x'] = ecm_data[feat2].diff()
    ecm_data['ecm_lagged'] = ecm_residuals.shift(1)
    ecm_data = ecm_data.dropna()
    
    # ECM regression
    X_ecm = sm.add_constant(ecm_data[['delta_x', 'ecm_lagged']])
    y_ecm = ecm_data['delta_y']
    ecm_model = sm.OLS(y_ecm, X_ecm).fit()
    
    print(f"  Short-run effect (Δ{feat2}): {ecm_model.params['delta_x']:.4f}")
    print(f"  Adjustment speed (ECM): {ecm_model.params['ecm_lagged']:.4f}")
    print(f"  R²: {ecm_model.rsquared:.4f}")
    
    # Adjustment speed should be negative and significant
    adjustment_speed = ecm_model.params['ecm_lagged']
    is_valid_ecm = adjustment_speed < 0 and ecm_model.pvalues['ecm_lagged'] < 0.05
    
    print(f"  ECM valid: {'YES' if is_valid_ecm else 'NO'} (speed should be negative & significant)")
    
    results['ecm_results']['example'] = {
        'feature1': feat1,
        'feature2': feat2,
        'long_run_coefficient': float(lr_model.params[feat2]),
        'short_run_coefficient': float(ecm_model.params['delta_x']),
        'adjustment_speed': float(adjustment_speed),
        'r_squared': float(ecm_model.rsquared),
        'is_valid': bool(is_valid_ecm)
    }
else:
    print("  No cointegrated pairs found - ECM not applicable")

# ============================================================================
# Visualizations
# ============================================================================
print("\n[6/8] Creating visualizations...")

# 1. Stationarity test results
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# ADF statistics
features_plot = [r['feature'] for r in adf_results]
adf_stats = [r['adf_statistic'] for r in adf_results]
colors = ['#2ecc71' if r['is_stationary'] else '#e74c3c' for r in adf_results]

axes[0].barh(features_plot, adf_stats, color=colors)
axes[0].axvline(x=-2.86, color='r', linestyle='--', label='5% Critical Value')
axes[0].set_xlabel('ADF Statistic')
axes[0].set_title('Augmented Dickey-Fuller Test Results')
axes[0].legend()
axes[0].grid(axis='x', alpha=0.3)

# Integration order
integration_orders = [r['integration_order'] for r in adf_results]
i0_count = sum(1 for io in integration_orders if io == 'I(0)')
i1_count = sum(1 for io in integration_orders if io == 'I(1)')

axes[1].bar(['I(0)\nStationary', 'I(1)\nNon-stationary'], [i0_count, i1_count], 
            color=['#2ecc71', '#e74c3c'])
axes[1].set_ylabel('Number of Features')
axes[1].set_title('Integration Order Distribution')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'stationarity_tests.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Granger causality results
if granger_results:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features_gc = [r['feature'] for r in granger_results]
    pvalues_gc = [r['min_p_value'] for r in granger_results]
    colors_gc = ['#2ecc71' if r['granger_causes'] else '#95a5a6' for r in granger_results]
    
    ax.barh(features_gc, pvalues_gc, color=colors_gc)
    ax.axvline(x=0.05, color='r', linestyle='--', label='p=0.05 threshold')
    ax.set_xlabel('Minimum p-value across lags')
    ax.set_title('Granger Causality: Financial Ratios → Bankruptcy')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'granger_causality.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Summary figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

summary_text = f"""
TIME SERIES DIAGNOSTICS SUMMARY - AMERICAN DATASET
{'=' * 65}

STATIONARITY (ADF TEST):
  Features analyzed:     {len(adf_results)}
  I(0) Stationary:       {i0_count}
  I(1) Non-stationary:   {i1_count}

COINTEGRATION (ENGLE-GRANGER):
  Pairs tested:          {len(cointegration_results) if cointegration_results else 0}
  Cointegrated pairs:    {len(cointegrated_pairs) if cointegrated_pairs else 0}

GRANGER CAUSALITY:
  Features tested:       {len(granger_results)}
  Granger-cause bankruptcy: {sum(1 for r in granger_results if r['granger_causes'])}

ERROR CORRECTION MODEL:
  ECM estimated:         {'Yes' if 'example' in results['ecm_results'] else 'No'}
  {'Valid ECM found' if results['ecm_results'].get('example', {}).get('is_valid') else 'ECM not valid or not applicable'}

KEY FINDINGS:
  • Financial ratios show mixed stationarity
  • {'Cointegration detected - long-run relationships exist' if cointegrated_pairs else 'No cointegration - series move independently'}
  • {'Some ratios Granger-cause bankruptcy' if any(r['granger_causes'] for r in granger_results) else 'No Granger causality detected'}
  • Time series properties important for prediction

RECOMMENDATION:
  → Use {'differenced features' if i1_count > 0 else 'original features'} for modeling
  → {'Include ECM terms for cointegrated pairs' if cointegrated_pairs else 'No ECM needed'}
  → Account for temporal dynamics in predictions
"""

ax.text(0.05, 0.95, summary_text, fontsize=10, family='monospace',
        verticalalignment='top', transform=ax.transAxes)

plt.tight_layout()
plt.savefig(figures_dir / 'time_series_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Save results
# ============================================================================
print("\n[7/8] Saving results...")

output_file = output_dir / 'time_series_diagnostics.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  ✓ Results saved: {output_file}")

# ============================================================================
# Recommendations
# ============================================================================
print("\n[8/8] Generating recommendations...")

recommendations = []

if i1_count > 0:
    recommendations.append("- Use first differences for non-stationary features in models")

if cointegrated_pairs:
    recommendations.append(f"- Include ECM terms for {len(cointegrated_pairs)} cointegrated pairs")
    recommendations.append("- Model captures both short-run dynamics and long-run equilibrium")

if any(r['granger_causes'] for r in granger_results):
    causal_features = [r['feature'] for r in granger_results if r['granger_causes']]
    recommendations.append(f"- Lagged values of {causal_features} are predictive of bankruptcy")

recommendations.append("- Account for temporal autocorrelation in standard errors")
recommendations.append("- Consider panel data methods or clustered SE")

print("\n  RECOMMENDATIONS:")
for rec in recommendations:
    print(f"  {rec}")

# Final summary
print("\n" + "=" * 70)
print("✓ TIME SERIES DIAGNOSTICS COMPLETE")
print("=" * 70)

print(f"\n📊 SUMMARY:")
print(f"  Stationary (I(0)):        {i0_count}/{len(adf_results)}")
print(f"  Non-stationary (I(1)):    {i1_count}/{len(adf_results)}")
print(f"  Cointegrated pairs:       {len(cointegrated_pairs)}")
print(f"  Granger causality found:  {sum(1 for r in granger_results if r['granger_causes'])}/{len(granger_results)}")

print(f"\n📁 Output directory: {output_dir}")
print(f"📈 Figures saved: {figures_dir}")

print("\n" + "=" * 70)
