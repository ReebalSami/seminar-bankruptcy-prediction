#!/usr/bin/env python3
"""
Script 10c: Complete Econometric Diagnostics
Based on course materials in econometrics-materials/

Tests ALL OLS assumptions from the course:
1. Durbin-Watson (autocorrelation AR(1))
2. Breusch-Pagan (heteroscedasticity)
3. Jarque-Bera (normality)
4. Breusch-Godfrey (higher-order autocorrelation)
5. Box-Pierce/Ljung-Box (autocorrelation)
6. Plus existing: VIF, Hosmer-Lemeshow, Cook's D, EPV
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("COMPLETE ECONOMETRIC DIAGNOSTICS")
print("=" * 70)

# Setup
data_dir = project_root / 'data' / 'processed'
output_dir = project_root / 'results' / 'script_outputs' / '10c_complete_diagnostics'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# Load Polish H1 data
print("\n[1/8] Loading Polish Horizon 1 data...")
df = pd.read_parquet(data_dir / 'poland_clean_full.parquet')
df_h1 = df[df['horizon'] == 1].copy()

# Prepare data
y = df_h1['y']
feature_cols = [col for col in df_h1.columns 
                if col.startswith('Attr') and not col.endswith('__isna')]
X = df_h1[feature_cols]

print(f"  Samples: {len(df_h1)}")
print(f"  Features: {len(feature_cols)}")
print(f"  Bankruptcy rate: {y.mean():.2%}")

# Remove constant columns
feature_std = X.std()
non_constant = feature_std[feature_std > 0].index.tolist()
X = X[non_constant]
print(f"  Non-constant features: {len(non_constant)}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# Fit logistic regression
print("\n[2/8] Fitting logistic regression model...")
X_with_const = sm.add_constant(X_scaled_df)
model = sm.Logit(y, X_with_const).fit(disp=0)

print(f"  Converged: {model.mle_retvals['converged']}")
print(f"  Log-likelihood: {model.llf:.2f}")
print(f"  Pseudo R²: {model.prsquared:.4f}")

# Get residuals and predictions
y_pred_proba = model.predict(X_with_const)
# For logistic regression, we use Pearson residuals
# Clip predictions to avoid division by zero
y_pred_proba_clip = np.clip(y_pred_proba, 1e-7, 1-1e-7)
residuals_pearson = (y.values - y_pred_proba_clip) / np.sqrt(y_pred_proba_clip * (1 - y_pred_proba_clip))

# Initialize results dictionary
results = {}

# ============================================================================
# TEST 1: Durbin-Watson Test (Autocorrelation AR(1))
# ============================================================================
print("\n[3/8] Durbin-Watson Test (Autocorrelation AR(1))...")
# Formula from PDF01 line 1345-1366:
# dw = Σ(ût - ût-1)² / Σû²t ≈ 2(1 - ρ̂₁)

dw_statistic = durbin_watson(residuals_pearson)

# Interpretation
# dw ≈ 2 → no autocorrelation
# dw < 2 → positive autocorrelation
# dw > 2 → negative autocorrelation
# Critical values depend on n and k, but rule of thumb:
# dw < 1.5 or dw > 2.5 → potential problem

if 1.5 <= dw_statistic <= 2.5:
    dw_result = "PASS"
    dw_interpretation = "No significant autocorrelation"
elif dw_statistic < 1.5:
    dw_result = "FAIL"
    dw_interpretation = "Positive autocorrelation detected"
else:
    dw_result = "FAIL"
    dw_interpretation = "Negative autocorrelation detected"

results['durbin_watson'] = {
    'statistic': float(dw_statistic),
    'result': dw_result,
    'interpretation': dw_interpretation,
    'note': 'dw ≈ 2 indicates no autocorrelation'
}

print(f"  DW statistic: {dw_statistic:.4f}")
print(f"  Result: {dw_result} - {dw_interpretation}")

# ============================================================================
# TEST 2: Breusch-Pagan Test (Heteroscedasticity)
# ============================================================================
print("\n[4/8] Breusch-Pagan Test (Heteroscedasticity)...")
# Formula from PDF01 line 1289-1309:
# LM(H) = 0.5 · f̂'f̂ ~ χ²(r)
# For logistic regression, test if squared residuals correlate with predictors

resid_squared = residuals_pearson**2
# Regress squared residuals on X
aux_model = sm.OLS(resid_squared, X_with_const).fit()
bp_stat = len(resid_squared) * aux_model.rsquared
bp_pvalue = 1 - stats.chi2.cdf(bp_stat, len(feature_cols))

bp_result = "PASS" if bp_pvalue > 0.05 else "FAIL"
bp_interpretation = "Homoscedastic" if bp_pvalue > 0.05 else "Heteroscedastic"

results['breusch_pagan'] = {
    'statistic': float(bp_stat),
    'p_value': float(bp_pvalue),
    'result': bp_result,
    'interpretation': bp_interpretation,
    'note': 'Tests constant error variance'
}

print(f"  LM statistic: {bp_stat:.4f}")
print(f"  p-value: {bp_pvalue:.4f}")
print(f"  Result: {bp_result} - {bp_interpretation}")

# ============================================================================
# TEST 3: Jarque-Bera Test (Normality)
# ============================================================================
print("\n[5/8] Jarque-Bera Test (Normality of residuals)...")
# Formula from PDF01 line 1440-1477:
# Z₁ = (Σû³/σ̂³)/√6
# Z₂ = [(Σû⁴/σ̂⁴) - 3]/√24
# JB = Z₁ + Z₂ ~ χ²(2)

n = len(residuals_pearson)
skewness = stats.skew(residuals_pearson)
kurtosis = stats.kurtosis(residuals_pearson)

jb_statistic = n/6 * (skewness**2 + (kurtosis**2)/4)
jb_pvalue = 1 - stats.chi2.cdf(jb_statistic, 2)

jb_result = "PASS" if jb_pvalue > 0.05 else "FAIL"
jb_interpretation = "Residuals are normal" if jb_pvalue > 0.05 else "Residuals deviate from normality"

results['jarque_bera'] = {
    'statistic': float(jb_statistic),
    'p_value': float(jb_pvalue),
    'skewness': float(skewness),
    'kurtosis': float(kurtosis),
    'result': jb_result,
    'interpretation': jb_interpretation,
    'note': 'Tests if residuals follow normal distribution'
}

print(f"  JB statistic: {jb_statistic:.4f}")
print(f"  p-value: {jb_pvalue:.4f}")
print(f"  Skewness: {skewness:.4f}")
print(f"  Kurtosis: {kurtosis:.4f}")
print(f"  Result: {jb_result} - {jb_interpretation}")

# ============================================================================
# TEST 4: Breusch-Godfrey Test (Higher-order autocorrelation)
# ============================================================================
print("\n[6/8] Breusch-Godfrey Test (Autocorrelation up to lag 5)...")
# Formula from PDF01 line 1375-1389:
# LM(A) = n · R²û ~ χ²(m)
# For logistic regression, manually implement the test

nlags = 5
# Create lagged residuals
resid_df = pd.DataFrame({'resid': residuals_pearson}, index=X_scaled_df.index)
for lag in range(1, nlags + 1):
    resid_df[f'resid_lag{lag}'] = resid_df['resid'].shift(lag)
resid_df = resid_df.dropna()

# Regress residuals on X and lagged residuals
X_with_lags = pd.concat([
    X_scaled_df.loc[resid_df.index],
    resid_df[[f'resid_lag{i}' for i in range(1, nlags+1)]]
], axis=1)
X_with_lags = sm.add_constant(X_with_lags)

aux_bg_model = sm.OLS(resid_df['resid'].values, X_with_lags).fit()
bg_stat = len(resid_df) * aux_bg_model.rsquared
bg_pvalue = 1 - stats.chi2.cdf(bg_stat, nlags)

bg_result = "PASS" if bg_pvalue > 0.05 else "FAIL"
bg_interpretation = "No autocorrelation" if bg_pvalue > 0.05 else "Autocorrelation detected"

results['breusch_godfrey'] = {
    'statistic': float(bg_stat),
    'p_value': float(bg_pvalue),
    'lags_tested': 5,
    'result': bg_result,
    'interpretation': bg_interpretation,
    'note': 'Tests autocorrelation up to lag 5'
}

print(f"  LM statistic: {bg_stat:.4f}")
print(f"  p-value: {bg_pvalue:.4f}")
print(f"  Result: {bg_result} - {bg_interpretation}")

# ============================================================================
# TEST 5: Box-Pierce / Ljung-Box Test (Autocorrelation)
# ============================================================================
print("\n[7/8] Ljung-Box Test (Autocorrelation)...")
# Formula from PDF01 line 1397-1431:
# Q(m) = T(T+2) · Σ[ρ̂²k/(T-k)] ~ χ²(m)

from statsmodels.stats.diagnostic import acorr_ljungbox

lb_results = acorr_ljungbox(residuals_pearson, lags=10, return_df=True)
lb_stat = lb_results['lb_stat'].iloc[-1]  # Last lag
lb_pvalue = lb_results['lb_pvalue'].iloc[-1]

lb_result = "PASS" if lb_pvalue > 0.05 else "FAIL"
lb_interpretation = "No autocorrelation" if lb_pvalue > 0.05 else "Autocorrelation detected"

results['ljung_box'] = {
    'statistic': float(lb_stat),
    'p_value': float(lb_pvalue),
    'lags_tested': 10,
    'result': lb_result,
    'interpretation': lb_interpretation,
    'note': 'Tests autocorrelation up to lag 10'
}

print(f"  LB statistic: {lb_stat:.4f}")
print(f"  p-value: {lb_pvalue:.4f}")
print(f"  Result: {lb_result} - {lb_interpretation}")

# ============================================================================
# Combine with existing diagnostics from script 10
# ============================================================================
print("\n[8/8] Adding existing diagnostics (VIF, Hosmer-Lemeshow, etc.)...")

# Load existing diagnostics
existing_diag_file = project_root / 'results' / 'script_outputs' / '10_econometric_diagnostics' / 'diagnostics_summary.json'
if existing_diag_file.exists():
    with open(existing_diag_file) as f:
        existing_diag = json.load(f)
    
    # Add to results
    results['hosmer_lemeshow'] = existing_diag.get('hosmer_lemeshow', {})
    results['multicollinearity'] = existing_diag.get('multicollinearity', {})
    results['sample_size'] = existing_diag.get('sample_size', {})
    results['separation'] = existing_diag.get('separation', {})
    results['influential_observations'] = existing_diag.get('influential_observations', {})
    
    print("  ✓ Integrated existing diagnostics")

# ============================================================================
# Create summary
# ============================================================================
summary = {
    'test_results': results,
    'overall_assessment': {
        'autocorrelation': {
            'durbin_watson': results['durbin_watson']['result'],
            'breusch_godfrey': results['breusch_godfrey']['result'],
            'ljung_box': results['ljung_box']['result'],
            'overall': 'PASS' if all([
                results['durbin_watson']['result'] == 'PASS',
                results['breusch_godfrey']['result'] == 'PASS',
                results['ljung_box']['result'] == 'PASS'
            ]) else 'FAIL'
        },
        'heteroscedasticity': {
            'breusch_pagan': results['breusch_pagan']['result']
        },
        'normality': {
            'jarque_bera': results['jarque_bera']['result']
        },
        'multicollinearity': {
            'condition_number': results['multicollinearity']['condition_number'],
            'severe': results['multicollinearity']['condition_number'] > 1e10
        },
        'sample_size': {
            'epv': results['sample_size']['events_per_variable'],
            'adequate': results['sample_size']['events_per_variable'] >= 10
        }
    }
}

# Save results
output_file = output_dir / 'complete_diagnostics.json'
with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Complete diagnostics saved: {output_file}")

# ============================================================================
# Visualizations
# ============================================================================
print("\n[Viz] Creating diagnostic plots...")

# 1. Residual plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs fitted
axes[0, 0].scatter(y_pred_proba, residuals_pearson, alpha=0.5, s=20)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Pearson residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(residuals_pearson, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot')

# Scale-location
axes[1, 0].scatter(y_pred_proba, np.sqrt(np.abs(residuals_pearson)), alpha=0.5, s=20)
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('√|Standardized residuals|')
axes[1, 0].set_title('Scale-Location')

# Residual histogram
axes[1, 1].hist(residuals_pearson, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Pearson residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Residual Distribution (Skew={skewness:.2f}, Kurt={kurtosis:.2f})')

plt.tight_layout()
plt.savefig(figures_dir / 'residual_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Autocorrelation function plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(residuals_pearson, lags=20, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(residuals_pearson, lags=20, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.savefig(figures_dir / 'autocorrelation_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Test results summary
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

test_summary = f"""
COMPLETE ECONOMETRIC DIAGNOSTICS SUMMARY
{'=' * 50}

AUTOCORRELATION TESTS:
  Durbin-Watson:      {results['durbin_watson']['statistic']:.4f} → {results['durbin_watson']['result']}
  Breusch-Godfrey:    p={results['breusch_godfrey']['p_value']:.4f} → {results['breusch_godfrey']['result']}
  Ljung-Box:          p={results['ljung_box']['p_value']:.4f} → {results['ljung_box']['result']}

HETEROSCEDASTICITY TEST:
  Breusch-Pagan:      p={results['breusch_pagan']['p_value']:.4f} → {results['breusch_pagan']['result']}

NORMALITY TEST:
  Jarque-Bera:        p={results['jarque_bera']['p_value']:.4f} → {results['jarque_bera']['result']}
  Skewness:           {results['jarque_bera']['skewness']:.4f}
  Kurtosis:           {results['jarque_bera']['kurtosis']:.4f}

MULTICOLLINEARITY:
  Condition Number:   {results['multicollinearity']['condition_number']:.2e}
  Status:             {'SEVERE' if results['multicollinearity']['condition_number'] > 1e10 else 'OK'}

SAMPLE SIZE:
  Events/Variable:    {results['sample_size']['events_per_variable']:.2f}
  Status:             {'ADEQUATE' if results['sample_size']['events_per_variable'] >= 10 else 'LOW'}

OVERALL ASSESSMENT:
  ⚠️  Issues detected - remediation required
"""

ax.text(0.1, 0.5, test_summary, fontsize=11, family='monospace',
        verticalalignment='center', transform=ax.transData)
plt.tight_layout()
plt.savefig(figures_dir / 'test_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Diagnostic plots saved")

# ============================================================================
# Final summary
# ============================================================================
print("\n" + "=" * 70)
print("✓ COMPLETE DIAGNOSTICS FINISHED")
print("=" * 70)

print("\n📊 RESULTS SUMMARY:")
print(f"  Autocorrelation:     {summary['overall_assessment']['autocorrelation']['overall']}")
print(f"  Heteroscedasticity:  {results['breusch_pagan']['result']}")
print(f"  Normality:           {results['jarque_bera']['result']}")
print(f"  Multicollinearity:   {'SEVERE' if summary['overall_assessment']['multicollinearity']['severe'] else 'OK'}")
print(f"  Sample Size:         {'ADEQUATE' if summary['overall_assessment']['sample_size']['adequate'] else 'LOW'}")

issues = []
if summary['overall_assessment']['autocorrelation']['overall'] == 'FAIL':
    issues.append('Autocorrelation')
if results['breusch_pagan']['result'] == 'FAIL':
    issues.append('Heteroscedasticity')
if results['jarque_bera']['result'] == 'FAIL':
    issues.append('Non-normality')
if summary['overall_assessment']['multicollinearity']['severe']:
    issues.append('Multicollinearity')
if not summary['overall_assessment']['sample_size']['adequate']:
    issues.append('Low EPV')

if issues:
    print(f"\n⚠️  ISSUES DETECTED: {', '.join(issues)}")
    print("   → Proceed to Script 10d for remediation")
else:
    print("\n✓ All tests passed - model assumptions satisfied")

print("\n" + "=" * 70)
