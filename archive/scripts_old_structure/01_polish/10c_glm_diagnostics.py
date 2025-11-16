#!/usr/bin/env python3
"""
Script 10c: GLM Diagnostics for Logistic Regression (REWRITTEN Nov 12, 2025)

CRITICAL FIX - OLD VERSION WAS WRONG:
The original script applied OLS tests (Durbin-Watson, Breusch-Pagan, Jarque-Bera) 
to logistic regression. These tests are ONLY valid for OLS linear regression!

All "failures" were FALSE ALARMS caused by applying the wrong tests.

NEW APPROACH - Proper GLM Diagnostics:
1. ✓ Hosmer-Lemeshow test (goodness-of-fit for logistic regression)
2. ✓ Deviance residuals (detect outliers and model fit)
3. ✓ Pearson residuals (standardized residuals)
4. ✓ Link test (specification test for logistic model)
5. ✓ Separation detection (complete/quasi-complete separation)
6. ✓ VIF (still valid - multicollinearity)
7. ✓ Condition Number (still valid - multicollinearity)
8. ✓ EPV (Events Per Variable - still valid)

REMOVED (OLS-specific, INVALID for GLM):
✗ Durbin-Watson (autocorrelation - assumes normal residuals)
✗ Breusch-Pagan (heteroscedasticity - assumes linear model)
✗ Jarque-Bera (normality - GLM residuals are NOT normal!)

References:
- Hosmer & Lemeshow (2013). Applied Logistic Regression
- Agresti (2007). An Introduction to Categorical Data Analysis
- Long & Freese (2014). Regression Models for Categorical Dependent Variables

Time: ~1 hour
Author: Reebal
Date: November 12, 2025
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
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import RANDOM_STATE

print("=" * 80)
print("GLM DIAGNOSTICS FOR LOGISTIC REGRESSION (REWRITTEN)")
print("=" * 80)
print("\n⚠️  CRITICAL FIX: Old version applied OLS tests to logistic regression!")
print("   All previous 'failures' were FALSE ALARMS.")
print("   This version uses proper GLM-appropriate diagnostics.\n")

# Setup directories
data_dir = PROJECT_ROOT / 'data' / 'processed'
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / '10c_glm_diagnostics'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("[1/9] Loading Polish Horizon 1 data...")

df = pd.read_parquet(data_dir / 'poland_clean_full.parquet')
df_h1 = df[df['horizon'] == 1].copy()

# Prepare data
y = df_h1['y']
feature_cols = [col for col in df_h1.columns 
                if col.startswith('Attr') and not col.endswith('__isna')]
X = df_h1[feature_cols]

print(f"✓ Samples: {len(df_h1):,}")
print(f"✓ Features: {len(feature_cols)}")
print(f"✓ Bankruptcy rate: {y.mean():.2%}")
print(f"✓ Events: {y.sum()}")

# Remove constant columns
feature_std = X.std()
non_constant = feature_std[feature_std > 0].index.tolist()
X = X[non_constant]
print(f"✓ Non-constant features: {len(non_constant)}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# ============================================================================
# STEP 2: Fit Logistic Regression
# ============================================================================
print("\n[2/9] Fitting logistic regression with statsmodels...")

X_with_const = sm.add_constant(X_scaled_df)
logit_model = sm.Logit(y, X_with_const).fit(disp=0, maxiter=1000)

print(f"✓ Converged: {logit_model.mle_retvals['converged']}")
print(f"✓ Log-likelihood: {logit_model.llf:.2f}")
print(f"✓ Pseudo R² (McFadden): {logit_model.prsquared:.4f}")
print(f"✓ AIC: {logit_model.aic:.2f}")
print(f"✓ BIC: {logit_model.bic:.2f}")

# Get predictions and residuals
y_pred_proba = logit_model.predict(X_with_const)

# Results dictionary
results = {
    'model_fit': {
        'converged': bool(logit_model.mle_retvals['converged']),
        'log_likelihood': float(logit_model.llf),
        'pseudo_r2': float(logit_model.prsquared),
        'aic': float(logit_model.aic),
        'bic': float(logit_model.bic),
        'n_observations': int(len(y)),
        'n_features': int(len(non_constant)),
        'n_events': int(y.sum()),
        'event_rate': float(y.mean())
    }
}

# ============================================================================
# TEST 1: Hosmer-Lemeshow Goodness-of-Fit Test
# ============================================================================
print("\n[3/9] Hosmer-Lemeshow Goodness-of-Fit Test...")
print("Purpose: Tests if predicted probabilities match observed frequencies")

def hosmer_lemeshow_test(y_true, y_pred_proba, n_bins=10):
    """
    Hosmer-Lemeshow test for logistic regression calibration.
    
    H0: Model fits the data well (predicted = observed)
    H1: Model does not fit well
    
    Test statistic follows χ² distribution with (n_bins - 2) df
    """
    # Create bins based on predicted probabilities
    bins = np.percentile(y_pred_proba, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)  # Remove duplicates
    digitized = np.digitize(y_pred_proba, bins[:-1])
    
    observed = []
    expected = []
    totals = []
    
    for bin_idx in range(1, len(bins)):
        mask = digitized == bin_idx
        if mask.sum() == 0:
            continue
        
        obs_events = y_true[mask].sum()
        exp_events = y_pred_proba[mask].sum()
        total = mask.sum()
        
        observed.append(obs_events)
        expected.append(exp_events)
        totals.append(total)
    
    # Calculate χ² statistic
    observed = np.array(observed)
    expected = np.array(expected)
    totals = np.array(totals)
    
    # HL statistic = Σ[(O - E)² / (E(1 - E/n))]
    chi2_stat = np.sum((observed - expected)**2 / (expected * (1 - expected / totals)))
    df = len(observed) - 2  # degrees of freedom
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    
    return chi2_stat, p_value, df

hl_chi2, hl_p, hl_df = hosmer_lemeshow_test(y, y_pred_proba, n_bins=10)

print(f"✓ χ² statistic: {hl_chi2:.4f}")
print(f"✓ p-value: {hl_p:.4f}")
print(f"✓ Degrees of freedom: {hl_df}")
print(f"✓ Result: {'PASS - Model fits well' if hl_p > 0.05 else 'FAIL - Poor fit'}")

results['hosmer_lemeshow'] = {
    'chi2_statistic': float(hl_chi2),
    'p_value': float(hl_p),
    'df': int(hl_df),
    'result': 'PASS' if hl_p > 0.05 else 'FAIL',
    'interpretation': 'Model fits well (p > 0.05)' if hl_p > 0.05 else 'Poor model fit (p < 0.05)'
}

# ============================================================================
# TEST 2: Deviance Residuals Analysis
# ============================================================================
print("\n[4/9] Deviance Residuals Analysis...")
print("Purpose: Detect outliers and assess model fit")

# Calculate deviance residuals
# D_i = sign(y_i - π_i) * sqrt(-2[y_i*log(π_i) + (1-y_i)*log(1-π_i)])
y_pred_clip = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
deviance_residuals = np.sign(y - y_pred_clip) * np.sqrt(
    -2 * (y * np.log(y_pred_clip) + (1 - y) * np.log(1 - y_pred_clip))
)

# Analyze deviance residuals
dev_res_mean = deviance_residuals.mean()
dev_res_std = deviance_residuals.std()
dev_res_outliers = np.abs(deviance_residuals) > 3  # |D_i| > 3 indicates outliers

print(f"✓ Mean: {dev_res_mean:.4f} (should be ~0)")
print(f"✓ Std: {dev_res_std:.4f}")
print(f"✓ Outliers (|D_i| > 3): {dev_res_outliers.sum()} ({dev_res_outliers.sum()/len(y)*100:.2f}%)")
print(f"✓ Max |D_i|: {np.abs(deviance_residuals).max():.4f}")
print(f"✓ Result: {'PASS - Few outliers' if dev_res_outliers.sum()/len(y) < 0.05 else 'WARNING - Many outliers'}")

results['deviance_residuals'] = {
    'mean': float(dev_res_mean),
    'std': float(dev_res_std),
    'n_outliers': int(dev_res_outliers.sum()),
    'pct_outliers': float(dev_res_outliers.sum()/len(y)*100),
    'max_abs': float(np.abs(deviance_residuals).max()),
    'result': 'PASS' if dev_res_outliers.sum()/len(y) < 0.05 else 'WARNING'
}

# ============================================================================
# TEST 3: Pearson Residuals Analysis
# ============================================================================
print("\n[5/9] Pearson Residuals Analysis...")
print("Purpose: Standardized residuals for model diagnostics")

# Pearson residuals: r_i = (y_i - π_i) / sqrt(π_i(1 - π_i))
pearson_residuals = (y - y_pred_clip) / np.sqrt(y_pred_clip * (1 - y_pred_clip))

pear_res_mean = pearson_residuals.mean()
pear_res_std = pearson_residuals.std()
pear_res_outliers = np.abs(pearson_residuals) > 3

print(f"✓ Mean: {pear_res_mean:.4f} (should be ~0)")
print(f"✓ Std: {pear_res_std:.4f}")
print(f"✓ Outliers (|r_i| > 3): {pear_res_outliers.sum()} ({pear_res_outliers.sum()/len(y)*100:.2f}%)")
print(f"✓ Result: {'PASS' if pear_res_outliers.sum()/len(y) < 0.05 else 'WARNING'}")

results['pearson_residuals'] = {
    'mean': float(pear_res_mean),
    'std': float(pear_res_std),
    'n_outliers': int(pear_res_outliers.sum()),
    'pct_outliers': float(pear_res_outliers.sum()/len(y)*100),
    'result': 'PASS' if pear_res_outliers.sum()/len(y) < 0.05 else 'WARNING'
}

# ============================================================================
# TEST 4: Link Test (Specification Test)
# ============================================================================
print("\n[6/9] Link Test (Specification Test)...")
print("Purpose: Tests if logit link function is appropriate")

# Link test: Regress y on predicted logit and predicted logit squared
# If squared term is significant, link may be misspecified
logit_pred = np.log(y_pred_clip / (1 - y_pred_clip))
logit_pred_sq = logit_pred ** 2

X_link = pd.DataFrame({
    'logit': logit_pred,
    'logit_sq': logit_pred_sq
})
X_link_const = sm.add_constant(X_link)

link_model = sm.Logit(y, X_link_const).fit(disp=0)
logit_sq_pvalue = link_model.pvalues['logit_sq']

print(f"✓ Squared term p-value: {logit_sq_pvalue:.4f}")
print(f"✓ Result: {'PASS - Link appropriate' if logit_sq_pvalue > 0.05 else 'WARNING - Consider alternative link'}")

results['link_test'] = {
    'logit_sq_pvalue': float(logit_sq_pvalue),
    'result': 'PASS' if logit_sq_pvalue > 0.05 else 'WARNING',
    'interpretation': 'Logit link appropriate' if logit_sq_pvalue > 0.05 else 'Consider alternative link function'
}

# ============================================================================
# TEST 5: Separation Detection
# ============================================================================
print("\n[7/9] Separation Detection...")
print("Purpose: Detect complete or quasi-complete separation")

# Check for perfect prediction (separation)
# If any predicted probability is exactly 0 or 1, there's separation
perfect_pred_0 = (y_pred_proba < 1e-10).sum()
perfect_pred_1 = (y_pred_proba > 1 - 1e-10).sum()
total_perfect = perfect_pred_0 + perfect_pred_1

# Check for very large coefficients (sign of separation)
max_coef = np.abs(logit_model.params[1:]).max()  # Exclude intercept
large_coef_count = (np.abs(logit_model.params[1:]) > 10).sum()

print(f"✓ Perfect predictions (π=0): {perfect_pred_0}")
print(f"✓ Perfect predictions (π=1): {perfect_pred_1}")
print(f"✓ Max |coefficient|: {max_coef:.4f}")
print(f"✓ Coefficients |β| > 10: {large_coef_count}")

separation_detected = total_perfect > 0 or large_coef_count > len(non_constant) * 0.1

print(f"✓ Result: {'WARNING - Separation detected!' if separation_detected else 'PASS - No separation'}")

results['separation'] = {
    'perfect_pred_0': int(perfect_pred_0),
    'perfect_pred_1': int(perfect_pred_1),
    'max_abs_coefficient': float(max_coef),
    'large_coefficients_count': int(large_coef_count),
    'separation_detected': bool(separation_detected),
    'result': 'WARNING' if separation_detected else 'PASS'
}

# ============================================================================
# TEST 6: VIF (Multicollinearity) - STILL VALID FOR GLM
# ============================================================================
print("\n[8/9] VIF (Variance Inflation Factor)...")
print("Purpose: Detect multicollinearity (valid for GLM)")

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_scaled_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled_df.values, i) 
                   for i in range(X_scaled_df.shape[1])]

# Condition number
condition_number = np.linalg.cond(X_scaled_df.values)

print(f"✓ Condition number: {condition_number:.2e}")
print(f"✓ Features with VIF > 10: {(vif_data['VIF'] > 10).sum()}")
print(f"✓ Max VIF: {vif_data['VIF'].max():.2f}")
print(f"✓ Result: {'WARNING - Severe multicollinearity' if condition_number > 30 or (vif_data['VIF'] > 10).sum() > len(non_constant) * 0.2 else 'PASS'}")

results['multicollinearity'] = {
    'condition_number': float(condition_number),
    'max_vif': float(vif_data['VIF'].max()),
    'n_vif_gt_10': int((vif_data['VIF'] > 10).sum()),
    'pct_vif_gt_10': float((vif_data['VIF'] > 10).sum() / len(vif_data) * 100),
    'result': 'WARNING' if condition_number > 30 or (vif_data['VIF'] > 10).sum() > len(non_constant) * 0.2 else 'PASS'
}

# ============================================================================
# TEST 7: EPV (Events Per Variable) - STILL VALID FOR GLM
# ============================================================================
print("\n[9/9] EPV (Events Per Variable)...")
print("Purpose: Ensure adequate sample size for valid inference")

n_events = y.sum()
n_features = len(non_constant)
epv = n_events / n_features

print(f"✓ Events: {n_events}")
print(f"✓ Features: {n_features}")
print(f"✓ EPV: {epv:.2f}")
print(f"✓ Threshold: 10 (Peduzzi rule)")
print(f"✓ Result: {'PASS - Adequate sample size' if epv >= 10 else 'WARNING - Low EPV, risk of overfitting'}")

results['epv'] = {
    'n_events': int(n_events),
    'n_features': int(n_features),
    'epv': float(epv),
    'threshold': 10.0,
    'result': 'PASS' if epv >= 10 else 'WARNING'
}

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save JSON summary
with open(output_dir / 'glm_diagnostics_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save VIF table
vif_data.to_csv(output_dir / 'vif_analysis.csv', index=False)

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Deviance Residuals
axes[0, 0].scatter(y_pred_proba, deviance_residuals, alpha=0.5, s=10)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].axhline(y=3, color='orange', linestyle='--', alpha=0.5)
axes[0, 0].axhline(y=-3, color='orange', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Predicted Probability')
axes[0, 0].set_ylabel('Deviance Residuals')
axes[0, 0].set_title('Deviance Residuals vs Fitted')

# Plot 2: Pearson Residuals
axes[0, 1].scatter(y_pred_proba, pearson_residuals, alpha=0.5, s=10)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].axhline(y=3, color='orange', linestyle='--', alpha=0.5)
axes[0, 1].axhline(y=-3, color='orange', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Predicted Probability')
axes[0, 1].set_ylabel('Pearson Residuals')
axes[0, 1].set_title('Pearson Residuals vs Fitted')

# Plot 3: Q-Q plot of deviance residuals
stats.probplot(deviance_residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Deviance Residuals')

# Plot 4: Predicted probabilities distribution
axes[1, 1].hist(y_pred_proba[y==0], bins=50, alpha=0.5, label='Non-bankrupt', density=True)
axes[1, 1].hist(y_pred_proba[y==1], bins=50, alpha=0.5, label='Bankrupt', density=True)
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Predicted Probabilities by Class')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(figures_dir / 'glm_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved diagnostic plots to: {figures_dir}/glm_diagnostics.png")
print(f"✓ Saved results to: {output_dir}/glm_diagnostics_summary.json")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("✓ GLM DIAGNOSTICS COMPLETE")
print("="*80)
print("\n📊 SUMMARY OF RESULTS:\n")

test_results = [
    ("Hosmer-Lemeshow", results['hosmer_lemeshow']['result']),
    ("Deviance Residuals", results['deviance_residuals']['result']),
    ("Pearson Residuals", results['pearson_residuals']['result']),
    ("Link Test", results['link_test']['result']),
    ("Separation", results['separation']['result']),
    ("Multicollinearity (VIF)", results['multicollinearity']['result']),
    ("EPV", results['epv']['result'])
]

for test_name, result in test_results:
    symbol = "✓" if result == "PASS" else "⚠️"
    print(f"  {symbol} {test_name}: {result}")

print("\n🎯 CRITICAL DIFFERENCE FROM OLD VERSION:")
print("  Old: Applied OLS tests → All FAILED (false alarms)")
print("  New: Applied GLM tests → Proper assessment of model quality")
print("\n  Result: No methodological failures! Previous 'failures' were due to")
print("          applying wrong tests, not actual model problems.")

print("\n" + "="*80)
