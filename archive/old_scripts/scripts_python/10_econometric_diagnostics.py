#!/usr/bin/env python3
"""
Econometric Diagnostics & Assumption Testing
Rigorous testing of model assumptions for logistic regression
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.stats import chi2
import json

# Setup
project_root = Path(__file__).parent.parent
output_dir = project_root / 'results' / 'script_outputs' / '10_econometric_diagnostics'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("ECONOMETRIC DIAGNOSTICS & ASSUMPTION TESTING")
print("="*70)

# ========================================================================
# Load Polish Dataset (Horizon 1 for focused analysis)
# ========================================================================
print("\n[1/8] Loading data...")

data_dir = project_root / 'data' / 'processed'
df = pd.read_parquet(data_dir / 'poland_clean_full.parquet')

# Filter for Horizon 1
df = df[df['horizon'] == 1].copy()

# Prepare data
feature_cols = [col for col in df.columns if col.startswith('Attr') and '__isna' not in col]
X = df[feature_cols]
y = df['y']

# Remove missing and infinite values
mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1))
X = X[mask]
y = y[mask]

print(f"✓ Loaded {len(X):,} samples, {X.shape[1]} features")
print(f"  Bankruptcy rate: {y.mean()*100:.2f}%")

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
logit = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
logit.fit(X_train_scaled, y_train)

y_pred_proba = logit.predict_proba(X_test_scaled)[:, 1]
y_pred = logit.predict(X_test_scaled)

print(f"✓ Model trained")

# ========================================================================
# Test 1: Hosmer-Lemeshow Goodness of Fit
# ========================================================================
print("\n[2/8] Hosmer-Lemeshow goodness of fit test...")

def hosmer_lemeshow_test(y_true, y_pred_proba, g=10):
    """Hosmer-Lemeshow test for goodness of fit"""
    # Create deciles
    df_hl = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred_proba
    })
    df_hl['decile'] = pd.qcut(df_hl['y_pred'], q=g, duplicates='drop')
    
    # Calculate observed and expected
    grouped = df_hl.groupby('decile')
    obs_1 = grouped['y_true'].sum()
    obs_0 = grouped['y_true'].count() - obs_1
    exp_1 = grouped['y_pred'].sum()
    exp_0 = grouped['y_pred'].count() - exp_1
    
    # Chi-square statistic
    hl_stat = (((obs_1 - exp_1)**2 / exp_1) + ((obs_0 - exp_0)**2 / exp_0)).sum()
    
    # P-value (degrees of freedom = g - 2)
    p_value = 1 - chi2.cdf(hl_stat, g - 2)
    
    return hl_stat, p_value, df_hl

hl_stat, hl_p_value, df_hl = hosmer_lemeshow_test(y_test, y_pred_proba)

hl_result = "PASS" if hl_p_value > 0.05 else "FAIL"
print(f"  Hosmer-Lemeshow statistic: {hl_stat:.4f}")
print(f"  P-value: {hl_p_value:.4f}")
print(f"  Result: {hl_result} (p > 0.05 = good fit)")

# ========================================================================
# Test 2: Residual Diagnostics
# ========================================================================
print("\n[3/8] Computing residual diagnostics...")

# Pearson residuals
pearson_residuals = (y_test - y_pred_proba) / np.sqrt(y_pred_proba * (1 - y_pred_proba))

# Deviance residuals
def deviance_residuals(y_true, y_pred_proba):
    """Calculate deviance residuals"""
    y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
    dev_res = np.zeros(len(y_true))
    
    for i in range(len(y_true)):
        if y_true.iloc[i] == 1:
            dev_res[i] = np.sqrt(-2 * np.log(y_pred_proba[i]))
        else:
            dev_res[i] = -np.sqrt(-2 * np.log(1 - y_pred_proba[i]))
    
    return dev_res

dev_residuals = deviance_residuals(y_test, y_pred_proba)

print(f"  Pearson residuals - Mean: {pearson_residuals.mean():.4f}, Std: {pearson_residuals.std():.4f}")
print(f"  Deviance residuals - Mean: {dev_residuals.mean():.4f}, Std: {dev_residuals.std():.4f}")

# Check for large residuals (potential outliers)
large_pearson = (np.abs(pearson_residuals) > 3).sum()
large_deviance = (np.abs(dev_residuals) > 3).sum()

print(f"  Large Pearson residuals (|r| > 3): {large_pearson} ({large_pearson/len(y_test)*100:.2f}%)")
print(f"  Large Deviance residuals (|r| > 3): {large_deviance} ({large_deviance/len(y_test)*100:.2f}%)")

# ========================================================================
# Test 3: Influential Observations (Cook's Distance)
# ========================================================================
print("\n[4/8] Computing influential observations...")

# Calculate leverage (hat values)
from numpy.linalg import inv

X_with_intercept = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
H = X_with_intercept @ inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
leverage = np.diag(H)

# Calculate Cook's distance (approximation)
# Cook's D = (residuals^2 / p) * (leverage / (1-leverage)^2)
p = X_train_scaled.shape[1]
y_train_pred = logit.predict_proba(X_train_scaled)[:, 1]
train_pearson = (y_train - y_train_pred) / np.sqrt(y_train_pred * (1 - y_train_pred))

cooks_d = (train_pearson**2 / p) * (leverage / (1 - leverage)**2)

# Threshold: 4/n
threshold = 4 / len(X_train)
influential = cooks_d > threshold

print(f"  Average leverage: {leverage.mean():.4f}")
print(f"  Max Cook's distance: {cooks_d.max():.4f}")
print(f"  Influential observations (D > 4/n = {threshold:.4f}): {influential.sum()} ({influential.sum()/len(X_train)*100:.2f}%)")

# ========================================================================
# Test 4: Specification Tests (Ramsey RESET-like)
# ========================================================================
print("\n[5/8] Specification tests...")

# Add squared predictions to test for omitted non-linearity
X_train_augmented = np.column_stack([X_train_scaled, y_train_pred**2, y_train_pred**3])

logit_augmented = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
logit_augmented.fit(X_train_augmented, y_train)

# Compare log-likelihoods (pseudo R-squared change)
from sklearn.metrics import log_loss

ll_original = -log_loss(y_train, y_train_pred, normalize=False)
y_train_pred_aug = logit_augmented.predict_proba(X_train_augmented)[:, 1]
ll_augmented = -log_loss(y_train, y_train_pred_aug, normalize=False)

# Likelihood ratio test
lr_stat = 2 * (ll_augmented - ll_original)
lr_p_value = 1 - chi2.cdf(lr_stat, 2)  # 2 additional parameters

spec_result = "PASS" if lr_p_value > 0.05 else "POTENTIAL ISSUE"
print(f"  Log-likelihood original: {ll_original:.2f}")
print(f"  Log-likelihood augmented: {ll_augmented:.2f}")
print(f"  LR statistic: {lr_stat:.4f}, p-value: {lr_p_value:.4f}")
print(f"  Result: {spec_result} (p > 0.05 = specification OK)")

# ========================================================================
# Test 5: Multicollinearity (Condition Number)
# ========================================================================
print("\n[6/8] Multicollinearity checks...")

# Condition number
cond_number = np.linalg.cond(X_train_scaled)

print(f"  Condition number: {cond_number:.2f}")
if cond_number < 30:
    print(f"  Result: GOOD (< 30)")
elif cond_number < 100:
    print(f"  Result: MODERATE (30-100)")
else:
    print(f"  Result: SEVERE (> 100)")

# ========================================================================
# Test 6: Sample Size Adequacy
# ========================================================================
print("\n[7/8] Sample size adequacy...")

events = y_train.sum()
n_predictors = X_train.shape[1]
events_per_variable = events / n_predictors

print(f"  Events (bankruptcies): {events}")
print(f"  Predictors: {n_predictors}")
print(f"  Events per variable (EPV): {events_per_variable:.2f}")

if events_per_variable >= 10:
    print(f"  Result: EXCELLENT (EPV ≥ 10)")
elif events_per_variable >= 5:
    print(f"  Result: ACCEPTABLE (EPV ≥ 5)")
else:
    print(f"  Result: INSUFFICIENT (EPV < 5, consider feature reduction)")

# ========================================================================
# Test 7: Complete Separation Check
# ========================================================================
print("\n[8/8] Complete separation check...")

# Check for perfect prediction patterns
perfect_pred_0 = ((y_train_pred < 0.01) & (y_train == 0)).sum()
perfect_pred_1 = ((y_train_pred > 0.99) & (y_train == 1)).sum()

separation_ratio = (perfect_pred_0 + perfect_pred_1) / len(y_train)

print(f"  Perfect predictions for class 0: {perfect_pred_0}")
print(f"  Perfect predictions for class 1: {perfect_pred_1}")
print(f"  Separation ratio: {separation_ratio*100:.2f}%")

if separation_ratio < 0.01:
    print(f"  Result: NO SEPARATION ISSUE")
else:
    print(f"  Result: POTENTIAL SEPARATION (check coefficients)")

# ========================================================================
# Save Results
# ========================================================================
print("\nSaving results...")

# Summary JSON
diagnostics_summary = {
    'hosmer_lemeshow': {
        'statistic': float(hl_stat),
        'p_value': float(hl_p_value),
        'result': hl_result
    },
    'residuals': {
        'pearson_mean': float(pearson_residuals.mean()),
        'pearson_std': float(pearson_residuals.std()),
        'deviance_mean': float(dev_residuals.mean()),
        'deviance_std': float(dev_residuals.std()),
        'large_pearson_count': int(large_pearson),
        'large_deviance_count': int(large_deviance)
    },
    'influential_observations': {
        'avg_leverage': float(leverage.mean()),
        'max_cooks_d': float(cooks_d.max()),
        'influential_count': int(influential.sum()),
        'influential_pct': float(influential.sum()/len(X_train)*100)
    },
    'specification': {
        'll_original': float(ll_original),
        'll_augmented': float(ll_augmented),
        'lr_statistic': float(lr_stat),
        'p_value': float(lr_p_value),
        'result': spec_result
    },
    'multicollinearity': {
        'condition_number': float(cond_number)
    },
    'sample_size': {
        'events': int(events),
        'predictors': int(n_predictors),
        'events_per_variable': float(events_per_variable)
    },
    'separation': {
        'separation_ratio': float(separation_ratio)
    }
}

with open(output_dir / 'diagnostics_summary.json', 'w') as f:
    json.dump(diagnostics_summary, f, indent=2)

# ========================================================================
# Visualizations
# ========================================================================
print("Creating visualizations...")

# 1. Residual plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Pearson residuals vs fitted
axes[0, 0].scatter(y_pred_proba, pearson_residuals, alpha=0.5, s=10)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].axhline(y=3, color='orange', linestyle=':', alpha=0.5)
axes[0, 0].axhline(y=-3, color='orange', linestyle=':', alpha=0.5)
axes[0, 0].set_xlabel('Fitted Values', fontweight='bold')
axes[0, 0].set_ylabel('Pearson Residuals', fontweight='bold')
axes[0, 0].set_title('Pearson Residuals vs Fitted', fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# Deviance residuals vs fitted
axes[0, 1].scatter(y_pred_proba, dev_residuals, alpha=0.5, s=10)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Fitted Values', fontweight='bold')
axes[0, 1].set_ylabel('Deviance Residuals', fontweight='bold')
axes[0, 1].set_title('Deviance Residuals vs Fitted', fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Q-Q plot for Pearson residuals
stats.probplot(pearson_residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Pearson Residuals', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Residuals histogram
axes[1, 1].hist(pearson_residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Pearson Residuals', fontweight='bold')
axes[1, 1].set_ylabel('Frequency', fontweight='bold')
axes[1, 1].set_title('Distribution of Pearson Residuals', fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'residual_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved residual diagnostics")

# 2. Influence plot (Cook's distance)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Cook's distance
axes[0].stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt=' ')
axes[0].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold (4/n = {threshold:.4f})')
axes[0].set_xlabel('Observation', fontweight='bold')
axes[0].set_ylabel("Cook's Distance", fontweight='bold')
axes[0].set_title("Cook's Distance Plot", fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Leverage vs residuals
axes[1].scatter(leverage, train_pearson**2, alpha=0.5, s=10)
axes[1].set_xlabel('Leverage', fontweight='bold')
axes[1].set_ylabel('Squared Residuals', fontweight='bold')
axes[1].set_title('Leverage vs Squared Residuals', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'influence_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved influence diagnostics")

# 3. Hosmer-Lemeshow plot
grouped_hl = df_hl.groupby('decile').agg({
    'y_true': ['sum', 'count'],
    'y_pred': 'sum'
}).reset_index()

grouped_hl.columns = ['decile', 'observed', 'total', 'expected']
grouped_hl['observed_rate'] = grouped_hl['observed'] / grouped_hl['total']
grouped_hl['expected_rate'] = grouped_hl['expected'] / grouped_hl['total']

plt.figure(figsize=(10, 6))
x = range(len(grouped_hl))
plt.plot(x, grouped_hl['observed_rate'], 'o-', label='Observed', linewidth=2, markersize=8)
plt.plot(x, grouped_hl['expected_rate'], 's-', label='Expected', linewidth=2, markersize=8)
plt.xlabel('Decile', fontweight='bold')
plt.ylabel('Bankruptcy Rate', fontweight='bold')
plt.title(f'Hosmer-Lemeshow Test: Observed vs Expected\n(χ²={hl_stat:.2f}, p={hl_p_value:.4f})',
         fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'hosmer_lemeshow.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved Hosmer-Lemeshow plot")

# Summary table
summary_data = {
    'Test': [
        'Hosmer-Lemeshow',
        'Large Residuals',
        'Influential Obs',
        'Specification',
        'Multicollinearity',
        'Sample Size',
        'Separation'
    ],
    'Result': [
        hl_result,
        f'{large_pearson}/{len(y_test)} outliers',
        f'{influential.sum()} influential',
        spec_result,
        f'Cond# = {cond_number:.1f}',
        f'EPV = {events_per_variable:.2f}',
        'No issues' if separation_ratio < 0.01 else 'Check'
    ],
    'Status': [
        '✓' if hl_result == 'PASS' else '⚠',
        '✓' if large_pearson < len(y_test)*0.05 else '⚠',
        '✓' if influential.sum() < len(X_train)*0.05 else '⚠',
        '✓' if spec_result == 'PASS' else '⚠',
        '✓' if cond_number < 30 else '⚠',
        '✓' if events_per_variable >= 10 else '⚠',
        '✓' if separation_ratio < 0.01 else '⚠'
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_dir / 'diagnostics_summary.csv', index=False)

print("\n" + "="*70)
print("✓ ECONOMETRIC DIAGNOSTICS COMPLETE")
print("="*70)
print("\nSummary:")
for _, row in summary_df.iterrows():
    print(f"  {row['Status']} {row['Test']}: {row['Result']}")
print("\n" + "="*70)
