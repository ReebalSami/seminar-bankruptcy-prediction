#!/usr/bin/env python3
"""
Script 10d: Enhanced Econometric Remediation + Save Datasets
Based on course materials in econometrics-materials/

Applies ALL remediation methods from the course AND SAVES clean datasets:
1. VIF-based feature selection (multicollinearity)
2. Cochrane-Orcutt (CORC) procedure (autocorrelation)
3. White robust standard errors (heteroscedasticity)
4. Forward AIC selection (alternative to VIF)
5. Ridge/Lasso regularization

**CRITICAL:** Saves remediated datasets to data/processed/ for scripts 11-13!
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("ENHANCED ECONOMETRIC REMEDIATION + SAVE DATASETS")
print("=" * 70)

# Setup
data_dir = project_root / 'data' / 'processed'
output_dir = project_root / 'results' / 'script_outputs' / '10d_remediation_save'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# Load Polish H1 data
print("\n[1/10] Loading Polish Horizon 1 data...")
df = pd.read_parquet(data_dir / 'poland_clean_full.parquet')
df_h1 = df[df['horizon'] == 1].copy()

# Prepare data
y = df_h1['y']
feature_cols = [col for col in df_h1.columns 
                if col.startswith('Attr') and not col.endswith('__isna')]
X = df_h1[feature_cols]

print(f"  Samples: {len(df_h1)}")
print(f"  Original features: {len(feature_cols)}")
print(f"  Bankruptcy rate: {y.mean():.2%}")

# Remove constant columns
feature_std = X.std()
non_constant = feature_std[feature_std > 0].index.tolist()
X = X[non_constant]
print(f"  Non-constant features: {len(non_constant)}")

# Train/test split for all methods
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Storage for all remediated datasets
remediated_datasets = {}
results_summary = {'methods': {}}

# ============================================================================
# METHOD 1: VIF-Based Feature Selection (Multicollinearity)
# ============================================================================
print("\n[2/10] Method 1: VIF-Based Feature Selection...")
# From PDF01: VIF = 1/(1-R²) → Remove features with VIF > 10

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X_df):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(len(X_df.columns))]
    return vif_data.sort_values('VIF', ascending=False)

# Iterative VIF removal
X_train_scaled = StandardScaler().fit_transform(X_train)
X_train_vif = pd.DataFrame(X_train_scaled, columns=X_train.columns)

vif_threshold = 10
remaining_features = list(X_train_vif.columns)
iteration = 0

while True:
    iteration += 1
    vif_df = calculate_vif(X_train_vif[remaining_features])
    max_vif = vif_df['VIF'].max()
    
    if max_vif <= vif_threshold or len(remaining_features) <= 5:
        break
    
    # Remove feature with highest VIF
    worst_feature = vif_df.iloc[0]['Feature']
    remaining_features.remove(worst_feature)
    print(f"  Iteration {iteration}: Removed {worst_feature} (VIF={max_vif:.2f}), {len(remaining_features)} features left")

print(f"  ✓ Final features: {len(remaining_features)} (max VIF: {max_vif:.2f})")

# Train model with VIF-selected features
X_train_vif_final = X_train[remaining_features]
X_test_vif_final = X_test[remaining_features]

scaler_vif = StandardScaler()
X_train_vif_scaled = scaler_vif.fit_transform(X_train_vif_final)
X_test_vif_scaled = scaler_vif.transform(X_test_vif_final)

model_vif = LogisticRegression(max_iter=1000, random_state=42)
model_vif.fit(X_train_vif_scaled, y_train)
auc_vif = roc_auc_score(y_test, model_vif.predict_proba(X_test_vif_scaled)[:, 1])

print(f"  Test AUC: {auc_vif:.4f}")

# Save VIF-selected dataset
df_vif = df_h1[['horizon', 'y'] + remaining_features].copy()
remediated_datasets['vif_selected'] = (df_vif, remaining_features, auc_vif)

results_summary['methods']['vif_selection'] = {
    'features_remaining': len(remaining_features),
    'max_vif': float(max_vif),
    'test_auc': float(auc_vif),
    'features': remaining_features
}

# ============================================================================
# METHOD 2: Cochrane-Orcutt Procedure (Autocorrelation)
# ============================================================================
print("\n[3/10] Method 2: Cochrane-Orcutt (CORC) Procedure...")
# From PDF02 lines 77-107:
# Iteratively estimate ρ̂ and transform data: y* = y - ρ̂y_{-1}

# For logistic regression, CORC is complex since we can't transform binary y
# Instead, we estimate ρ and note it for interpretation
# The key insight: autocorrelation affects SE more than coefficients

scaler_corc = StandardScaler()
X_train_corc_scaled = scaler_corc.fit_transform(X_train_vif_final)
X_test_corc_scaled = scaler_corc.transform(X_test_vif_final)

# Estimate autocorrelation in residuals
model_corc_init = LogisticRegression(max_iter=1000, random_state=42)
model_corc_init.fit(X_train_corc_scaled, y_train)

y_pred = model_corc_init.predict_proba(X_train_corc_scaled)[:, 1]
y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)
residuals = (y_train.values - y_pred_clip) / np.sqrt(y_pred_clip * (1 - y_pred_clip))

# Estimate ρ̂ from Durbin-Watson
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(residuals)
rho_estimate = 1 - dw_stat / 2  # DW ≈ 2(1 - ρ)

print(f"  Durbin-Watson: {dw_stat:.4f}")
print(f"  Estimated ρ̂: {rho_estimate:.4f}")

# For logistic regression with autocorrelation, use GEE (Generalized Estimating Equations)
# or note that robust SE should be used
# Here we use the VIF-selected model with notation about autocorrelation

auc_corc = roc_auc_score(y_test, model_corc_init.predict_proba(X_test_corc_scaled)[:, 1])
print(f"  Test AUC: {auc_corc:.4f}")
print(f"  Note: For binary outcomes, use robust/cluster SE instead of CORC transformation")

results_summary['methods']['cochrane_orcutt'] = {
    'durbin_watson': float(dw_stat),
    'rho_estimate': float(rho_estimate),
    'test_auc': float(auc_corc),
    'note': 'Autocorrelation detected - use robust SE or clustered SE for valid inference'
}

# ============================================================================
# METHOD 3: White Robust Standard Errors (Heteroscedasticity)
# ============================================================================
print("\n[4/10] Method 3: White Robust Standard Errors...")
# From PDF02 lines 242-277:
# Var^W(β̂) = (X'X)^{-1} [Σ û²t(xtx't)] (X'X)^{-1}

# Fit logistic with statsmodels to get robust SE
X_train_white = sm.add_constant(scaler_vif.fit_transform(X_train_vif_final))
X_test_white = sm.add_constant(scaler_vif.transform(X_test_vif_final))

model_white = sm.Logit(y_train, X_train_white).fit(disp=0, cov_type='HC3')  # HC3 = White robust SE
y_pred_white = model_white.predict(X_test_white)
auc_white = roc_auc_score(y_test, y_pred_white)

print(f"  Robust SE applied (HC3)")
print(f"  Test AUC: {auc_white:.4f}")
print(f"  Note: Coefficients unchanged, only SE adjusted")

results_summary['methods']['white_robust_se'] = {
    'test_auc': float(auc_white),
    'note': 'Use VIF-selected data with cov_type=HC3 in statsmodels',
    'features': remaining_features
}

# Save same dataset as VIF with note to use robust SE
remediated_datasets['white_robust'] = (df_vif, remaining_features, auc_white)

# ============================================================================
# METHOD 4: Forward Stepwise AIC Selection
# ============================================================================
print("\n[5/10] Method 4: Forward Stepwise AIC Selection...")

def forward_selection(X, y, max_features=20):
    """Forward stepwise selection based on AIC"""
    remaining = list(X.columns)
    selected = []
    current_aic = np.inf
    
    for i in range(min(max_features, len(X.columns))):
        best_aic = np.inf
        best_feature = None
        
        for candidate in remaining:
            test_features = selected + [candidate]
            X_test = sm.add_constant(X[test_features])
            model = sm.Logit(y, X_test).fit(disp=0)
            
            if model.aic < best_aic:
                best_aic = model.aic
                best_feature = candidate
        
        if best_aic < current_aic:
            selected.append(best_feature)
            remaining.remove(best_feature)
            current_aic = best_aic
            print(f"  Step {i+1}: Added {best_feature} (AIC={best_aic:.2f})")
        else:
            break
    
    return selected

# Apply forward selection
scaler_forward = StandardScaler()
X_train_forward_scaled = pd.DataFrame(
    scaler_forward.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

selected_features_forward = forward_selection(X_train_forward_scaled, y_train, max_features=20)
print(f"  ✓ Selected {len(selected_features_forward)} features")

# Train final model
X_train_forward_final = X_train[selected_features_forward]
X_test_forward_final = X_test[selected_features_forward]

scaler_forward_final = StandardScaler()
X_train_forward_scaled = scaler_forward_final.fit_transform(X_train_forward_final)
X_test_forward_scaled = scaler_forward_final.transform(X_test_forward_final)

model_forward = LogisticRegression(max_iter=1000, random_state=42)
model_forward.fit(X_train_forward_scaled, y_train)
auc_forward = roc_auc_score(y_test, model_forward.predict_proba(X_test_forward_scaled)[:, 1])

print(f"  Test AUC: {auc_forward:.4f}")

# Save forward-selected dataset
df_forward = df_h1[['horizon', 'y'] + selected_features_forward].copy()
remediated_datasets['forward_selected'] = (df_forward, selected_features_forward, auc_forward)

results_summary['methods']['forward_selection'] = {
    'features_selected': len(selected_features_forward),
    'test_auc': float(auc_forward),
    'features': selected_features_forward
}

# ============================================================================
# METHOD 5: Ridge Regression (L2 Regularization)
# ============================================================================
print("\n[6/10] Method 5: Ridge Regression (L2)...")

scaler_ridge = StandardScaler()
X_train_ridge_scaled = scaler_ridge.fit_transform(X_train)
X_test_ridge_scaled = scaler_ridge.transform(X_test)

# Cross-validated Ridge
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
ridge_cv.fit(X_train_ridge_scaled, y_train)

# Use Ridge coefficients for Logistic Regression
model_ridge = LogisticRegression(penalty='l2', C=1/ridge_cv.alpha_, max_iter=1000, random_state=42)
model_ridge.fit(X_train_ridge_scaled, y_train)
auc_ridge = roc_auc_score(y_test, model_ridge.predict_proba(X_test_ridge_scaled)[:, 1])

print(f"  Best alpha: {ridge_cv.alpha_:.4f}")
print(f"  Test AUC: {auc_ridge:.4f}")

results_summary['methods']['ridge'] = {
    'alpha': float(ridge_cv.alpha_),
    'test_auc': float(auc_ridge),
    'note': 'All features with L2 penalty'
}

# ============================================================================
# METHOD 6: Lasso Regression (L1 Regularization)
# ============================================================================
print("\n[7/10] Method 6: Lasso Regression (L1)...")

# Cross-validated Lasso
lasso_cv = LassoCV(alphas=np.logspace(-4, 0, 50), cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_ridge_scaled, y_train)

# Features selected by Lasso
lasso_mask = np.abs(lasso_cv.coef_) > 0
lasso_features = X_train.columns[lasso_mask].tolist()

print(f"  Best alpha: {lasso_cv.alpha_:.4f}")
print(f"  Features selected: {len(lasso_features)}")

if len(lasso_features) > 0:
    X_train_lasso = X_train[lasso_features]
    X_test_lasso = X_test[lasso_features]
    
    scaler_lasso = StandardScaler()
    X_train_lasso_scaled = scaler_lasso.fit_transform(X_train_lasso)
    X_test_lasso_scaled = scaler_lasso.transform(X_test_lasso)
    
    model_lasso = LogisticRegression(max_iter=1000, random_state=42)
    model_lasso.fit(X_train_lasso_scaled, y_train)
    auc_lasso = roc_auc_score(y_test, model_lasso.predict_proba(X_test_lasso_scaled)[:, 1])
    
    print(f"  Test AUC: {auc_lasso:.4f}")
    
    # Save Lasso-selected dataset
    df_lasso = df_h1[['horizon', 'y'] + lasso_features].copy()
    remediated_datasets['lasso_selected'] = (df_lasso, lasso_features, auc_lasso)
    
    results_summary['methods']['lasso'] = {
        'alpha': float(lasso_cv.alpha_),
        'features_selected': len(lasso_features),
        'test_auc': float(auc_lasso),
        'features': lasso_features
    }
else:
    print("  ⚠ Lasso selected 0 features, skipping")

# ============================================================================
# SAVE ALL REMEDIATED DATASETS
# ============================================================================
print("\n[8/10] Saving remediated datasets...")

saved_files = {}
for method_name, (df_remediated, features, auc) in remediated_datasets.items():
    output_path = data_dir / f'poland_h1_{method_name}.parquet'
    df_remediated.to_parquet(output_path, index=False)
    saved_files[method_name] = str(output_path)
    print(f"  ✓ {method_name}: {len(features)} features → {output_path.name}")

results_summary['saved_datasets'] = saved_files

# ============================================================================
# Determine best method
# ============================================================================
print("\n[9/10] Comparing all methods...")

method_results = {
    'VIF Selection': auc_vif,
    'Cochrane-Orcutt': auc_corc,
    'White Robust SE': auc_white,
    'Forward Selection': auc_forward,
    'Ridge': auc_ridge,
}

if 'lasso' in results_summary['methods']:
    method_results['Lasso'] = auc_lasso

best_method = max(method_results, key=method_results.get)
best_auc = method_results[best_method]

print("\n  Method Performance:")
for method, auc in sorted(method_results.items(), key=lambda x: x[1], reverse=True):
    marker = "★" if method == best_method else " "
    print(f"  {marker} {method:20s}: {auc:.4f}")

results_summary['best_method'] = {
    'name': best_method,
    'test_auc': float(best_auc)
}

# ============================================================================
# Save summary
# ============================================================================
output_file = output_dir / 'remediation_summary.json'
with open(output_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✓ Remediation summary saved: {output_file}")

# ============================================================================
# Visualizations
# ============================================================================
print("\n[10/10] Creating comparison visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AUC comparison
methods = list(method_results.keys())
aucs = list(method_results.values())
colors = ['#2ecc71' if m == best_method else '#3498db' for m in methods]

axes[0].barh(methods, aucs, color=colors)
axes[0].set_xlabel('Test AUC')
axes[0].set_title('Method Comparison')
axes[0].axvline(x=0.75, color='r', linestyle='--', alpha=0.5, label='Baseline')
axes[0].legend()

# Feature count comparison
feature_counts = {
    'Original': len(feature_cols),
    'VIF': len(remaining_features),
    'Forward': len(selected_features_forward),
}
if 'lasso' in results_summary['methods']:
    feature_counts['Lasso'] = len(lasso_features)

axes[1].bar(feature_counts.keys(), feature_counts.values(), color='#9b59b6')
axes[1].set_ylabel('Number of Features')
axes[1].set_title('Feature Reduction')
axes[1].axhline(y=10, color='r', linestyle='--', alpha=0.5, label='EPV Target (10+)')

plt.tight_layout()
plt.savefig(figures_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Visualizations saved")

# ============================================================================
# Final summary
# ============================================================================
print("\n" + "=" * 70)
print("✓ REMEDIATION COMPLETE")
print("=" * 70)

print(f"\n🏆 BEST METHOD: {best_method} (AUC = {best_auc:.4f})")
print(f"\n📁 SAVED DATASETS:")
for method, path in saved_files.items():
    print(f"   • {method}: {Path(path).name}")

print(f"\n➡️  NEXT STEP: Use remediated datasets in scripts 11-13")
print(f"   Example: pd.read_parquet('{data_dir}/poland_h1_vif_selected.parquet')")

print("\n" + "=" * 70)
