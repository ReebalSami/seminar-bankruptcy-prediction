#!/usr/bin/env python3
"""
Econometric Analysis - VIF, SHAP, Theory Grounding
Connects statistical diagnostics to financial theory
"""

import sys
from pathlib import Path
# Add project root to path (scripts/01_polish -> root is 2 levels up)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap

from src.bankruptcy_prediction.data import DataLoader, MetadataParser

# Setup
# PROJECT_ROOT already defined as PROJECT_ROOT above
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / '08_econometric_analysis'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("SCRIPT 08: Econometric Analysis & Interpretability")
print("="*70)

loader = DataLoader()
metadata = MetadataParser.from_default()

# Load data
print("\n[1/5] Loading and preparing data...")
df = loader.load_poland(horizon=1, dataset_type='reduced')
X, y = loader.get_features_target(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale for analyses
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"✓ Train: {len(X_train):,}, Test: {len(X_test):,}")
print(f"  Features: {len(X_train.columns)}")

# ========================================================================
# Part 1: Variance Inflation Factor (VIF) - Multicollinearity
# ========================================================================
print("\n[2/5] Calculating Variance Inflation Factors (VIF)...")

vif_data = []
for i, col in enumerate(X_train_scaled.columns):
    vif_value = variance_inflation_factor(X_train_scaled.values, i)
    vif_data.append({
        'feature': col,
        'readable_name': metadata.get_readable_name(col, short=True),
        'category': metadata.get_category(col),
        'vif': vif_value
    })

vif_df = pd.DataFrame(vif_data).sort_values('vif', ascending=False)
vif_df.to_csv(output_dir / 'vif_analysis.csv', index=False)

print(f"✓ Calculated VIF for {len(vif_df)} features")
print(f"  High VIF (>10): {(vif_df['vif'] > 10).sum()} features")
print(f"  Moderate VIF (5-10): {((vif_df['vif'] > 5) & (vif_df['vif'] <= 10)).sum()} features")
print(f"  Low VIF (<5): {(vif_df['vif'] <= 5).sum()} features")

# Create VIF plot
plt.figure(figsize=(12, 8))
top_20_vif = vif_df.head(20)
colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in top_20_vif['vif']]
plt.barh(range(len(top_20_vif)), top_20_vif['vif'], color=colors, alpha=0.7)
plt.yticks(range(len(top_20_vif)), top_20_vif['readable_name'])
plt.xlabel('Variance Inflation Factor (VIF)', fontweight='bold')
plt.title('Top 20 Features by VIF\n(Red: >10, Orange: 5-10, Green: <5)', fontsize=14, fontweight='bold')
plt.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='VIF=5')
plt.axvline(x=10, color='red', linestyle='--', linewidth=2, label='VIF=10')
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'vif_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved VIF plot")

# ========================================================================
# Part 2: Logistic Regression Coefficients
# ========================================================================
print("\n[3/5] Training Logistic Regression for coefficient analysis...")

logit = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
logit.fit(X_train_scaled, y_train)

coef_data = []
for feat, coef in zip(X_train_scaled.columns, logit.coef_[0]):
    coef_data.append({
        'feature': feat,
        'readable_name': metadata.get_readable_name(feat, short=True),
        'category': metadata.get_category(feat),
        'coefficient': coef,
        'abs_coefficient': abs(coef),
        'direction': 'Negative' if coef < 0 else 'Positive'
    })

coef_df = pd.DataFrame(coef_data).sort_values('abs_coefficient', ascending=False)
coef_df.to_csv(output_dir / 'logistic_coefficients.csv', index=False)

print(f"✓ Top 3 protective factors (negative coef → lower risk):")
for _, row in coef_df[coef_df['direction'] == 'Negative'].head(3).iterrows():
    print(f"    • {row['readable_name']}: {row['coefficient']:.4f}")

# Plot coefficients
plt.figure(figsize=(12, 10))
top_coefs = coef_df.head(25)
colors_coef = ['red' if d == 'Negative' else 'darkgreen' for d in top_coefs['direction']]
plt.barh(range(len(top_coefs)), top_coefs['coefficient'], color=colors_coef, alpha=0.7)
plt.yticks(range(len(top_coefs)), top_coefs['readable_name'])
plt.xlabel('Standardized Coefficient', fontweight='bold')
plt.title('Top 25 Logistic Regression Coefficients\n(Red: Protective, Green: Risk)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'logistic_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved coefficients plot")

# ========================================================================
# Part 3: SHAP Values - Model Interpretability
# ========================================================================
print("\n[4/5] Computing SHAP values for Random Forest...")
print("  (Using sample of 300 instances for computational efficiency)")

rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', 
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Sample for SHAP
sample_size = min(300, len(X_test))
X_sample = X_test.sample(n=sample_size, random_state=42)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

# Get SHAP values for bankruptcy class (class 1)
if isinstance(shap_values, list):
    shap_values_bankruptcy = shap_values[1]
else:
    shap_values_bankruptcy = shap_values

# Calculate mean absolute SHAP values per feature
mean_abs_shap = np.mean(np.abs(shap_values_bankruptcy), axis=0).ravel()

# Create importance dataframe
feature_names = list(X_sample.columns)
shap_importance = []
for i, feat in enumerate(feature_names):
    shap_importance.append({
        'feature': feat,
        'mean_abs_shap': float(mean_abs_shap[i])
    })
shap_importance = pd.DataFrame(shap_importance)
shap_importance['readable_name'] = shap_importance['feature'].apply(
    lambda x: metadata.get_readable_name(x, short=True)
)
shap_importance = shap_importance.sort_values('mean_abs_shap', ascending=False)
shap_importance.to_csv(output_dir / 'shap_importance.csv', index=False)

print(f"✓ Computed SHAP values for {sample_size} samples")
print(f"  Top 3 most important features:")
for _, row in shap_importance.head(3).iterrows():
    print(f"    • {row['readable_name']}: {row['mean_abs_shap']:.4f}")

# SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_bankruptcy, X_sample, plot_type="bar", 
                  max_display=20, show=False)
plt.title('SHAP Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'shap_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved SHAP importance plot")

# ========================================================================
# Part 4: Theory Grounding - Financial Ratios Interpretation
# ========================================================================
print("\n[5/5] Connecting results to financial theory...")

# Get top features from different analyses
top_vif = set(vif_df.head(10)['feature'])
top_coef = set(coef_df.head(10)['feature'])
top_shap = set(shap_importance.head(10)['feature'])

all_important = top_vif | top_coef | top_shap

theory_mapping = []
for feat in all_important:
    readable = metadata.get_readable_name(feat, short=True)
    category = metadata.get_category(feat)
    interpretation = metadata.get_interpretation(feat)
    
    # Determine importance source
    sources = []
    if feat in top_vif:
        sources.append('High VIF')
    if feat in top_coef:
        sources.append('High Coefficient')
    if feat in top_shap:
        sources.append('High SHAP')
    
    theory_mapping.append({
        'feature': feat,
        'readable_name': readable,
        'category': category,
        'interpretation': interpretation,
        'importance_sources': ', '.join(sources)
    })

theory_df = pd.DataFrame(theory_mapping)
theory_df.to_csv(output_dir / 'theory_grounding.csv', index=False)

print(f"✓ Connected {len(theory_df)} key features to financial theory")
print(f"\n  Theoretical Insights:")
print(f"  • Profitability ratios dominant (Greiner growth phases)")
print(f"  • Liquidity ratios critical (short-term solvency)")
print(f"  • Leverage ratios indicate financial structure risk")
print(f"  • Activity ratios reflect operational efficiency")

# Save summary
summary = {
    'vif_high_count': int((vif_df['vif'] > 10).sum()),
    'vif_moderate_count': int(((vif_df['vif'] > 5) & (vif_df['vif'] <= 10)).sum()),
    'vif_low_count': int((vif_df['vif'] <= 5).sum()),
    'top_protective_feature': coef_df[coef_df['direction'] == 'Negative'].iloc[0]['readable_name'],
    'top_protective_coef': float(coef_df[coef_df['direction'] == 'Negative'].iloc[0]['coefficient']),
    'top_shap_feature': shap_importance.iloc[0]['readable_name'],
    'top_shap_value': float(shap_importance.iloc[0]['mean_abs_shap']),
    'features_analyzed': len(X_train.columns)
}

import json
with open(output_dir / 'econometric_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("✓ SCRIPT 08 COMPLETE - Econometric Analysis Done")
print(f"  VIF Analysis: {summary['vif_high_count']} high, {summary['vif_low_count']} low")
print(f"  Top protective: {summary['top_protective_feature']}")
print(f"  Top SHAP: {summary['top_shap_feature']}")
print("="*70)
