#!/usr/bin/env python3
"""
Taiwan Dataset - Baseline Models
Logistic Regression, Random Forest, CatBoost (95 features, severe imbalance)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                            brier_score_loss, roc_curve)

# Setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
data_dir = PROJECT_ROOT / 'data' / 'processed' / 'taiwan'
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / 'taiwan'
figures_dir = output_dir / 'figures'

print("="*70)
print("TAIWAN DATASET - Baseline Models")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_parquet(data_dir / 'taiwan_clean.parquet')

import json
with open(data_dir / 'taiwan_features_metadata.json', 'r') as f:
    feature_mapping = json.load(f)

feature_cols = [col for col in df.columns if col.startswith('F')]
X = df[feature_cols].values
y = df['bankrupt'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Loaded {len(df):,} samples")
print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
print(f"  Features: {len(feature_cols)}")
print(f"  Bankruptcy rate: {y.mean()*100:.2f}%")

# Helper function
def evaluate_model(y_true, y_pred_proba, model_name):
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    idx_1pct = np.argmin(np.abs(fpr - 0.01))
    recall_1pct = tpr[idx_1pct]
    
    idx_5pct = np.argmin(np.abs(fpr - 0.05))
    recall_5pct = tpr[idx_5pct]
    
    return {
        'model_name': model_name,
        'dataset': 'Taiwan',
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'recall_1pct_fpr': recall_1pct,
        'recall_5pct_fpr': recall_5pct
    }

results_list = []

# ========================================================================
# Model 1: Logistic Regression (L2 regularized for high dimensions)
# ========================================================================
print("\n[2/5] Training Logistic Regression...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logit = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000,
                           penalty='l2', random_state=42)
logit.fit(X_train_scaled, y_train)

y_pred_logit = logit.predict_proba(X_test_scaled)[:, 1]
results_logit = evaluate_model(y_test, y_pred_logit, 'Logistic Regression')
results_list.append(results_logit)

print(f"✓ Logistic Regression (L2 regularized)")
print(f"  ROC-AUC: {results_logit['roc_auc']:.4f}")
print(f"  PR-AUC: {results_logit['pr_auc']:.4f}")

# ========================================================================
# Model 2: Random Forest
# ========================================================================
print("\n[3/5] Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',  # Limit features per split for high dimensions
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict_proba(X_test)[:, 1]
results_rf = evaluate_model(y_test, y_pred_rf, 'Random Forest')
results_list.append(results_rf)

print(f"✓ Random Forest")
print(f"  ROC-AUC: {results_rf['roc_auc']:.4f}")
print(f"  PR-AUC: {results_rf['pr_auc']:.4f}")

# ========================================================================
# Model 3: CatBoost
# ========================================================================
print("\n[4/5] Training CatBoost...")

try:
    from catboost import CatBoostClassifier
    
    cat = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        auto_class_weights='Balanced',
        random_state=42,
        verbose=False
    )
    cat.fit(X_train, y_train)
    
    y_pred_cat = cat.predict_proba(X_test)[:, 1]
    results_cat = evaluate_model(y_test, y_pred_cat, 'CatBoost')
    results_list.append(results_cat)
    
    print(f"✓ CatBoost")
    print(f"  ROC-AUC: {results_cat['roc_auc']:.4f}")
    print(f"  PR-AUC: {results_cat['pr_auc']:.4f}")
except Exception as e:
    print(f"⚠ CatBoost failed: {str(e)[:100]}")

# ========================================================================
# Save Results & Visualizations
# ========================================================================
print("\n[5/5] Saving results...")

results_df = pd.DataFrame(results_list)
results_df.to_csv(output_dir / 'baseline_results.csv', index=False)

summary = {
    'models_trained': len(results_list),
    'best_model': results_df.loc[results_df['roc_auc'].idxmax(), 'model_name'],
    'best_roc_auc': float(results_df['roc_auc'].max()),
    'best_pr_auc': float(results_df['pr_auc'].max()),
    'samples_train': int(len(X_train)),
    'samples_test': int(len(X_test)),
    'features': len(feature_cols)
}

with open(output_dir / 'baseline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# ROC Curves
plt.figure(figsize=(10, 8))

for model_name in results_df['model_name'].unique():
    if model_name == 'Logistic Regression':
        y_pred = y_pred_logit
    elif model_name == 'Random Forest':
        y_pred = y_pred_rf
    elif model_name == 'CatBoost':
        y_pred = y_pred_cat
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = results_df[results_df['model_name'] == model_name]['roc_auc'].values[0]
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('ROC Curves - Taiwan Dataset', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# Model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(range(len(results_df)), results_df['roc_auc'],
           color=['steelblue', 'darkgreen', 'crimson'][:len(results_df)], alpha=0.7)
axes[0].set_xticks(range(len(results_df)))
axes[0].set_xticklabels(results_df['model_name'], rotation=15, ha='right')
axes[0].set_ylabel('ROC-AUC', fontweight='bold')
axes[0].set_title('Model Performance', fontsize=12, fontweight='bold')
axes[0].set_ylim([0.5, 1.0])
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(results_df['roc_auc']):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

axes[1].bar(range(len(results_df)), results_df['pr_auc'],
           color=['steelblue', 'darkgreen', 'crimson'][:len(results_df)], alpha=0.7)
axes[1].set_xticks(range(len(results_df)))
axes[1].set_xticklabels(results_df['model_name'], rotation=15, ha='right')
axes[1].set_ylabel('PR-AUC', fontweight='bold')
axes[1].set_title('Precision-Recall AUC', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(results_df['pr_auc']):
    axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ Saved visualizations")

print("\n" + "="*70)
print("✓ TAIWAN DATASET BASELINE MODELS COMPLETE")
print(f"  Best model: {summary['best_model']}")
print(f"  Best ROC-AUC: {summary['best_roc_auc']:.4f}")
print(f"  Best PR-AUC: {summary['best_pr_auc']:.4f}")
print("="*70)
