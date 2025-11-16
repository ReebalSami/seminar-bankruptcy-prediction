#!/usr/bin/env python3
"""
American Dataset - Baseline Models
Logistic Regression, Random Forest, CatBoost
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                            brier_score_loss, roc_curve, precision_recall_curve)

# Setup
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / 'data' / 'processed' / 'american'
output_dir = project_root / 'results' / 'script_outputs' / 'american'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("AMERICAN DATASET - Baseline Models")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_parquet(data_dir / 'american_modeling.parquet')

import json
with open(data_dir / 'american_features_metadata.json', 'r') as f:
    metadata = json.load(f)

feature_cols = [col for col in df.columns if col.startswith('X')]
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
    """Evaluate model and return metrics"""
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # Recall at specific FPR thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Find threshold for 1% FPR
    idx_1pct = np.argmin(np.abs(fpr - 0.01))
    recall_1pct = tpr[idx_1pct]
    
    # Find threshold for 5% FPR
    idx_5pct = np.argmin(np.abs(fpr - 0.05))
    recall_5pct = tpr[idx_5pct]
    
    return {
        'model_name': model_name,
        'dataset': 'American',
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'recall_1pct_fpr': recall_1pct,
        'recall_5pct_fpr': recall_5pct
    }

results_list = []

# ========================================================================
# Model 1: Logistic Regression
# ========================================================================
print("\n[2/5] Training Logistic Regression...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logit = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
logit.fit(X_train_scaled, y_train)

y_pred_logit = logit.predict_proba(X_test_scaled)[:, 1]
results_logit = evaluate_model(y_test, y_pred_logit, 'Logistic Regression')
results_list.append(results_logit)

print(f"✓ Logistic Regression")
print(f"  ROC-AUC: {results_logit['roc_auc']:.4f}")
print(f"  PR-AUC: {results_logit['pr_auc']:.4f}")
print(f"  Recall@1%FPR: {results_logit['recall_1pct_fpr']*100:.1f}%")

# ========================================================================
# Model 2: Random Forest
# ========================================================================
print("\n[3/5] Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
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
print(f"  Recall@1%FPR: {results_rf['recall_1pct_fpr']*100:.1f}%")

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
    print(f"  Recall@1%FPR: {results_cat['recall_1pct_fpr']*100:.1f}%")
except Exception as e:
    print(f"⚠ CatBoost failed: {str(e)[:100]}")

# ========================================================================
# Save Results & Visualizations
# ========================================================================
print("\n[5/5] Saving results and visualizations...")

# Save results
results_df = pd.DataFrame(results_list)
results_df.to_csv(output_dir / 'baseline_results.csv', index=False)
print(f"✓ Saved results for {len(results_list)} models")

# Save summary
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

# Visualization 1: ROC Curves
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
plt.title('ROC Curves - American Dataset', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved ROC curves")

# Visualization 2: Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC-AUC comparison
axes[0].bar(range(len(results_df)), results_df['roc_auc'], 
           color=['steelblue', 'darkgreen', 'crimson'][:len(results_df)], alpha=0.7)
axes[0].set_xticks(range(len(results_df)))
axes[0].set_xticklabels(results_df['model_name'], rotation=15, ha='right')
axes[0].set_ylabel('ROC-AUC', fontweight='bold')
axes[0].set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
axes[0].set_ylim([0.5, 1.0])
axes[0].grid(axis='y', alpha=0.3)

# Add values on bars
for i, v in enumerate(results_df['roc_auc']):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

# Recall@1%FPR comparison
axes[1].bar(range(len(results_df)), results_df['recall_1pct_fpr'] * 100,
           color=['steelblue', 'darkgreen', 'crimson'][:len(results_df)], alpha=0.7)
axes[1].set_xticks(range(len(results_df)))
axes[1].set_xticklabels(results_df['model_name'], rotation=15, ha='right')
axes[1].set_ylabel('Recall @ 1% FPR (%)', fontweight='bold')
axes[1].set_title('Operational Performance', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Add values on bars
for i, v in enumerate(results_df['recall_1pct_fpr'] * 100):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved model comparison")

# Feature importance (Random Forest)
feature_names_list = [metadata['features'][col]['name'] for col in feature_cols]
importance_df = pd.DataFrame({
    'feature': feature_names_list,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

plt.figure(figsize=(10, 8))
top_15 = importance_df.head(15)
plt.barh(range(len(top_15)), top_15['importance'], color='forestgreen', alpha=0.7)
plt.yticks(range(len(top_15)), top_15['feature'])
plt.xlabel('Importance', fontweight='bold')
plt.title('Random Forest Feature Importance (Top 15)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved feature importance")

print("\n" + "="*70)
print("✓ AMERICAN DATASET BASELINE MODELS COMPLETE")
print(f"  Best model: {summary['best_model']}")
print(f"  Best ROC-AUC: {summary['best_roc_auc']:.4f}")
print(f"  Models trained: {summary['models_trained']}")
print("="*70)
