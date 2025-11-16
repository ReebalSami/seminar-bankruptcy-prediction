#!/usr/bin/env python3
"""Baseline Models - Logistic, RF, GLM"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

from src.bankruptcy_prediction.data import DataLoader
from src.bankruptcy_prediction.evaluation import ResultsCollector

# Setup
project_root = Path(__file__).parent.parent
output_dir = project_root / 'results' / 'script_outputs' / '04_baseline_models'
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("SCRIPT 04: Baseline Models")
print("="*70)

def evaluate_model(y_true, y_pred_proba, model_name='Model'):
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    idx_1pct = np.where(fpr <= 0.01)[0]
    recall_1pct = tpr[idx_1pct[-1]] if len(idx_1pct) > 0 else 0.0
    idx_5pct = np.where(fpr <= 0.05)[0]
    recall_5pct = tpr[idx_5pct[-1]] if len(idx_5pct) > 0 else 0.0
    
    return {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'recall_1pct_fpr': recall_1pct,
        'recall_5pct_fpr': recall_5pct,
        'horizon': 1
    }

# Load data
print("\n[1/4] Loading data...")
loader = DataLoader()
df_full = loader.load_poland(horizon=1, dataset_type='full')
df_reduced = loader.load_poland(horizon=1, dataset_type='reduced')

X_full, y = loader.get_features_target(df_full)
X_reduced, _ = loader.get_features_target(df_reduced)

X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42, stratify=y)
X_train_reduced, X_test_reduced, _, _ = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_reduced_scaled = pd.DataFrame(scaler.fit_transform(X_train_reduced), columns=X_train_reduced.columns, index=X_train_reduced.index)
X_test_reduced_scaled = pd.DataFrame(scaler.transform(X_test_reduced), columns=X_test_reduced.columns, index=X_test_reduced.index)

print(f"✓ Train: {len(y_train):,}, Test: {len(y_test):,}")

# Train Logistic Regression
print("\n[2/4] Training Logistic Regression...")
logit = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
logit.fit(X_train_reduced_scaled, y_train)
y_pred_logit = logit.predict_proba(X_test_reduced_scaled)[:, 1]
results_logit = evaluate_model(y_test, y_pred_logit, 'Logistic Regression')
print(f"✓ Logit ROC-AUC: {results_logit['roc_auc']:.4f}")

# Train Random Forest
print("\n[3/4] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_full, y_train)
y_pred_rf = rf.predict_proba(X_test_full)[:, 1]
results_rf = evaluate_model(y_test, y_pred_rf, 'Random Forest')
print(f"✓ RF ROC-AUC: {results_rf['roc_auc']:.4f}")

# Save results
print("\n[4/4] Saving results...")
results_collector = ResultsCollector()
results_collector.add(results_logit)
results_collector.add(results_rf)
results_collector.save()

comparison = pd.DataFrame([results_logit, results_rf])
comparison.to_csv(output_dir / 'baseline_results.csv', index=False)

print("\n" + "="*70)
print("✓ SCRIPT 04 COMPLETE")
print(f"  Logistic: {results_logit['roc_auc']:.4f}")
print(f"  RF:       {results_rf['roc_auc']:.4f}")
print("="*70)
