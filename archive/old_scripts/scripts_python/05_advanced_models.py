#!/usr/bin/env python3
"""Advanced Models - XGBoost, LightGBM"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

from src.bankruptcy_prediction.data import DataLoader
from src.bankruptcy_prediction.evaluation import ResultsCollector

# Setup
project_root = Path(__file__).parent.parent
output_dir = project_root / 'results' / 'script_outputs' / '05_advanced_models'
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("SCRIPT 05: Advanced Models")
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
print("\n[1/5] Loading data...")
loader = DataLoader()
df = loader.load_poland(horizon=1, dataset_type='full')
X, y = loader.get_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✓ Train: {len(y_train):,}, Test: {len(y_test):,}")

results_list = []

# Train XGBoost
print("\n[2/5] Training XGBoost...")
try:
    import xgboost as xgb
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
    results_xgb = evaluate_model(y_test, y_pred_xgb, 'XGBoost')
    results_list.append(results_xgb)
    print(f"✓ XGBoost ROC-AUC: {results_xgb['roc_auc']:.4f}")
except (ImportError, Exception) as e:
    print(f"⚠ XGBoost not available: {str(e)[:100]}")
    print("  Mac users: Run 'brew install libomp' and reinstall")

# Train LightGBM
print("\n[3/5] Training LightGBM...")
try:
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict_proba(X_test)[:, 1]
    results_lgb = evaluate_model(y_test, y_pred_lgb, 'LightGBM')
    results_list.append(results_lgb)
    print(f"✓ LightGBM ROC-AUC: {results_lgb['roc_auc']:.4f}")
except (ImportError, Exception) as e:
    print(f"⚠ LightGBM not available: {str(e)[:100]}")

# Train CatBoost
print("\n[4/5] Training CatBoost...")
try:
    from catboost import CatBoostClassifier
    cat_model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        auto_class_weights='Balanced',
        random_state=42, verbose=False
    )
    cat_model.fit(X_train, y_train)
    y_pred_cat = cat_model.predict_proba(X_test)[:, 1]
    results_cat = evaluate_model(y_test, y_pred_cat, 'CatBoost')
    results_list.append(results_cat)
    print(f"✓ CatBoost ROC-AUC: {results_cat['roc_auc']:.4f}")
except (ImportError, Exception) as e:
    print(f"⚠ CatBoost not available: {str(e)[:100]}")

# Save results
if results_list:
    print("\n[5/5] Saving results...")
    results_collector = ResultsCollector()
    for result in results_list:
        results_collector.add(result)
    results_collector.save()
    
    pd.DataFrame(results_list).to_csv(output_dir / 'advanced_results.csv', index=False)
    print(f"✓ Saved {len(results_list)} model results")
    
    print("\n" + "="*70)
    print("✓ SCRIPT 05 COMPLETE - Advanced Models Trained")
    for r in results_list:
        print(f"  {r['model_name']:15s}: {r['roc_auc']:.4f}")
    print("="*70)
else:
    print("\n" + "="*70)
    print("⚠ WARNING: No models trained (dependency issues)")
    print("  To fix XGBoost on Mac: brew install libomp")
    print("="*70)
