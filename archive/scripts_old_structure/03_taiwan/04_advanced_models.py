#!/usr/bin/env python3
"""Taiwan Dataset - Advanced Models (XGBoost, LightGBM, CatBoost)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scripts.config import RANDOM_STATE, XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
data_dir = PROJECT_ROOT / 'data' / 'processed' / 'taiwan'
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / 'taiwan' / '04_advanced'
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("TAIWAN - Advanced Models")
print("="*80)

df = pd.read_parquet(data_dir / 'taiwan_clean.parquet')
feature_cols = [col for col in df.columns if col.startswith('F')]
X = df[feature_cols].values
y = df['bankrupt'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

results = []

# XGBoost
try:
    import xgboost as xgb
    xgb_params = XGBOOST_PARAMS.copy()
    xgb_params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_scaled, y_train, verbose=False)
    y_pred = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred)
    results.append({'model': 'XGBoost', 'auc': auc, 'pr_auc': pr_auc})
    print(f"✓ XGBoost: AUC={auc:.4f}, PR-AUC={pr_auc:.4f}")
except Exception as e:
    print(f"⚠ XGBoost error: {str(e)[:50]}")

# LightGBM
try:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred)
    results.append({'model': 'LightGBM', 'auc': auc, 'pr_auc': pr_auc})
    print(f"✓ LightGBM: AUC={auc:.4f}, PR-AUC={pr_auc:.4f}")
except Exception as e:
    print(f"⚠ LightGBM error: {str(e)[:50]}")

# CatBoost
try:
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred)
    results.append({'model': 'CatBoost', 'auc': auc, 'pr_auc': pr_auc})
    print(f"✓ CatBoost: AUC={auc:.4f}, PR-AUC={pr_auc:.4f}")
except Exception as e:
    print(f"⚠ CatBoost error: {str(e)[:50]}")

import json
with open(output_dir / 'advanced_models_results.json', 'w') as f:
    json.dump({'results': results}, f, indent=2)

print(f"\n✓ TAIWAN ADVANCED MODELS COMPLETE - {len(results)} models")
print("="*80)
