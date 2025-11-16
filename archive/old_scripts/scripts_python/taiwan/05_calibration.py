#!/usr/bin/env python3
"""Taiwan Dataset - Model Calibration"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from scripts.config import RANDOM_STATE
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
data_dir = project_root / 'data' / 'processed' / 'taiwan'
output_dir = project_root / 'results' / 'script_outputs' / 'taiwan' / '05_calibration'
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("TAIWAN - Model Calibration")
print("="*80)

df = pd.read_parquet(data_dir / 'taiwan_clean.parquet')
feature_cols = [col for col in df.columns if col.startswith('F')]
X = df[feature_cols].values
y = df['bankrupt'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
brier_lr = brier_score_loss(y_test, y_pred_lr)

lr_cal = CalibratedClassifierCV(lr, method='sigmoid', cv=3)
lr_cal.fit(X_train_scaled, y_train)
y_pred_lr_cal = lr_cal.predict_proba(X_test_scaled)[:, 1]
brier_lr_cal = brier_score_loss(y_test, y_pred_lr_cal)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict_proba(X_test_scaled)[:, 1]
brier_rf = brier_score_loss(y_test, y_pred_rf)

rf_cal = CalibratedClassifierCV(rf, method='isotonic', cv=3)
rf_cal.fit(X_train_scaled, y_train)
y_pred_rf_cal = rf_cal.predict_proba(X_test_scaled)[:, 1]
brier_rf_cal = brier_score_loss(y_test, y_pred_rf_cal)

improvement_lr = (brier_lr - brier_lr_cal) / brier_lr * 100
improvement_rf = (brier_rf - brier_rf_cal) / brier_rf * 100

print(f"LR:  {brier_lr:.4f} → {brier_lr_cal:.4f} ({improvement_lr:+.1f}%)")
print(f"RF:  {brier_rf:.4f} → {brier_rf_cal:.4f} ({improvement_rf:+.1f}%)")

import json
with open(output_dir / 'calibration_results.json', 'w') as f:
    json.dump({
        'logistic': {'before': float(brier_lr), 'after': float(brier_lr_cal), 'improvement_pct': float(improvement_lr)},
        'random_forest': {'before': float(brier_rf), 'after': float(brier_rf_cal), 'improvement_pct': float(improvement_rf)}
    }, f, indent=2)

print(f"\n✓ TAIWAN CALIBRATION COMPLETE")
print("="*80)
