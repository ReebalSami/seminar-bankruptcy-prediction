#!/usr/bin/env python3
"""
American Dataset - Model Calibration

Created: November 12, 2025 (Phase 2 - Equal Treatment)
Purpose: Calibration analysis for American dataset (mirrors Polish script 06)
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
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from scripts.config import RANDOM_STATE

print("="*80)
print("AMERICAN DATASET - Model Calibration")
print("="*80)
print("\n📌 Phase 2: Equal Treatment - Calibration analysis for American dataset\n")

# Setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
data_dir = PROJECT_ROOT / 'data' / 'processed' / 'american'
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / 'american' / '05_calibration'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# Load data
print("[1/4] Loading data...")
df = pd.read_parquet(data_dir / 'american_modeling.parquet')

feature_cols = [col for col in df.columns if col.startswith('X')]
X = df[feature_cols].values
y = df['bankrupt'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Train: {len(X_train):,}, Test: {len(X_test):,}")

# Train models
print("\n[2/4] Training models...")

# Logistic Regression
lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
brier_lr = brier_score_loss(y_test, y_pred_lr)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=50,
                            class_weight='balanced', random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict_proba(X_test_scaled)[:, 1]
brier_rf = brier_score_loss(y_test, y_pred_rf)

print(f"✓ Logistic Regression - Brier: {brier_lr:.4f}")
print(f"✓ Random Forest - Brier: {brier_rf:.4f}")

# Apply calibration
print("\n[3/4] Applying calibration...")

lr_calibrated = CalibratedClassifierCV(lr, method='sigmoid', cv=5)
lr_calibrated.fit(X_train_scaled, y_train)
y_pred_lr_cal = lr_calibrated.predict_proba(X_test_scaled)[:, 1]
brier_lr_cal = brier_score_loss(y_test, y_pred_lr_cal)

rf_calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=5)
rf_calibrated.fit(X_train_scaled, y_train)
y_pred_rf_cal = rf_calibrated.predict_proba(X_test_scaled)[:, 1]
brier_rf_cal = brier_score_loss(y_test, y_pred_rf_cal)

print(f"✓ LR Calibrated - Brier: {brier_lr_cal:.4f} (Δ={brier_lr_cal-brier_lr:+.4f})")
print(f"✓ RF Calibrated - Brier: {brier_rf_cal:.4f} (Δ={brier_rf_cal-brier_rf:+.4f})")

# Create calibration plots
print("\n[4/4] Creating calibration plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Logistic Regression
prob_true_lr, prob_pred_lr = calibration_curve(y_test, y_pred_lr, n_bins=10)
prob_true_lr_cal, prob_pred_lr_cal = calibration_curve(y_test, y_pred_lr_cal, n_bins=10)

axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
axes[0].plot(prob_pred_lr, prob_true_lr, 'o-', label=f'Uncalibrated (Brier={brier_lr:.3f})')
axes[0].plot(prob_pred_lr_cal, prob_true_lr_cal, 's-', label=f'Calibrated (Brier={brier_lr_cal:.3f})')
axes[0].set_xlabel('Predicted Probability')
axes[0].set_ylabel('True Probability')
axes[0].set_title('Logistic Regression Calibration', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Random Forest
prob_true_rf, prob_pred_rf = calibration_curve(y_test, y_pred_rf, n_bins=10)
prob_true_rf_cal, prob_pred_rf_cal = calibration_curve(y_test, y_pred_rf_cal, n_bins=10)

axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
axes[1].plot(prob_pred_rf, prob_true_rf, 'o-', label=f'Uncalibrated (Brier={brier_rf:.3f})')
axes[1].plot(prob_pred_rf_cal, prob_true_rf_cal, 's-', label=f'Calibrated (Brier={brier_rf_cal:.3f})')
axes[1].set_xlabel('Predicted Probability')
axes[1].set_ylabel('True Probability')
axes[1].set_title('Random Forest Calibration', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'calibration_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved calibration curves to: {figures_dir}/calibration_curves.png")

# Save results
results = {
    'logistic_regression': {
        'brier_uncalibrated': float(brier_lr),
        'brier_calibrated': float(brier_lr_cal),
        'improvement_pct': float((brier_lr - brier_lr_cal) / brier_lr * 100)
    },
    'random_forest': {
        'brier_uncalibrated': float(brier_rf),
        'brier_calibrated': float(brier_rf_cal),
        'improvement_pct': float((brier_rf - brier_rf_cal) / brier_rf * 100)
    }
}

import json
with open(output_dir / 'calibration_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Saved results to: {output_dir}/calibration_results.json")

print("\n" + "="*80)
print("✓ AMERICAN MODEL CALIBRATION COMPLETE")
print("="*80)
print(f"\n  LR:  {brier_lr:.4f} → {brier_lr_cal:.4f} ({results['logistic_regression']['improvement_pct']:+.1f}%)")
print(f"  RF:  {brier_rf:.4f} → {brier_rf_cal:.4f} ({results['random_forest']['improvement_pct']:+.1f}%)")
print("="*80)
