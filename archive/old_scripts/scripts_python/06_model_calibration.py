#!/usr/bin/env python3
"""Model Calibration Analysis"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_curve

from src.bankruptcy_prediction.data import DataLoader

# Setup
project_root = Path(__file__).parent.parent
output_dir = project_root / 'results' / 'script_outputs' / '06_model_calibration'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("SCRIPT 06: Model Calibration")
print("="*70)

loader = DataLoader()

print("\n[1/4] Loading and preparing data...")
df_full = loader.load_poland(horizon=1, dataset_type='full')
df_reduced = loader.load_poland(horizon=1, dataset_type='reduced')

X_full, y = loader.get_features_target(df_full)
X_reduced, _ = loader.get_features_target(df_reduced)

X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42, stratify=y)
X_train_reduced, X_test_reduced, _, _ = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_reduced_scaled = scaler.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler.transform(X_test_reduced)

print(f"✓ Data prepared")

print("\n[2/4] Training models...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_full, y_train)
y_pred_rf = rf.predict_proba(X_test_full)[:, 1]
brier_rf = brier_score_loss(y_test, y_pred_rf)

logit = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
logit.fit(X_train_reduced_scaled, y_train)
y_pred_logit = logit.predict_proba(X_test_reduced_scaled)[:, 1]
brier_logit = brier_score_loss(y_test, y_pred_logit)

print(f"✓ RF Brier: {brier_rf:.4f}, Logit Brier: {brier_logit:.4f}")

print("\n[3/4] Applying calibration...")
rf_cal = CalibratedClassifierCV(rf, method='isotonic', cv='prefit')
rf_cal.fit(X_train_full, y_train)
y_pred_rf_cal = rf_cal.predict_proba(X_test_full)[:, 1]
brier_rf_cal = brier_score_loss(y_test, y_pred_rf_cal)

logit_cal = CalibratedClassifierCV(logit, method='isotonic', cv='prefit')
logit_cal.fit(X_train_reduced_scaled, y_train)
y_pred_logit_cal = logit_cal.predict_proba(X_test_reduced_scaled)[:, 1]
brier_logit_cal = brier_score_loss(y_test, y_pred_logit_cal)

print(f"✓ RF Calibrated: {brier_rf_cal:.4f}, Logit Calibrated: {brier_logit_cal:.4f}")

print("\n[4/4] Creating calibration plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RF calibration
ax1 = axes[0]
fraction_pos_rf, mean_pred_rf = calibration_curve(y_test, y_pred_rf, n_bins=10, strategy='uniform')
fraction_pos_rf_cal, mean_pred_rf_cal = calibration_curve(y_test, y_pred_rf_cal, n_bins=10, strategy='uniform')
ax1.plot(mean_pred_rf, fraction_pos_rf, 's-', label=f'Before (Brier={brier_rf:.4f})', linewidth=2, markersize=8, alpha=0.6)
ax1.plot(mean_pred_rf_cal, fraction_pos_rf_cal, 'o-', label=f'After (Brier={brier_rf_cal:.4f})', linewidth=2, markersize=8, color='green')
ax1.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=1)
ax1.set_xlabel('Mean Predicted Probability', fontweight='bold')
ax1.set_ylabel('Fraction of Positives', fontweight='bold')
ax1.set_title('Random Forest - Calibration', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Logit calibration
ax2 = axes[1]
fraction_pos_logit, mean_pred_logit = calibration_curve(y_test, y_pred_logit, n_bins=10, strategy='uniform')
fraction_pos_logit_cal, mean_pred_logit_cal = calibration_curve(y_test, y_pred_logit_cal, n_bins=10, strategy='uniform')
ax2.plot(mean_pred_logit, fraction_pos_logit, 's-', label=f'Before (Brier={brier_logit:.4f})', linewidth=2, markersize=8, alpha=0.6, color='orange')
ax2.plot(mean_pred_logit_cal, fraction_pos_logit_cal, 'o-', label=f'After (Brier={brier_logit_cal:.4f})', linewidth=2, markersize=8, color='green')
ax2.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=1)
ax2.set_xlabel('Mean Predicted Probability', fontweight='bold')
ax2.set_ylabel('Fraction of Positives', fontweight='bold')
ax2.set_title('Logistic - Calibration', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'calibration_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved calibration plot")

# Save summary
calibration_summary = {
    'rf_brier_before': float(brier_rf),
    'rf_brier_after': float(brier_rf_cal),
    'rf_improvement': float(brier_rf - brier_rf_cal),
    'logit_brier_before': float(brier_logit),
    'logit_brier_after': float(brier_logit_cal),
    'logit_improvement': float(brier_logit - brier_logit_cal),
}

import json
with open(output_dir / 'calibration_summary.json', 'w') as f:
    json.dump(calibration_summary, f, indent=2)

print("\n" + "="*70)
print("✓ SCRIPT 06 COMPLETE")
print(f"  RF: {brier_rf:.4f} → {brier_rf_cal:.4f} ({(brier_rf_cal-brier_rf)/brier_rf*100:+.1f}%)")
print(f"  Logit: {brier_logit:.4f} → {brier_logit_cal:.4f} ({(brier_logit_cal-brier_logit)/brier_logit*100:+.1f}%)")
print("="*70)
