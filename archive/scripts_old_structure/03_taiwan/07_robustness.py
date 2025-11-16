#!/usr/bin/env python3
"""Taiwan Dataset - Robustness (Bootstrap Validation)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scripts.config import RANDOM_STATE
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
data_dir = PROJECT_ROOT / 'data' / 'processed' / 'taiwan'
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / 'taiwan' / '07_robustness'
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("TAIWAN - Robustness (Bootstrap)")
print("="*80)

df = pd.read_parquet(data_dir / 'taiwan_clean.parquet')
feature_cols = [col for col in df.columns if col.startswith('F')]
X = df[feature_cols].values
y = df['bankrupt'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bootstrap validation
n_bootstrap = 100
aucs = []

print(f"Running {n_bootstrap} bootstrap iterations...")

for i in range(n_bootstrap):
    indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
    X_boot = X_test_scaled[indices]
    y_boot = y_test[indices]
    
    if len(np.unique(y_boot)) < 2:
        continue
    
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE+i)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict_proba(X_boot)[:, 1]
    auc = roc_auc_score(y_boot, y_pred)
    aucs.append(auc)

aucs = np.array(aucs)
mean_auc = aucs.mean()
std_auc = aucs.std()
ci_lower = np.percentile(aucs, 2.5)
ci_upper = np.percentile(aucs, 97.5)

print(f"\nBootstrap AUC: {mean_auc:.4f} ± {std_auc:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

import json
with open(output_dir / 'robustness_results.json', 'w') as f:
    json.dump({
        'mean_auc': float(mean_auc),
        'std_auc': float(std_auc),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_bootstrap': n_bootstrap
    }, f, indent=2)

print(f"\n✓ TAIWAN ROBUSTNESS COMPLETE")
print("="*80)
