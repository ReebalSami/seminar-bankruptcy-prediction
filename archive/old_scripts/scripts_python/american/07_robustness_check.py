#!/usr/bin/env python3
"""American Dataset - Robustness Check (Cross-Year Validation)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scripts.config import RANDOM_STATE
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
data_dir = project_root / 'data' / 'NYSE-and-NASDAQ-companies'
output_dir = project_root / 'results' / 'script_outputs' / 'american' / '07_robustness'
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("AMERICAN - Robustness Check (Cross-Year)")
print("="*80)

df = pd.read_csv(data_dir / 'american-bankruptcy.csv')
df['status_binary'] = df['status_label'].apply(lambda x: 1 if 'failed' in str(x).lower() else 0)

feature_cols = [col for col in df.columns if col.startswith('X')]
X = df[feature_cols]
y = df['status_binary']
years = df['year']

X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

unique_years = sorted(years.unique())
print(f"\nYears available: {unique_years[:5]}...{unique_years[-5:]}")

# Cross-year validation: train on 2015-2017, test on 2018
train_mask = years.isin([2015, 2016, 2017])
test_mask = years == 2018

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTrain: {len(X_train):,} samples (years 2015-2017)")
print(f"Test:  {len(X_test):,} samples (year 2018)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_train)

y_pred = rf.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred)

print(f"\n✓ Cross-Year AUC: {auc:.4f}")

import json
with open(output_dir / 'robustness_results.json', 'w') as f:
    json.dump({'cross_year_auc': float(auc)}, f, indent=2)

print(f"\n✓ AMERICAN ROBUSTNESS CHECK COMPLETE")
print("="*80)
