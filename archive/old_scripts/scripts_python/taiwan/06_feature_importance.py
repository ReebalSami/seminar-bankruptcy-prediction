#!/usr/bin/env python3
"""Taiwan Dataset - Feature Importance"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scripts.config import RANDOM_STATE
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
data_dir = project_root / 'data' / 'processed' / 'taiwan'
output_dir = project_root / 'results' / 'script_outputs' / 'taiwan' / '06_importance'
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("TAIWAN - Feature Importance")
print("="*80)

df = pd.read_parquet(data_dir / 'taiwan_clean.parquet')
feature_cols = [col for col in df.columns if col.startswith('F')]
X = df[feature_cols].values
y = df['bankrupt'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_train)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Features:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

print(f"\n✓ TAIWAN FEATURE IMPORTANCE COMPLETE")
print("="*80)
