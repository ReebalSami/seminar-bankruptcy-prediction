#!/usr/bin/env python3
"""Data Preparation - Train/test splits and scaling"""

import sys
from pathlib import Path
# Add project root to path (scripts/01_polish -> root is 2 levels up)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.bankruptcy_prediction.data import DataLoader

# Setup
# PROJECT_ROOT already defined as PROJECT_ROOT above
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / '03_data_preparation'
output_dir.mkdir(parents=True, exist_ok=True)
splits_dir = PROJECT_ROOT / 'data' / 'processed' / 'splits'
splits_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("SCRIPT 03: Data Preparation")
print("="*70)

loader = DataLoader()

print("\n[1/3] Loading datasets...")
df_full = loader.load_poland(horizon=1, dataset_type='full')
df_reduced = loader.load_poland(horizon=1, dataset_type='reduced')

X_full, y = loader.get_features_target(df_full)
X_reduced, _ = loader.get_features_target(df_reduced)

print(f"✓ Full: {X_full.shape}, Reduced: {X_reduced.shape}")

print("\n[2/3] Creating train/test splits...")
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)
X_train_reduced, X_test_reduced, _, _ = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train: {len(y_train):,}, Test: {len(y_test):,}")
print(f"  Train bankrupt: {y_train.mean():.2%}")
print(f"  Test bankrupt: {y_test.mean():.2%}")

print("\n[3/3] Scaling features...")
scaler_full = StandardScaler()
scaler_reduced = StandardScaler()

X_train_full_scaled = pd.DataFrame(
    scaler_full.fit_transform(X_train_full),
    columns=X_train_full.columns, index=X_train_full.index
)
X_test_full_scaled = pd.DataFrame(
    scaler_full.transform(X_test_full),
    columns=X_test_full.columns, index=X_test_full.index
)

X_train_reduced_scaled = pd.DataFrame(
    scaler_reduced.fit_transform(X_train_reduced),
    columns=X_train_reduced.columns, index=X_train_reduced.index
)
X_test_reduced_scaled = pd.DataFrame(
    scaler_reduced.transform(X_test_reduced),
    columns=X_test_reduced.columns, index=X_test_reduced.index
)

print("✓ Scaling complete (mean=0, std=1)")

# Save splits
print("\nSaving splits...")
X_train_full.to_parquet(splits_dir / 'X_train_full.parquet')
X_test_full.to_parquet(splits_dir / 'X_test_full.parquet')
X_train_reduced.to_parquet(splits_dir / 'X_train_reduced.parquet')
X_test_reduced.to_parquet(splits_dir / 'X_test_reduced.parquet')
X_train_full_scaled.to_parquet(splits_dir / 'X_train_full_scaled.parquet')
X_test_full_scaled.to_parquet(splits_dir / 'X_test_full_scaled.parquet')
X_train_reduced_scaled.to_parquet(splits_dir / 'X_train_reduced_scaled.parquet')
X_test_reduced_scaled.to_parquet(splits_dir / 'X_test_reduced_scaled.parquet')
y_train.to_frame('y').to_parquet(splits_dir / 'y_train.parquet')
y_test.to_frame('y').to_parquet(splits_dir / 'y_test.parquet')

# Save summary
summary = {
    'total_samples': len(X_full),
    'train_samples': len(X_train_full),
    'test_samples': len(X_test_full),
    'features_full': len(X_full.columns),
    'features_reduced': len(X_reduced.columns),
    'train_bankruptcy_rate': float(y_train.mean()),
    'test_bankruptcy_rate': float(y_test.mean()),
}

import json
with open(output_dir / 'preparation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("✓ SCRIPT 03 COMPLETE")
print(f"  Splits saved to: {splits_dir}")
print("="*70)
