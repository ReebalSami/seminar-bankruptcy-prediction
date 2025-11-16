#!/usr/bin/env python3
"""
Script 12: Cross-Dataset Transfer Learning with SEMANTIC MAPPING

REWRITTEN (Nov 12, 2025) - USES SCRIPT 00 MAPPINGS!

Critical Fix:
- OLD: Used positional matching or top-N features → AUC 0.32 (catastrophic failure)
- NEW: Uses semantic feature alignment from Script 00 → Expected AUC 0.65-0.80

This script ACTUALLY implements the semantic mapping approach:
1. Loads common_features.json from Script 00 (10 features: ROA, Debt_Ratio, etc.)
2. Loads feature_alignment_matrix.csv
3. Extracts SAME 10 semantic features from Polish/American/Taiwan
4. Trains on one dataset, tests on others (6 directions)
5. Reports improvement over positional matching

Author: Reebal
Date: November 12, 2025
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scripts.config import RANDOM_STATE

print("=" * 80)
print("SCRIPT 12: CROSS-DATASET TRANSFER LEARNING (SEMANTIC MAPPING)")
print("=" * 80)
print("\n✅ USING SCRIPT 00 SEMANTIC MAPPINGS!")
print("   Loading: common_features.json, feature_alignment_matrix.csv\n")

# Setup
data_dir = PROJECT_ROOT / 'data' / 'processed'
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / '12_transfer_learning_semantic'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: Load Script 00 Outputs (CRITICAL!)
# ============================================================================
print("[1/6] Loading Script 00 semantic mappings...")

mapping_dir = PROJECT_ROOT / 'results' / '00_feature_mapping'

# Load common semantic features
with open(mapping_dir / 'common_features.json', 'r') as f:
    common_features = json.load(f)

print(f"✓ Common semantic features ({len(common_features)}): {', '.join(common_features)}")

# Load alignment matrix
alignment_matrix = pd.read_csv(mapping_dir / 'feature_alignment_matrix.csv')
print(f"✓ Alignment matrix: {len(alignment_matrix)} mappings")

# Load full semantic mapping
with open(mapping_dir / 'feature_semantic_mapping.json', 'r') as f:
    semantic_mapping = json.load(f)

print(f"✓ Semantic mappings loaded\n")

# ============================================================================
# STEP 2: Load All Datasets
# ============================================================================
print("[2/6] Loading datasets...")

# Polish (Horizon 1)
df_polish = pd.read_parquet(data_dir / 'poland_clean_full.parquet')
df_polish = df_polish[df_polish['horizon'] == 1].copy()
print(f"✓ Polish: {len(df_polish):,} samples")

# American
df_american = pd.read_parquet(data_dir / 'american' / 'american_modeling.parquet')
print(f"✓ American: {len(df_american):,} samples")

# Taiwan
df_taiwan = pd.read_parquet(data_dir / 'taiwan' / 'taiwan_clean.parquet')
print(f"✓ Taiwan: {len(df_taiwan):,} samples\n")

# ============================================================================
# STEP 3: Extract Common Semantic Features from Each Dataset
# ============================================================================
print("[3/6] Extracting common semantic features...")

def extract_semantic_features(df, dataset_name, semantic_mapping):
    """
    Extract common semantic features using Script 00 mappings.
    
    For each semantic feature (ROA, Debt_Ratio, etc.), extract the
    corresponding column(s) from the dataset and take the mean if multiple.
    """
    features_dict = {}
    
    for semantic_feat in common_features:
        # Get dataset-specific column names for this semantic feature
        dataset_cols = semantic_mapping['semantic_mappings'][semantic_feat][dataset_name]
        
        # Handle case where column names might have leading/trailing spaces
        available_cols = []
        for col in dataset_cols:
            # Try exact match
            if col in df.columns:
                available_cols.append(col)
            # Try stripped match
            elif col.strip() in df.columns:
                available_cols.append(col.strip())
            # Try finding similar columns
            else:
                similar = [c for c in df.columns if col.lower().replace(' ', '') in c.lower().replace(' ', '')]
                if similar:
                    available_cols.append(similar[0])
        
        if available_cols:
            # If multiple columns map to same semantic feature, take mean
            if len(available_cols) > 1:
                features_dict[semantic_feat] = df[available_cols].mean(axis=1)
            else:
                features_dict[semantic_feat] = df[available_cols[0]]
        else:
            print(f"  ⚠️  {dataset_name}: Could not find columns for {semantic_feat}")
            print(f"      Looking for: {dataset_cols}")
            # Use zeros as fallback (will hurt performance but won't crash)
            features_dict[semantic_feat] = pd.Series(0, index=df.index)
    
    return pd.DataFrame(features_dict)

# Extract for Polish
X_polish_semantic = extract_semantic_features(df_polish, 'polish', semantic_mapping)
y_polish = df_polish['y']
print(f"✓ Polish: {X_polish_semantic.shape}")

# Extract for American  
X_american_semantic = extract_semantic_features(df_american, 'american', semantic_mapping)
y_american = df_american['bankrupt']
print(f"✓ American: {X_american_semantic.shape}")

# Extract for Taiwan
X_taiwan_semantic = extract_semantic_features(df_taiwan, 'taiwan', semantic_mapping)
y_taiwan = df_taiwan['bankrupt']
print(f"✓ Taiwan: {X_taiwan_semantic.shape}\n")

# ============================================================================
# STEP 4: Transfer Learning Experiments (All 6 Directions)
# ============================================================================
print("[4/6] Running transfer learning experiments...")

# All possible transfer directions
transfer_experiments = [
    ('Polish', 'American', X_polish_semantic, y_polish, X_american_semantic, y_american),
    ('Polish', 'Taiwan', X_polish_semantic, y_polish, X_taiwan_semantic, y_taiwan),
    ('American', 'Polish', X_american_semantic, y_american, X_polish_semantic, y_polish),
    ('American', 'Taiwan', X_american_semantic, y_american, X_taiwan_semantic, y_taiwan),
    ('Taiwan', 'Polish', X_taiwan_semantic, y_taiwan, X_polish_semantic, y_polish),
    ('Taiwan', 'American', X_taiwan_semantic, y_taiwan, X_american_semantic, y_american),
]

results = []

for source, target, X_train, y_train, X_test, y_test in transfer_experiments:
    print(f"\n  {source} → {target}:")
    
    # Handle inf/nan
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.median())
    
    # Standardize (fit on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    
    # Train Random Forest on source dataset
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # Test on target dataset
    y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"    AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
    
    results.append({
        'source': source,
        'target': target,
        'direction': f"{source}→{target}",
        'auc': auc,
        'pr_auc': pr_auc,
        'n_train': len(X_train),
        'n_test': len(X_test)
    })

results_df = pd.DataFrame(results)

print(f"\n✓ Completed {len(results)} transfer experiments")
print(f"  Average AUC: {results_df['auc'].mean():.4f}")
print(f"  Best: {results_df.loc[results_df['auc'].idxmax(), 'direction']} (AUC {results_df['auc'].max():.4f})")

# ============================================================================
# STEP 5: Visualize Transfer Matrix
# ============================================================================
print("\n[5/6] Creating transfer learning heatmap...")

# Create pivot table for heatmap
transfer_matrix = results_df.pivot(index='source', columns='target', values='auc')

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(transfer_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
            vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'ROC-AUC'})
ax.set_title('Cross-Dataset Transfer Learning Performance\n(Semantic Feature Alignment)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Target Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Source Dataset (Training)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'transfer_learning_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: transfer_learning_matrix.png")

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n[6/6] Saving results...")

results_df.to_csv(output_dir / 'transfer_learning_results.csv', index=False)

# Summary statistics
summary = {
    'method': 'Semantic Feature Alignment (Script 00)',
    'common_features': common_features,
    'n_semantic_features': len(common_features),
    'n_experiments': len(results),
    'average_auc': float(results_df['auc'].mean()),
    'std_auc': float(results_df['auc'].std()),
    'min_auc': float(results_df['auc'].min()),
    'max_auc': float(results_df['auc'].max()),
    'best_transfer': results_df.loc[results_df['auc'].idxmax(), 'direction'],
    'results': results
}

with open(output_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Saved: transfer_learning_results.csv")
print(f"✓ Saved: summary.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ SCRIPT 12 COMPLETE - SEMANTIC TRANSFER LEARNING")
print("=" * 80)

print(f"\n📊 RESULTS SUMMARY:")
print(f"   Method: Semantic Feature Alignment (from Script 00)")
print(f"   Common features: {len(common_features)} ({', '.join(common_features[:5])}, ...)")
print(f"   Transfer directions: {len(results)}")
print(f"   Average AUC: {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
print(f"   Range: {results_df['auc'].min():.4f} - {results_df['auc'].max():.4f}")

print(f"\n🎯 INTERPRETATION:")
if results_df['auc'].mean() > 0.60:
    print(f"   ✅ Semantic mapping SUCCESS! Average AUC {results_df['auc'].mean():.4f} > 0.60")
    print(f"   ✅ Cross-dataset knowledge transfer works with aligned features")
elif results_df['auc'].mean() > 0.55:
    print(f"   ⚠️  Moderate success: Average AUC {results_df['auc'].mean():.4f}")
    print(f"   → Dataset differences still significant despite semantic alignment")
else:
    print(f"   ❌ Transfer failed: Average AUC {results_df['auc'].mean():.4f} < 0.55")
    print(f"   → Check feature extraction and semantic mappings")

print(f"\n💡 COMPARISON TO OLD METHOD:")
print(f"   Old (positional matching): ~0.32 AUC")
print(f"   New (semantic alignment): {results_df['auc'].mean():.4f} AUC")
improvement = ((results_df['auc'].mean() - 0.32) / 0.32) * 100
print(f"   Improvement: +{improvement:.1f}%")

print("\n" + "=" * 80)
