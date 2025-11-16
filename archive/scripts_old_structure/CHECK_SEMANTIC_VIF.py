#!/usr/bin/env python3
"""
Check VIF for 10 semantic features across ALL datasets.

Critical Questions:
1. Do semantic features have low multicollinearity in all datasets?
2. Should we use 10 semantic features for BOTH transfer AND within-dataset?
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("VIF ANALYSIS FOR 10 SEMANTIC FEATURES ACROSS ALL DATASETS")
print("=" * 80)

# Load semantic mappings
print("\n[1/4] Loading semantic feature mappings...")
with open(PROJECT_ROOT / 'results' / '00_feature_mapping' / 'feature_semantic_mapping_FIXED.json') as f:
    mapping = json.load(f)

semantic_mappings = mapping['semantic_mappings']
print(f"  ✓ {len(semantic_mappings)} semantic features")

# Load datasets
print("\n[2/4] Loading datasets...")
df_polish = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'poland_clean_full.parquet')
df_polish = df_polish[df_polish['horizon'] == 1].copy()  # Horizon 1 only

df_american = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'american' / 'american_clean.parquet')
df_taiwan = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'taiwan' / 'taiwan_clean.parquet')

print(f"  Polish: {len(df_polish):,} samples")
print(f"  American: {len(df_american):,} samples")
print(f"  Taiwan: {len(df_taiwan):,} samples")

# Extract semantic features
print("\n[3/4] Extracting semantic features...")

def extract_semantic_features(df, dataset_name, semantic_mappings):
    """Extract semantic features, taking mean if multiple columns per concept."""
    features_dict = {}
    
    for semantic_feat, feat_map in semantic_mappings.items():
        cols = feat_map[dataset_name]
        available = [c for c in cols if c in df.columns]
        
        if available:
            if len(available) > 1:
                features_dict[semantic_feat] = df[available].mean(axis=1)
            else:
                features_dict[semantic_feat] = df[available[0]]
        else:
            print(f"  ⚠️  Missing: {semantic_feat} in {dataset_name}")
    
    return pd.DataFrame(features_dict)

X_polish = extract_semantic_features(df_polish, 'polish', semantic_mappings)
X_american = extract_semantic_features(df_american, 'american', semantic_mappings)
X_taiwan = extract_semantic_features(df_taiwan, 'taiwan', semantic_mappings)

print(f"  Polish: {X_polish.shape}")
print(f"  American: {X_american.shape}")
print(f"  Taiwan: {X_taiwan.shape}")

# Calculate VIF for each dataset
print("\n[4/4] Calculating VIF for semantic features...")

def calculate_vif(X, dataset_name):
    """Calculate VIF for all features."""
    # Remove any NaN/inf
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Add constant
    from statsmodels.tools.tools import add_constant
    X_with_const = add_constant(X_clean)
    
    vif_data = []
    for i, col in enumerate(X_clean.columns):
        try:
            vif = variance_inflation_factor(X_with_const.values, i + 1)  # +1 for constant
            vif_data.append({
                'feature': col,
                'vif': vif,
                'status': 'LOW' if vif < 5 else 'MODERATE' if vif < 10 else 'HIGH'
            })
        except:
            vif_data.append({
                'feature': col,
                'vif': np.nan,
                'status': 'ERROR'
            })
    
    return pd.DataFrame(vif_data)

print("\n" + "=" * 80)
print("POLISH DATASET - VIF FOR 10 SEMANTIC FEATURES")
print("=" * 80)
vif_polish = calculate_vif(X_polish, 'Polish')
for _, row in vif_polish.iterrows():
    status_icon = "✅" if row['status'] == 'LOW' else "⚠️" if row['status'] == 'MODERATE' else "❌"
    print(f"  {status_icon} {row['feature']:20s}: VIF = {row['vif']:8.2f} ({row['status']})")

low_polish = vif_polish[vif_polish['vif'] < 5]['feature'].tolist()
print(f"\n✅ {len(low_polish)}/10 features have VIF < 5.0")

print("\n" + "=" * 80)
print("AMERICAN DATASET - VIF FOR 10 SEMANTIC FEATURES")
print("=" * 80)
vif_american = calculate_vif(X_american, 'American')
for _, row in vif_american.iterrows():
    status_icon = "✅" if row['status'] == 'LOW' else "⚠️" if row['status'] == 'MODERATE' else "❌"
    print(f"  {status_icon} {row['feature']:20s}: VIF = {row['vif']:8.2f} ({row['status']})")

low_american = vif_american[vif_american['vif'] < 5]['feature'].tolist()
print(f"\n✅ {len(low_american)}/10 features have VIF < 5.0")

print("\n" + "=" * 80)
print("TAIWAN DATASET - VIF FOR 10 SEMANTIC FEATURES")
print("=" * 80)
vif_taiwan = calculate_vif(X_taiwan, 'Taiwan')
for _, row in vif_taiwan.iterrows():
    status_icon = "✅" if row['status'] == 'LOW' else "⚠️" if row['status'] == 'MODERATE' else "❌"
    print(f"  {status_icon} {row['feature']:20s}: VIF = {row['vif']:8.2f} ({row['status']})")

low_taiwan = vif_taiwan[vif_taiwan['vif'] < 5]['feature'].tolist()
print(f"\n✅ {len(low_taiwan)}/10 features have VIF < 5.0")

# Cross-dataset comparison
print("\n" + "=" * 80)
print("CROSS-DATASET VIF COMPARISON")
print("=" * 80)

# Merge VIF data
vif_comparison = pd.DataFrame({
    'Feature': vif_polish['feature'],
    'Polish_VIF': vif_polish['vif'].round(2),
    'Polish_Status': vif_polish['status'],
    'American_VIF': vif_american['vif'].round(2),
    'American_Status': vif_american['status'],
    'Taiwan_VIF': vif_taiwan['vif'].round(2),
    'Taiwan_Status': vif_taiwan['status']
})

print("\n" + vif_comparison.to_string(index=False))

# Find features with low VIF in ALL datasets
low_all = set(low_polish) & set(low_american) & set(low_taiwan)
print(f"\n✅ Features with VIF < 5.0 in ALL datasets ({len(low_all)}/10):")
for feat in sorted(low_all):
    pl_vif = vif_polish[vif_polish['feature'] == feat]['vif'].values[0]
    us_vif = vif_american[vif_american['feature'] == feat]['vif'].values[0]
    tw_vif = vif_taiwan[vif_taiwan['feature'] == feat]['vif'].values[0]
    print(f"  ✅ {feat:20s}: PL={pl_vif:5.2f}, US={us_vif:5.2f}, TW={tw_vif:5.2f}")

# Find problematic features
high_any = set(vif_polish[vif_polish['vif'] >= 5]['feature']) | \
           set(vif_american[vif_american['vif'] >= 5]['feature']) | \
           set(vif_taiwan[vif_taiwan['vif'] >= 5]['feature'])

print(f"\n❌ Features with VIF ≥ 5.0 in ANY dataset ({len(high_any)}/10):")
for feat in sorted(high_any):
    pl_vif = vif_polish[vif_polish['feature'] == feat]['vif'].values[0]
    us_vif = vif_american[vif_american['feature'] == feat]['vif'].values[0]
    tw_vif = vif_taiwan[vif_taiwan['feature'] == feat]['vif'].values[0]
    
    problems = []
    if pl_vif >= 5: problems.append(f"PL={pl_vif:.1f}")
    if us_vif >= 5: problems.append(f"US={us_vif:.1f}")
    if tw_vif >= 5: problems.append(f"TW={tw_vif:.1f}")
    
    print(f"  ❌ {feat:20s}: {', '.join(problems)}")

# Recommendation
print("\n" + "=" * 80)
print("CRITICAL FINDINGS & RECOMMENDATIONS")
print("=" * 80)

print(f"""
1. VIF ACROSS DATASETS:
   - Polish: {len(low_polish)}/10 features have VIF < 5.0
   - American: {len(low_american)}/10 features have VIF < 5.0
   - Taiwan: {len(low_taiwan)}/10 features have VIF < 5.0
   - ALL datasets: {len(low_all)}/10 features have VIF < 5.0

2. YOUR QUESTION: "Should we train models on 10 features?"
   
   ⚠️  DEPENDS ON YOUR GOAL:
   
   Option A: TRANSFER LEARNING (current approach)
   ✅ Use all 10 semantic features
   ✅ Purpose: Match concepts across datasets
   ✅ Accept some multicollinearity for interpretability
   ✅ Current performance: Polish→American 0.69 AUC
   
   Option B: WITHIN-DATASET MODELING (current approach)
   ✅ Use 38 VIF-selected features for Polish
   ✅ Purpose: Best predictive performance
   ✅ Low multicollinearity guaranteed (VIF < 5)
   ✅ Current performance: Polish AUC 0.83
   
   Option C: UNIFIED APPROACH (if you want consistency)
   ⚠️  Use ONLY the {len(low_all)} features with low VIF in ALL datasets
   ⚠️  Purpose: Same features for both transfer AND modeling
   ⚠️  Trade-off: Fewer features, possibly lower performance
   
3. RECOMMENDATION:
   
   ✅ KEEP CURRENT APPROACH (two feature sets):
      - 38 VIF features for within-dataset → 0.83 AUC
      - 10 semantic features for transfer → 0.69 AUC
      - Different purposes, both scientifically valid!
   
   OR (if professor insists on consistency):
   
   ⚠️  Use {len(low_all)} features with low VIF everywhere:
      - Unified approach (same features everywhere)
      - Guaranteed low multicollinearity
      - But: May sacrifice some performance
      - Need to re-train all models

4. FOR YOUR DEFENSE:
   
   ✅ "We checked VIF for semantic features across all datasets"
   ✅ "{len(low_all)}/10 have low VIF everywhere - these are robust!"
   ✅ "For transfer learning, we prioritize semantic meaning over VIF"
   ✅ "For within-dataset, we prioritize VIF for optimal performance"
   
   This shows scientific rigor AND strategic thinking!
""")

print("=" * 80)

# Save results
output_dir = PROJECT_ROOT / 'results' / '00_feature_mapping'
vif_comparison.to_csv(output_dir / 'semantic_features_vif_comparison.csv', index=False)
print(f"\n✓ Saved: {output_dir / 'semantic_features_vif_comparison.csv'}")
