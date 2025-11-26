#!/usr/bin/env python3
"""
Check 04d Execution Timeline
==============================
"""

import json
from pathlib import Path
from datetime import datetime
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FS_DIR = PROJECT_ROOT / "results" / "04_feature_selection"
LOG_DIR = PROJECT_ROOT / "logs" / "04_feature_selection"

print("=" * 80)
print("FILE MODIFICATION TIMESTAMPS")
print("=" * 80)

files_to_check = [
    ("04c non-nested H1", FS_DIR / "04c_H1_embedded_selected.json"),
    ("04c nested H1", FS_DIR / "nested" / "04c_H1_embedded_nested.json"),
    ("04d consensus log", LOG_DIR / "04d_stability_consensus.log"),
    ("04d H1 final features", PROJECT_ROOT / "data" / "processed" / "feature_sets_selected" / "H1_features_final.json"),
]

for name, path in files_to_check:
    if path.exists():
        mtime = os.path.getmtime(path)
        dt = datetime.fromtimestamp(mtime)
        print(f"\n{name}:")
        print(f"  Path: {path.relative_to(PROJECT_ROOT)}")
        print(f"  Modified: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\n{name}: NOT FOUND")

# Check final features to see which Lasso was used
print("\n" + "=" * 80)
print("WHICH LASSO WENT INTO FINAL FEATURES?")
print("=" * 80)

final_h1 = PROJECT_ROOT / "data" / "processed" / "feature_sets_selected" / "H1_features_final.json"
if final_h1.exists():
    with open(final_h1) as f:
        data = json.load(f)
    
    consensus_method = data.get("consensus_method", "unknown")
    consensus_feats = data.get("consensus_features", [])
    
    print(f"\nFinal H1:")
    print(f"  Consensus method: {consensus_method}")
    print(f"  Consensus features: {len(consensus_feats)}")
    
    # Check method selections
    method_sels = data.get("method_selections", {})
    if "Lasso_L1" in method_sels:
        lasso_feats = len(method_sels["Lasso_L1"])
        print(f"  Lasso_L1 features: {lasso_feats}")
        
        print(f"\n  Comparison:")
        print(f"    Non-nested Lasso: 35 features")
        print(f"    Nested Lasso:     36 features")
        print(f"    Used in 04d:      {lasso_feats} features")
        
        if lasso_feats == 35:
            print(f"\n  ⚠️  USING NON-NESTED LASSO!")
        elif lasso_feats == 36:
            print(f"\n  ✓  USING NESTED LASSO!")
        else:
            print(f"\n  ❓  UNKNOWN SOURCE!")
