#!/usr/bin/env python3
"""
Verify the difference between base and nested embedded selections.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
base_dir = PROJECT_ROOT / "results/04_feature_selection"
nested_dir = PROJECT_ROOT / "results/04_feature_selection/nested"

print("=" * 80)
print("EMBEDDED METHOD SELECTIONS: BASE vs NESTED")
print("=" * 80)

for h in range(1, 6):
    print(f"\n{'='*80}")
    print(f"H{h} EMBEDDED SELECTIONS")
    print(f"{'='*80}")
    
    # Load base embedded
    base_file = base_dir / f"04c_H{h}_embedded_selected.json"
    with open(base_file) as f:
        base_data = json.load(f)
    
    # Load nested embedded
    nested_file = nested_dir / f"04c_H{h}_embedded_nested.json"
    with open(nested_file) as f:
        nested_data = json.load(f)
    
    # Compare each method
    for method in ["lasso", "elastic_net", "ridge"]:
        base_feats = set(base_data.get("methods", {}).get(method, {}).get("selected_features", []))
        nested_feats = set(nested_data.get("methods", {}).get(method, {}).get("selected_features", []))
        
        diff = base_feats.symmetric_difference(nested_feats)
        
        print(f"\n  {method.upper()}:")
        print(f"    BASE: {len(base_feats)} features")
        print(f"    NESTED: {len(nested_feats)} features")
        if diff:
            only_base = base_feats - nested_feats
            only_nested = nested_feats - base_feats
            if only_base:
                print(f"    Only in BASE: {sorted(only_base)[:10]}...")
            if only_nested:
                print(f"    Only in NESTED: {sorted(only_nested)[:10]}...")
        else:
            print(f"    ✓ Identical!")
