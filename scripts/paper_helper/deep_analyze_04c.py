#!/usr/bin/env python3
"""
Deep Analysis: 04c Embedded Methods
====================================
Checks what's actually in non-nested vs nested embedded results.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FS_DIR = PROJECT_ROOT / "results" / "04_feature_selection"

def analyze_04c_nonnested():
    """Analyze non-nested 04c results."""
    print("=" * 80)
    print("04c NON-NESTED EMBEDDED METHODS")
    print("=" * 80)
    
    for h in range(1, 6):
        file = FS_DIR / f"04c_H{h}_embedded_selected.json"
        if not file.exists():
            continue
            
        print(f"\n{'='*40}")
        print(f"HORIZON {h}")
        print(f"{'='*40}")
        
        with open(file) as f:
            data = json.load(f)
        
        methods = data.get("methods", {})
        
        for method_name in ["lasso", "elastic_net", "ridge", "random_forest"]:
            if method_name in methods:
                method_data = methods[method_name]
                selected = method_data.get("selected_features", [])
                print(f"\n{method_name}:")
                print(f"  Selected features: {len(selected)}")
                if len(selected) > 0 and len(selected) <= 5:
                    print(f"  Features: {selected}")

def analyze_04c_nested():
    """Analyze nested 04c results."""
    print("\n" + "=" * 80)
    print("04c NESTED EMBEDDED METHODS")
    print("=" * 80)
    
    for h in range(1, 6):
        file = FS_DIR / "nested" / f"04c_H{h}_embedded_nested.json"
        if not file.exists():
            continue
            
        print(f"\n{'='*40}")
        print(f"HORIZON {h}")
        print(f"{'='*40}")
        
        with open(file) as f:
            data = json.load(f)
        
        methods = data.get("methods", {})
        
        for method_name in ["lasso", "elastic_net", "ridge"]:
            if method_name in methods:
                method_data = methods[method_name]
                
                # Check different possible keys
                final_feats = method_data.get("final_features_majority", [])
                selected_feats = method_data.get("selected_features", [])
                fold_sels = method_data.get("fold_selections", [])
                
                print(f"\n{method_name}:")
                print(f"  final_features_majority: {len(final_feats)}")
                print(f"  selected_features: {len(selected_feats)}")
                print(f"  fold_selections: {len(fold_sels)} folds")
                
                if len(fold_sels) > 0:
                    fold_lens = [len(f) for f in fold_sels]
                    print(f"  Features per fold: {fold_lens}")

def check_which_went_to_consensus():
    """Check which method actually went to consensus."""
    print("\n" + "=" * 80)
    print("WHAT WENT INTO CONSENSUS (04d)")
    print("=" * 80)
    
    # Check logs
    log_file = PROJECT_ROOT / "logs" / "04_feature_selection" / "04d_stability_consensus.log"
    
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()
        
        # Find "Methods available" lines
        for i, line in enumerate(lines):
            if "Methods available:" in line:
                print(f"\nLine {i}: {line.strip()}")
                # Print next few lines to see feature counts
                for j in range(1, min(8, len(lines)-i)):
                    next_line = lines[i+j].strip()
                    if next_line and (":" in next_line or "features" in next_line):
                        print(f"  {next_line}")
                    if "Computing" in next_line or "[" in next_line:
                        break
                break  # Only show first occurrence per horizon

def main():
    analyze_04c_nonnested()
    analyze_04c_nested()
    check_which_went_to_consensus()

if __name__ == "__main__":
    main()
