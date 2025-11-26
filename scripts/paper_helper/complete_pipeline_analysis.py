#!/usr/bin/env python3
"""
COMPLETE PIPELINE ANALYSIS
===========================
Analyzes EVERYTHING to understand the full pipeline flow.
"""

import json
from pathlib import Path
from datetime import datetime
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_file_timestamp(path):
    """Get file modification timestamp."""
    if path.exists():
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    return "NOT FOUND"

def analyze_all_scripts():
    """List ALL scripts in order."""
    print("=" * 80)
    print("ALL SCRIPTS IN PIPELINE")
    print("=" * 80)
    
    scripts_dir = PROJECT_ROOT / "scripts"
    
    for phase_dir in sorted(scripts_dir.iterdir()):
        if not phase_dir.is_dir() or phase_dir.name.startswith("__") or phase_dir.name.startswith("."):
            continue
        
        print(f"\n{phase_dir.name}/")
        scripts = sorted([f for f in phase_dir.glob("*.py") if not f.name.startswith("__")])
        for script in scripts:
            print(f"  {script.name}")

def analyze_all_results():
    """Analyze ALL results directories."""
    print("\n" + "=" * 80)
    print("ALL RESULTS DIRECTORIES")
    print("=" * 80)
    
    results_dir = PROJECT_ROOT / "results"
    
    for subdir in sorted(results_dir.iterdir()):
        if subdir.is_dir():
            files = list(subdir.rglob("*"))
            files = [f for f in files if f.is_file()]
            
            print(f"\n{subdir.name}/ ({len(files)} files)")
            
            # Show file types
            by_ext = {}
            for f in files:
                ext = f.suffix or "no_ext"
                by_ext[ext] = by_ext.get(ext, 0) + 1
            print(f"  Types: {dict(sorted(by_ext.items()))}")
            
            # Show first 5 files with timestamps
            for f in sorted(files)[:5]:
                rel = f.relative_to(subdir)
                ts = get_file_timestamp(f)
                print(f"  {rel} [{ts}]")

def analyze_04c_complete():
    """Complete analysis of 04c embedded methods."""
    print("\n" + "=" * 80)
    print("04c EMBEDDED METHODS - COMPLETE ANALYSIS")
    print("=" * 80)
    
    fs_dir = PROJECT_ROOT / "results" / "04_feature_selection"
    
    print("\n--- NON-NESTED (04c_embedded_methods.py) ---")
    for h in range(1, 6):
        file = fs_dir / f"04c_H{h}_embedded_selected.json"
        if not file.exists():
            continue
        
        with open(file) as f:
            data = json.load(f)
        
        methods = data.get("methods", {})
        print(f"\nH{h}:")
        for m in ["lasso", "elastic_net", "ridge", "random_forest"]:
            if m in methods:
                feats = methods[m].get("selected_features", [])
                print(f"  {m:15s}: {len(feats):2d} features")
    
    print("\n--- NESTED (04c_embedded_methods_nested.py) ---")
    for h in range(1, 6):
        file = fs_dir / "nested" / f"04c_H{h}_embedded_nested.json"
        if not file.exists():
            continue
        
        with open(file) as f:
            data = json.load(f)
        
        methods = data.get("methods", {})
        print(f"\nH{h}:")
        for m in ["lasso", "elastic_net", "ridge"]:
            if m in methods:
                final = methods[m].get("final_features_majority", [])
                print(f"  {m:15s}: {len(final):2d} features (majority vote)")

def analyze_04d_complete():
    """Complete analysis of what 04d used."""
    print("\n" + "=" * 80)
    print("04d CONSENSUS - WHAT WAS ACTUALLY USED")
    print("=" * 80)
    
    final_dir = PROJECT_ROOT / "data" / "processed" / "feature_sets_selected"
    
    for h in range(1, 6):
        file = final_dir / f"H{h}_features_final.json"
        if not file.exists():
            continue
        
        with open(file) as f:
            data = json.load(f)
        
        print(f"\nH{h}:")
        print(f"  Final features: {data.get('count', 0)}")
        print(f"  Method used: {data.get('method_used', 'unknown')}")
        print(f"  Timestamp: {get_file_timestamp(file)}")

def analyze_modeling_complete():
    """Complete analysis of modeling."""
    print("\n" + "=" * 80)
    print("MODELING - NON-NESTED VS NESTED")
    print("=" * 80)
    
    print("\n--- NON-NESTED (05_modeling_train_evaluate.py) ---")
    mod_dir = PROJECT_ROOT / "results" / "05_modeling"
    for h in range(1, 6):
        file = mod_dir / f"H{h}_metrics.json"
        if not file.exists():
            continue
        
        with open(file) as f:
            data = json.load(f)
        
        print(f"\nH{h}:")
        for model_name, metrics in list(data.items())[:3]:
            if isinstance(metrics, dict) and "roc_auc_mean" in metrics:
                auc = metrics.get("roc_auc_mean", 0)
                print(f"  {model_name:20s}: AUC={auc:.3f}")
    
    print("\n--- NESTED (results from nested feature selection) ---")
    mod_nested_dir = PROJECT_ROOT / "results" / "05_modeling_nested"
    if mod_nested_dir.exists():
        for h in range(1, 6):
            file = mod_nested_dir / f"H{h}_metrics.json"
            if not file.exists():
                continue
            
            with open(file) as f:
                data = json.load(f)
            
            print(f"\nH{h}:")
            for model_name, metrics in list(data.items())[:3]:
                if isinstance(metrics, dict) and "roc_auc_mean" in metrics:
                    auc = metrics.get("roc_auc_mean", 0)
                    print(f"  {model_name:20s}: AUC={auc:.3f}")

def analyze_delta():
    """Analyze delta between v1.0 and v1.1."""
    print("\n" + "=" * 80)
    print("DELTA v1.0 -> v1.1")
    print("=" * 80)
    
    delta_file = PROJECT_ROOT / "results" / "delta" / "v1_to_v1_1_delta.xlsx"
    print(f"\nDelta file exists: {delta_file.exists()}")
    if delta_file.exists():
        print(f"Modified: {get_file_timestamp(delta_file)}")

def main():
    print("STARTING COMPLETE PIPELINE ANALYSIS")
    print("=" * 80)
    
    analyze_all_scripts()
    analyze_all_results()
    analyze_04c_complete()
    analyze_04d_complete()
    analyze_modeling_complete()
    analyze_delta()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
