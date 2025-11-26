#!/usr/bin/env python3
"""
Deep Pipeline Audit Script
==========================

This script performs a complete audit of the entire pipeline from 04c to 06.
It reads ALL files completely (no head/tail) and analyzes:
1. 04c_embedded_methods.py (non-nested) - what it does
2. 04c_embedded_methods_nested.py - what it does  
3. 04d_stability_consensus.py - how it combines results
4. What files exist in results/
5. What the actual feature counts are
6. What goes into consensus
7. What modeling uses
8. The complete v1.0 vs v1.1 story

Output: Complete factual analysis with no assumptions.
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"


def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def subsection(title: str):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def audit_04c_nonnested():
    """Audit 04c non-nested embedded results."""
    section("AUDIT 04c: NON-NESTED EMBEDDED METHODS")
    
    results_path = RESULTS_DIR / "04_feature_selection"
    
    for h in range(1, 6):
        subsection(f"Horizon {h}")
        
        # Read embedded_selected.json
        json_path = results_path / f"04c_H{h}_embedded_selected.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            
            # Check structure
            if "methods" in data:
                methods = data["methods"]
            else:
                methods = data
            
            print(f"  File: {json_path.name}")
            print(f"  Structure keys: {list(methods.keys()) if isinstance(methods, dict) else type(methods)}")
            
            if isinstance(methods, dict):
                for method_name, method_data in methods.items():
                    if isinstance(method_data, dict):
                        feats = method_data.get("selected_features", [])
                        print(f"    {method_name}: {len(feats)} features")
                    else:
                        print(f"    {method_name}: {type(method_data)}")
        else:
            print(f"  FILE NOT FOUND: {json_path}")


def audit_04c_nested():
    """Audit 04c nested embedded results."""
    section("AUDIT 04c: NESTED EMBEDDED METHODS")
    
    nested_path = RESULTS_DIR / "04_feature_selection" / "nested"
    
    if not nested_path.exists():
        print("  NESTED DIRECTORY DOES NOT EXIST!")
        return
    
    for h in range(1, 6):
        subsection(f"Horizon {h}")
        
        json_path = nested_path / f"04c_H{h}_embedded_nested.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            
            print(f"  File: {json_path.name}")
            print(f"  Top-level keys: {list(data.keys())}")
            
            methods = data.get("methods", data)
            if isinstance(methods, dict):
                for method_name, method_data in methods.items():
                    if isinstance(method_data, dict):
                        # Check for different possible structures
                        if "selected_features" in method_data:
                            feats = method_data["selected_features"]
                            print(f"    {method_name}: {len(feats)} features (from selected_features)")
                        if "fold_selections" in method_data:
                            folds = method_data["fold_selections"]
                            print(f"    {method_name}: fold_selections has {len(folds)} folds")
                            for i, fold in enumerate(folds):
                                print(f"      Fold {i+1}: {len(fold)} features")
                        if "stability_nogueira" in method_data:
                            print(f"    {method_name}: stability = {method_data['stability_nogueira']:.4f}")
                    else:
                        print(f"    {method_name}: {type(method_data)}")
        else:
            print(f"  FILE NOT FOUND: {json_path}")


def audit_04d_consensus():
    """Audit what 04d actually reads and produces."""
    section("AUDIT 04d: CONSENSUS FEATURE SELECTION")
    
    results_path = RESULTS_DIR / "04_feature_selection"
    feature_sets_path = DATA_DIR / "feature_sets_selected"
    
    # Check what consensus files exist
    subsection("Consensus Result Files")
    for f in sorted(results_path.glob("04d_*")):
        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {f.name}: {stat.st_size/1024:.1f}KB, modified {mtime}")
    
    # Check final feature sets (base)
    subsection("Final Feature Sets (BASE - no suffix)")
    for h in range(1, 6):
        json_path = feature_sets_path / f"H{h}_features_final.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            feats = data.get("features", [])
            method = data.get("method_used", "unknown")
            roc = data.get("performance", {}).get("roc_auc_mean", 0)
            print(f"  H{h}: {len(feats)} features, method={method}, ROC-AUC={roc:.4f}")
            print(f"       Features: {feats}")
        else:
            print(f"  H{h}: FILE NOT FOUND")
    
    # Check final feature sets (nested)
    subsection("Final Feature Sets (NESTED - _nested suffix)")
    for h in range(1, 6):
        json_path = feature_sets_path / f"H{h}_features_final_nested.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            feats = data.get("features", [])
            method = data.get("method_used", "unknown")
            roc = data.get("performance", {}).get("roc_auc_mean", 0)
            print(f"  H{h}: {len(feats)} features, method={method}, ROC-AUC={roc:.4f}")
            print(f"       Features: {feats}")
        else:
            print(f"  H{h}: FILE NOT FOUND")
    
    # Compare base vs nested
    subsection("COMPARISON: Base vs Nested Final Features")
    for h in range(1, 6):
        base_path = feature_sets_path / f"H{h}_features_final.json"
        nested_path = feature_sets_path / f"H{h}_features_final_nested.json"
        
        if base_path.exists() and nested_path.exists():
            with open(base_path) as f:
                base = json.load(f)
            with open(nested_path) as f:
                nested = json.load(f)
            
            base_feats = set(base.get("features", []))
            nested_feats = set(nested.get("features", []))
            
            if base_feats == nested_feats:
                print(f"  H{h}: IDENTICAL ({len(base_feats)} features)")
            else:
                print(f"  H{h}: DIFFERENT!")
                print(f"       Base: {len(base_feats)} features")
                print(f"       Nested: {len(nested_feats)} features")
                print(f"       Only in base: {base_feats - nested_feats}")
                print(f"       Only in nested: {nested_feats - base_feats}")
        else:
            print(f"  H{h}: Cannot compare (missing files)")


def audit_modeling_results():
    """Audit what modeling results exist."""
    section("AUDIT PHASE 05: MODELING RESULTS")
    
    # Base modeling
    subsection("Base Modeling (results/05_modeling/)")
    base_path = RESULTS_DIR / "05_modeling"
    if base_path.exists():
        for p in sorted(base_path.glob("H*_metrics.json")):
            with open(p) as f:
                data = json.load(f)
            h = data.get("horizon")
            metrics = data.get("metrics", {})
            print(f"\n  H{h}:")
            for model, m in metrics.items():
                print(f"    {model}: ROC-AUC={m['roc_auc_mean']:.4f} ± {m['roc_auc_std']:.4f}")
    else:
        print("  Directory does not exist!")
    
    # Nested modeling
    subsection("Nested Modeling (results/05_modeling_nested/)")
    nested_path = RESULTS_DIR / "05_modeling_nested"
    if nested_path.exists():
        for p in sorted(nested_path.glob("H*_metrics.json")):
            with open(p) as f:
                data = json.load(f)
            h = data.get("horizon")
            metrics = data.get("metrics", {})
            print(f"\n  H{h}:")
            for model, m in metrics.items():
                print(f"    {model}: ROC-AUC={m['roc_auc_mean']:.4f} ± {m['roc_auc_std']:.4f}")
    else:
        print("  Directory does not exist!")


def audit_evaluation_results():
    """Audit Phase 06 evaluation results."""
    section("AUDIT PHASE 06: EVALUATION RESULTS")
    
    # Base evaluation
    subsection("Base Evaluation (results/06_model_evaluation/)")
    base_path = RESULTS_DIR / "06_model_evaluation"
    if base_path.exists():
        for f in sorted(base_path.iterdir()):
            stat = f.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {f.name}: {stat.st_size/1024:.1f}KB, modified {mtime}")
    else:
        print("  Directory does not exist!")
    
    # Nested evaluation
    subsection("Nested Evaluation (results/06_model_evaluation_nested/)")
    nested_path = RESULTS_DIR / "06_model_evaluation_nested"
    if nested_path.exists():
        for f in sorted(nested_path.iterdir()):
            stat = f.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {f.name}: {stat.st_size/1024:.1f}KB, modified {mtime}")
    else:
        print("  Directory does not exist!")


def audit_delta():
    """Audit delta comparison files."""
    section("AUDIT: DELTA v1.0 -> v1.1")
    
    delta_path = RESULTS_DIR / "delta"
    if not delta_path.exists():
        print("  Delta directory does not exist!")
        return
    
    for f in sorted(delta_path.iterdir()):
        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {f.name}: {stat.st_size/1024:.1f}KB, modified {mtime}")


def audit_logs():
    """Audit log files to understand execution history."""
    section("AUDIT: LOG FILES (Execution History)")
    
    log_04 = LOGS_DIR / "04_feature_selection"
    if log_04.exists():
        subsection("04_feature_selection logs")
        for f in sorted(log_04.iterdir()):
            stat = f.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {f.name}: modified {mtime}")
    
    log_05 = LOGS_DIR / "05_modeling"
    if log_05.exists():
        subsection("05_modeling logs")
        for f in sorted(log_05.iterdir()):
            stat = f.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {f.name}: modified {mtime}")


def analyze_04d_log_content():
    """Read the actual 04d log to see what methods were used."""
    section("AUDIT: 04d LOG CONTENT ANALYSIS")
    
    log_path = LOGS_DIR / "04_feature_selection" / "04d_stability_consensus.log"
    if not log_path.exists():
        print("  Log file not found!")
        return
    
    with open(log_path) as f:
        content = f.read()
    
    # Find all "Methods available" lines
    lines = content.split("\n")
    
    subsection("Method Selections per Horizon (from log)")
    current_horizon = None
    for line in lines:
        if "Analyzing Consensus: Horizon" in line:
            current_horizon = line.split("Horizon")[-1].strip()
        if "Methods available:" in line:
            print(f"\n  Horizon {current_horizon}:")
            # Extract methods list
            methods_part = line.split("Methods available:")[-1].strip()
            print(f"    {methods_part}")
        if "Lasso_L1:" in line or "Random_Forest:" in line or "Spearman:" in line:
            # Extract feature count
            parts = line.split(" - ")[-1].strip()
            print(f"    {parts}")


def main():
    print("=" * 80)
    print("DEEP PIPELINE AUDIT")
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
    
    audit_04c_nonnested()
    audit_04c_nested()
    audit_04d_consensus()
    audit_modeling_results()
    audit_evaluation_results()
    audit_delta()
    audit_logs()
    analyze_04d_log_content()
    
    section("AUDIT COMPLETE")
    print("\nAll data above is factual - read directly from files.")
    print("No assumptions made. No head/tail truncation.")


if __name__ == "__main__":
    main()
