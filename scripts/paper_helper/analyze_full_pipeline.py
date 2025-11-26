#!/usr/bin/env python3
"""
Complete Pipeline Analysis
===========================
Analyzes the entire pipeline to understand:
1. What files exist in each phase
2. Execution order
3. Dependencies between phases
4. v1.0 vs v1.1 differences
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def analyze_phase_results():
    """Analyze all phase results directories."""
    phases = {
        "04_feature_selection": PROJECT_ROOT / "results" / "04_feature_selection",
        "05_modeling": PROJECT_ROOT / "results" / "05_modeling",
        "05_modeling_nested": PROJECT_ROOT / "results" / "05_modeling_nested",
        "06_model_evaluation": PROJECT_ROOT / "results" / "06_model_evaluation",
        "06_model_evaluation_nested": PROJECT_ROOT / "results" / "06_model_evaluation_nested",
    }
    
    print("=" * 80)
    print("PIPELINE RESULTS ANALYSIS")
    print("=" * 80)
    
    for name, path in phases.items():
        print(f"\n{name}:")
        if path.exists():
            files = sorted(path.rglob("*"))
            files = [f for f in files if f.is_file()]
            print(f"  Exists: YES ({len(files)} files)")
            
            # Group by extension
            by_ext = {}
            for f in files:
                ext = f.suffix or "no_extension"
                by_ext[ext] = by_ext.get(ext, 0) + 1
            
            print(f"  File types: {dict(sorted(by_ext.items()))}")
            
            # Show first few files
            for f in files[:5]:
                rel = f.relative_to(path)
                print(f"    - {rel}")
            if len(files) > 5:
                print(f"    ... and {len(files)-5} more")
        else:
            print(f"  Exists: NO")

def analyze_scripts():
    """Analyze all scripts to understand execution order."""
    print("\n" + "=" * 80)
    print("SCRIPTS ANALYSIS")
    print("=" * 80)
    
    scripts_dir = PROJECT_ROOT / "scripts"
    
    phases = sorted([d for d in scripts_dir.iterdir() if d.is_dir() and not d.name.startswith("__")])
    
    for phase in phases:
        print(f"\n{phase.name}:")
        scripts = sorted([f for f in phase.glob("*.py") if not f.name.startswith("__")])
        for script in scripts:
            print(f"  - {script.name}")

def check_nested_vs_nonnested():
    """Check for nested vs non-nested results."""
    print("\n" + "=" * 80)
    print("NESTED VS NON-NESTED COMPARISON")
    print("=" * 80)
    
    # Check 04c results
    fs_dir = PROJECT_ROOT / "results" / "04_feature_selection"
    
    print("\n04c Embedded Methods:")
    for h in range(1, 6):
        embedded_file = fs_dir / f"04c_H{h}_embedded_selected.json"
        nested_file = fs_dir / "nested" / f"04c_H{h}_embedded_nested.json"
        
        print(f"\n  H{h}:")
        print(f"    Non-nested: {embedded_file.exists()}")
        print(f"    Nested:     {nested_file.exists()}")
        
        if embedded_file.exists():
            with open(embedded_file) as f:
                data = json.load(f)
            methods = data.get("methods", {})
            print(f"    Non-nested methods: {list(methods.keys())}")
        
        if nested_file.exists():
            with open(nested_file) as f:
                data = json.load(f)
            methods = data.get("methods", {})
            print(f"    Nested methods: {list(methods.keys())}")

def check_modeling_results():
    """Check modeling results to understand v1.0 vs v1.1."""
    print("\n" + "=" * 80)
    print("MODELING RESULTS")
    print("=" * 80)
    
    modeling_dir = PROJECT_ROOT / "results" / "05_modeling"
    modeling_nested_dir = PROJECT_ROOT / "results" / "05_modeling_nested"
    
    print("\nNon-nested modeling:")
    if modeling_dir.exists():
        metrics = sorted(modeling_dir.glob("H*_metrics.json"))
        print(f"  Found {len(metrics)} horizon metrics files")
        for m in metrics[:3]:
            print(f"    - {m.name}")
    
    print("\nNested modeling:")
    if modeling_nested_dir.exists():
        metrics = sorted(modeling_nested_dir.glob("H*_metrics.json"))
        print(f"  Found {len(metrics)} horizon metrics files")
        for m in metrics[:3]:
            print(f"    - {m.name}")

def main():
    analyze_phase_results()
    analyze_scripts()
    check_nested_vs_nonnested()
    check_modeling_results()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
