"""
Verify Phase 03 Facts

Extract and validate all facts from Phase 03 VIF analysis outputs.
Creates phase03_facts.json for paper writing.

Validates:
- Count consistency: Initial - Removed = Final
- VIF thresholds met
- File existence
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import json


def verify_horizon_facts(horizon, results_dir):
    """Extract and verify facts for one horizon."""
    excel_path = results_dir / f'03a_H{horizon}_vif.xlsx'
    json_path = PROJECT_ROOT / 'data' / 'processed' / 'feature_sets' / f'H{horizon}_features.json'
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    
    # Load Excel sheets
    metadata = pd.read_excel(excel_path, sheet_name='Metadata')
    iterations = pd.read_excel(excel_path, sheet_name='Iterations')
    final_vif = pd.read_excel(excel_path, sheet_name='Final_VIF')
    removed = pd.read_excel(excel_path, sheet_name='Removed_Features')
    
    # Load JSON
    with open(json_path, 'r') as f:
        features_list = json.load(f)
    
    # Extract facts
    meta_row = metadata.iloc[0]
    facts = {
        'horizon': int(meta_row['Horizon']),
        'initial_features': int(meta_row['Initial_Features']),
        'final_features': int(meta_row['Final_Features']),
        'removed_count': int(meta_row['Removed_Count']),
        'iterations': int(meta_row['Iterations']),
        'max_final_vif': float(meta_row['Max_Final_VIF']),
        'json_feature_count': len(features_list)
    }
    
    # Validation
    assert facts['initial_features'] - facts['removed_count'] == facts['final_features'], \
        f"H{horizon}: Count mismatch! {facts['initial_features']} - {facts['removed_count']} ≠ {facts['final_features']}"
    
    assert facts['final_features'] == facts['json_feature_count'], \
        f"H{horizon}: Excel final ({facts['final_features']}) ≠ JSON count ({facts['json_feature_count']})"
    
    assert facts['max_final_vif'] <= 10.0, \
        f"H{horizon}: Max VIF ({facts['max_final_vif']:.2f}) > 10!"
    
    # Extract top-5 worst initial VIFs
    iterations_df = pd.DataFrame(iterations)
    first_iter = iterations_df[iterations_df['Iteration'] == 1]
    if not first_iter.empty and first_iter.iloc[0]['Removed_Feature'] is not None:
        facts['first_removed_feature'] = first_iter.iloc[0]['Removed_Feature']
        facts['first_removed_vif'] = float(first_iter.iloc[0]['Removed_VIF'])
    else:
        facts['first_removed_feature'] = None
        facts['first_removed_vif'] = None
    
    # Removed features
    facts['removed_features'] = removed['Feature'].tolist() if not removed.empty else []
    
    # Final features
    facts['final_features_list'] = features_list
    
    print(f"✓ H{horizon}: {facts['initial_features']} → {facts['final_features']} ({facts['removed_count']} removed, {facts['iterations']} iters, max VIF={facts['max_final_vif']:.2f})")
    
    return facts


def compute_cross_horizon_stats(all_facts):
    """Compute statistics across all horizons."""
    total_removed = sum(f['removed_count'] for f in all_facts)
    avg_iterations = sum(f['iterations'] for f in all_facts) / len(all_facts)
    max_vif_overall = max(f['max_final_vif'] for f in all_facts)
    
    # Feature overlap analysis
    all_final_features = [set(f['final_features_list']) for f in all_facts]
    common_features = set.intersection(*all_final_features)
    
    # Features unique to each horizon
    unique_per_horizon = {}
    for i, facts in enumerate(all_facts, 1):
        h_features = set(facts['final_features_list'])
        other_features = set.union(*[all_final_features[j] for j in range(5) if j != i-1])
        unique = h_features - other_features
        unique_per_horizon[f'H{i}'] = list(unique)
    
    cross_stats = {
        'total_removed': total_removed,
        'avg_iterations': round(avg_iterations, 2),
        'max_vif_overall': round(max_vif_overall, 2),
        'common_features_count': len(common_features),
        'common_features': sorted(list(common_features)),
        'unique_per_horizon': unique_per_horizon
    }
    
    print(f"\n📊 Cross-Horizon Statistics:")
    print(f"  Total removed: {total_removed}")
    print(f"  Avg iterations: {avg_iterations:.1f}")
    print(f"  Max VIF (overall): {max_vif_overall:.2f}")
    print(f"  Common features across all horizons: {len(common_features)}")
    
    return cross_stats


def main():
    """Main verification function."""
    print("=" * 60)
    print("Phase 03 Fact Verification".center(60))
    print("=" * 60)
    print()
    
    results_dir = PROJECT_ROOT / 'results' / '03_multicollinearity'
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Verify each horizon
    all_facts = []
    for horizon in range(1, 6):
        facts = verify_horizon_facts(horizon, results_dir)
        all_facts.append(facts)
    
    # Cross-horizon stats
    cross_stats = compute_cross_horizon_stats(all_facts)
    
    # Combine all facts
    output = {
        'phase': '03_multicollinearity',
        'script': '03a_vif_analysis.py',
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'per_horizon': all_facts,
        'cross_horizon': cross_stats
    }
    
    # Save to JSON
    output_path = Path(__file__).parent / 'phase03_facts.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Verification complete!")
    print(f"✓ All validations passed")
    print(f"✓ Facts saved to: {output_path.name}")
    print(f"\n{'=' * 60}")


if __name__ == '__main__':
    main()
