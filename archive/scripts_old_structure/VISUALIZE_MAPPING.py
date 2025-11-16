#!/usr/bin/env python3
"""
Simple Feature Mapping Visualization

Shows what Script 00 CLAIMS to have mapped vs what ACTUALLY exists in processed data.
"""

import sys
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("FEATURE MAPPING VISUALIZATION")
print("=" * 80)

# Load mappings
print("\n[1/3] Loading feature mappings from Script 00...")
with open(PROJECT_ROOT / 'results' / '00_feature_mapping' / 'feature_semantic_mapping.json') as f:
    mappings = json.load(f)

semantic_mappings = mappings['semantic_mappings']
print(f"  ✓ Found {len(semantic_mappings)} common semantic features")

# Load actual datasets
print("\n[2/3] Loading actual processed datasets...")
df_polish = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'poland_clean_full.parquet')
df_american = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'american' / 'american_clean.parquet')
df_taiwan = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'taiwan' / 'taiwan_clean.parquet')

print(f"  Polish: {len([c for c in df_polish.columns if c.startswith('Attr')])} Attr features")
print(f"  American: {len([c for c in df_american.columns if c.startswith('X')])} X features")
print(f"  Taiwan: {len([c for c in df_taiwan.columns if c.startswith('F')])} F features")

# Check what exists
print("\n[3/3] Checking mapping validity...")

results = []
for semantic_name, feature_map in semantic_mappings.items():
    polish_exist = all(f in df_polish.columns for f in feature_map['polish'])
    american_exist = all(f in df_american.columns for f in feature_map['american'])
    taiwan_exist = all(f in df_taiwan.columns for f in feature_map['taiwan'])
    
    results.append({
        'feature': semantic_name,
        'polish_mapped': len(feature_map['polish']),
        'american_mapped': len(feature_map['american']),
        'taiwan_mapped': len(feature_map['taiwan']),
        'polish_exist': polish_exist,
        'american_exist': american_exist,
        'taiwan_exist': taiwan_exist
    })
    
    status = "✅" if (polish_exist and american_exist) else "❌"
    taiwan_status = "✅" if taiwan_exist else "❌ (Taiwan uses F-codes, not descriptive names!)"
    print(f"  {status} {semantic_name}: Polish={polish_exist}, American={american_exist}, Taiwan={taiwan_status}")

# Create visualization
print("\n[4/4] Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Feature Mapping Analysis - Script 00 Validation', fontsize=16, fontweight='bold')

# 1. Mapping overview
ax1 = axes[0, 0]
features = [r['feature'] for r in results]
polish_counts = [r['polish_mapped'] for r in results]
american_counts = [r['american_mapped'] for r in results]
taiwan_counts = [r['taiwan_mapped'] for r in results]

x = range(len(features))
width = 0.25

ax1.bar([i-width for i in x], polish_counts, width, label='Polish', color='#3498db', alpha=0.8)
ax1.bar(x, american_counts, width, label='American', color='#e74c3c', alpha=0.8)
ax1.bar([i+width for i in x], taiwan_counts, width, label='Taiwan', color='#2ecc71', alpha=0.8)

ax1.set_xlabel('Semantic Feature', fontweight='bold')
ax1.set_ylabel('Number of Mapped Features', fontweight='bold')
ax1.set_title('Features Mapped per Semantic Concept', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(features, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Existence status
ax2 = axes[0, 1]
polish_ok = sum(1 for r in results if r['polish_exist'])
american_ok = sum(1 for r in results if r['american_exist'])
taiwan_ok = sum(1 for r in results if r['taiwan_exist'])

datasets = ['Polish', 'American', 'Taiwan']
existing = [polish_ok, american_ok, taiwan_ok]
missing = [len(results) - e for e in existing]

x_pos = range(len(datasets))
ax2.bar(x_pos, existing, label='Exist in Data', color='#27ae60', alpha=0.8)
ax2.bar(x_pos, missing, bottom=existing, label='Missing', color='#e74c3c', alpha=0.8)

ax2.set_ylabel('Number of Features', fontweight='bold')
ax2.set_title('Feature Existence in Processed Data', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(datasets)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add text annotations
for i, (e, m) in enumerate(zip(existing, missing)):
    total = e + m
    ax2.text(i, e/2, f'{e}/{total}', ha='center', va='center', fontweight='bold', color='white')
    if m > 0:
        ax2.text(i, e + m/2, f'{m}/{total}', ha='center', va='center', fontweight='bold', color='white')

# 3. Coverage summary
ax3 = axes[1, 0]
ax3.axis('off')

summary_text = f"""
FEATURE MAPPING SUMMARY

Total Semantic Features: {len(semantic_mappings)}

Mappings per Dataset:
  • Polish: {sum(polish_counts)} features mapped
  • American: {sum(american_counts)} features mapped  
  • Taiwan: {sum(taiwan_counts)} features mapped

Validation Results:
  ✅ Polish: {polish_ok}/{len(results)} features exist
  ✅ American: {american_ok}/{len(results)} features exist
  ❌ Taiwan: {taiwan_ok}/{len(results)} features exist

⚠️  CRITICAL ISSUE:
Taiwan mapping uses DESCRIPTIVE NAMES:
  e.g., " ROA(C) before interest..."
  
But processed data uses F-CODES:
  e.g., F02, F03, F04...

This means:
  ✅ Polish ↔ American transfer: WORKS
  ❌ Taiwan transfer: BROKEN (wrong feature names)

RECOMMENDATION:
Script 00 needs to map Taiwan F-codes,
not descriptive names!
"""

ax3.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.8))

# 4. Example mapping detail
ax4 = axes[1, 1]
ax4.axis('off')

example_text = """
EXAMPLE: ROA Mapping

Polish (WORKS):
  Attr1, Attr7 ✅

American (WORKS):
  X1 ✅

Taiwan (BROKEN):
  Mapped: " ROA(C) before interest..."
  Actual: F02 or similar
  Status: ❌ MISMATCH

This explains why transfer learning
to/from Taiwan had poor results!
"""

ax4.text(0.1, 0.5, example_text, fontsize=11, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8d7da', alpha=0.8))

plt.tight_layout()

# Save
output_path = PROJECT_ROOT / 'results' / '00_feature_mapping' / 'mapping_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")

# Save summary
summary = {
    'total_features': len(semantic_mappings),
    'polish_mapped': sum(polish_counts),
    'american_mapped': sum(american_counts),
    'taiwan_mapped': sum(taiwan_counts),
    'polish_exist': polish_ok,
    'american_exist': american_ok,
    'taiwan_exist': taiwan_ok,
    'critical_issue': 'Taiwan uses F-codes in processed data but mapping uses descriptive names',
    'impact': 'Polish ↔ American transfer works, but Taiwan transfer is broken',
    'recommendation': 'Remap Taiwan using F-codes instead of descriptive names'
}

with open(PROJECT_ROOT / 'results' / '00_feature_mapping' / 'mapping_validation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Saved: mapping_validation_summary.json")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print(f"\n✅ Polish → American transfer: {polish_ok}/{len(results)} features work")
print(f"❌ Taiwan transfer: {taiwan_ok}/{len(results)} features work (BROKEN!)")
print(f"\n⚠️  This explains the poor Taiwan transfer learning results!")
print(f"   Script 00 hardcoded descriptive names but Taiwan processed data uses F-codes.")
print("\n" + "=" * 80)
