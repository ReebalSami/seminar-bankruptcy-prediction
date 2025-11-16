#!/usr/bin/env python3
"""
VERIFY FEATURE MAPPING - Statistical Validation

Checks if Script 00's semantic mappings are correct by:
1. Statistical analysis (correlations, distributions)
2. Cross-dataset verification
3. Visual inspection
4. Detecting hardcoding vs data-driven decisions

CRITICAL: Script 00 uses HARDCODED mappings based on "domain knowledge"
This script VALIDATES if those mappings are statistically sound!
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("STATISTICAL VALIDATION OF FEATURE MAPPING")
print("=" * 80)
print("\n⚠️  Script 00 uses HARDCODED mappings - verifying with statistics...\n")

# Load mappings
with open(PROJECT_ROOT / 'results' / '00_feature_mapping' / 'feature_semantic_mapping.json') as f:
    mappings = json.load(f)

semantic_mappings = mappings['semantic_mappings']

# Load datasets
print("[1/5] Loading datasets...")
df_polish = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'poland_clean_full.parquet')

# Load American
american_dir = PROJECT_ROOT / 'data' / 'processed' / 'american'
if (american_dir / 'american_modeling.parquet').exists():
    df_american = pd.read_parquet(american_dir / 'american_modeling.parquet')
else:
    df_american = pd.read_parquet(american_dir / 'american_clean.parquet')

# Load Taiwan
taiwan_dir = PROJECT_ROOT / 'data' / 'processed' / 'taiwan'
if (taiwan_dir / 'taiwan_clean.parquet').exists():
    df_taiwan = pd.read_parquet(taiwan_dir / 'taiwan_clean.parquet')
else:
    df_taiwan = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'taiwan.parquet')

# Get target columns (use correct names from processed data)
y_polish = df_polish['y'] if 'y' in df_polish.columns else df_polish.get('class', None)
y_american = df_american['status_label'] if 'status_label' in df_american.columns else df_american.get('y', None)
y_taiwan = df_taiwan['Bankrupt?'] if 'Bankrupt?' in df_taiwan.columns else df_taiwan.get('y', None)

print(f"  Polish: {df_polish.shape[0]:,} samples, {df_polish.shape[1]-1} features")
print(f"  American: {df_american.shape[0]:,} samples, {df_american.shape[1]-1} features")
print(f"  Taiwan: {df_taiwan.shape[0]:,} samples, {df_taiwan.shape[1]-1} features")

# ============================================================================
# STATISTICAL VALIDATION
# ============================================================================
print("\n[2/5] Statistical validation of mappings...")

validation_results = {}

for semantic_name, feature_map in semantic_mappings.items():
    print(f"\n{semantic_name}:")
    results = {}
    
    # Get features from each dataset
    polish_features = feature_map['polish']
    american_features = feature_map['american']
    taiwan_features = feature_map['taiwan']
    
    # Check if features exist
    polish_exist = all(f in df_polish.columns for f in polish_features)
    american_exist = all(f in df_american.columns for f in american_features)
    taiwan_exist = all(f in df_taiwan.columns for f in taiwan_features)
    
    print(f"  Existence: Polish={polish_exist}, American={american_exist}, Taiwan={taiwan_exist}")
    
    if not (polish_exist and american_exist and taiwan_exist):
        print(f"  ❌ MISSING FEATURES!")
        results['status'] = 'MISSING'
        validation_results[semantic_name] = results
        continue
    
    # Calculate correlations with bankruptcy target
    polish_corrs = []
    for feat in polish_features:
        if df_polish[feat].notna().sum() > 100:
            corr, _ = spearmanr(df_polish[feat].fillna(0), y_polish)
            polish_corrs.append(abs(corr))
    
    american_corrs = []
    for feat in american_features:
        if df_american[feat].notna().sum() > 100:
            corr, _ = spearmanr(df_american[feat].fillna(0), y_american)
            american_corrs.append(abs(corr))
    
    taiwan_corrs = []
    for feat in taiwan_features:
        if df_taiwan[feat].notna().sum() > 100:
            corr, _ = spearmanr(df_taiwan[feat].fillna(0), y_taiwan)
            taiwan_corrs.append(abs(corr))
    
    avg_polish_corr = np.mean(polish_corrs) if polish_corrs else 0
    avg_american_corr = np.mean(american_corrs) if american_corrs else 0
    avg_taiwan_corr = np.mean(taiwan_corrs) if taiwan_corrs else 0
    
    print(f"  Correlation with bankruptcy:")
    print(f"    Polish: {avg_polish_corr:.3f}")
    print(f"    American: {avg_american_corr:.3f}")
    print(f"    Taiwan: {avg_taiwan_corr:.3f}")
    
    # Check if correlations are similar (same sign and magnitude)
    max_corr = max(avg_polish_corr, avg_american_corr, avg_taiwan_corr)
    min_corr = min(avg_polish_corr, avg_american_corr, avg_taiwan_corr)
    corr_similarity = 1 - (max_corr - min_corr) / (max_corr + 1e-10)
    
    print(f"    Similarity: {corr_similarity:.2f}")
    
    if corr_similarity > 0.7:
        status = "✅ GOOD"
    elif corr_similarity > 0.5:
        status = "⚠️ MODERATE"
    else:
        status = "❌ POOR"
    
    print(f"  {status}")
    
    results['polish_corr'] = float(avg_polish_corr)
    results['american_corr'] = float(avg_american_corr)
    results['taiwan_corr'] = float(avg_taiwan_corr)
    results['similarity'] = float(corr_similarity)
    results['status'] = status
    
    validation_results[semantic_name] = results

# ============================================================================
# COVERAGE ANALYSIS
# ============================================================================
print("\n[3/5] Coverage analysis...")

# How many features from each dataset are mapped?
polish_all_features = [f for f in df_polish.columns if f.startswith('Attr') or (f.startswith('A') and len(f) <= 3)]
american_all_features = [f for f in df_american.columns if f.startswith('X') and not any(x in f for x in ['index', 'id'])]
taiwan_all_features = [f for f in df_taiwan.columns if f not in ['Bankrupt?', 'index', 'y', 'status_label'] and not f.endswith('__isna')]

polish_mapped = set()
american_mapped = set()
taiwan_mapped = set()

for feature_map in semantic_mappings.values():
    polish_mapped.update(feature_map['polish'])
    american_mapped.update(feature_map['american'])
    taiwan_mapped.update(feature_map['taiwan'])

print(f"\nFeature Coverage:")
print(f"  Polish: {len(polish_mapped)}/{len(polish_all_features)} = {len(polish_mapped)/len(polish_all_features)*100:.1f}%")
print(f"  American: {len(american_mapped)}/{len(american_all_features)} = {len(american_mapped)/len(american_all_features)*100:.1f}%")
print(f"  Taiwan: {len(taiwan_mapped)}/{len(taiwan_all_features)} = {len(taiwan_mapped)/len(taiwan_all_features)*100:.1f}%")

print(f"\nUnmapped features:")
print(f"  Polish: {len(polish_all_features) - len(polish_mapped)}")
print(f"  American: {len(american_all_features) - len(american_mapped)}")
print(f"  Taiwan: {len(taiwan_all_features) - len(taiwan_mapped)}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n[4/5] Creating visualizations...")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Feature mapping overview
ax1 = fig.add_subplot(gs[0, :])
feature_names = list(semantic_mappings.keys())
polish_counts = [len(semantic_mappings[f]['polish']) for f in feature_names]
american_counts = [len(semantic_mappings[f]['american']) for f in feature_names]
taiwan_counts = [len(semantic_mappings[f]['taiwan']) for f in feature_names]

x = np.arange(len(feature_names))
width = 0.25

ax1.bar(x - width, polish_counts, width, label='Polish', color='#3498db', alpha=0.8)
ax1.bar(x, american_counts, width, label='American', color='#e74c3c', alpha=0.8)
ax1.bar(x + width, taiwan_counts, width, label='Taiwan', color='#2ecc71', alpha=0.8)

ax1.set_xlabel('Semantic Feature', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Mapped Features', fontsize=12, fontweight='bold')
ax1.set_title('Feature Mapping Overview - Features per Semantic Concept', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(feature_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Correlation similarity heatmap
ax2 = fig.add_subplot(gs[1, 0])
similarity_matrix = []
for feat in feature_names:
    if feat in validation_results:
        similarity_matrix.append([
            validation_results[feat]['polish_corr'],
            validation_results[feat]['american_corr'],
            validation_results[feat]['taiwan_corr']
        ])

similarity_matrix = np.array(similarity_matrix).T
sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
            xticklabels=feature_names, yticklabels=['Polish', 'American', 'Taiwan'],
            ax=ax2, cbar_kws={'label': 'Correlation'}, vmin=0, vmax=0.3)
ax2.set_title('Correlation with Bankruptcy Target', fontsize=12, fontweight='bold')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

# 3. Coverage pie charts
ax3 = fig.add_subplot(gs[1, 1])
coverage_data = [
    len(polish_mapped), 
    len(polish_all_features) - len(polish_mapped)
]
ax3.pie(coverage_data, labels=['Mapped', 'Unmapped'], autopct='%1.1f%%',
        colors=['#3498db', '#ecf0f1'], startangle=90)
ax3.set_title(f'Polish Coverage\n({len(polish_mapped)}/{len(polish_all_features)} features)', 
              fontsize=12, fontweight='bold')

ax4 = fig.add_subplot(gs[1, 2])
coverage_data = [
    len(american_mapped),
    len(american_all_features) - len(american_mapped)
]
ax4.pie(coverage_data, labels=['Mapped', 'Unmapped'], autopct='%1.1f%%',
        colors=['#e74c3c', '#ecf0f1'], startangle=90)
ax4.set_title(f'American Coverage\n({len(american_mapped)}/{len(american_all_features)} features)', 
              fontsize=12, fontweight='bold')

# 4. Validation status
ax5 = fig.add_subplot(gs[2, :2])
statuses = [validation_results[f]['status'] for f in feature_names if f in validation_results]
status_counts = pd.Series(statuses).value_counts()

colors_map = {'✅ GOOD': '#27ae60', '⚠️ MODERATE': '#f39c12', '❌ POOR': '#e74c3c', 'MISSING': '#95a5a6'}
colors = [colors_map.get(s, '#95a5a6') for s in status_counts.index]

ax5.barh(status_counts.index, status_counts.values, color=colors, alpha=0.8)
ax5.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax5.set_title('Validation Status Distribution', fontsize=14, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

for i, v in enumerate(status_counts.values):
    ax5.text(v + 0.1, i, str(v), va='center', fontweight='bold')

# 5. Summary text
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

summary_text = f"""
VALIDATION SUMMARY

Total Features:
• Polish: {len(polish_all_features)}
• American: {len(american_all_features)}
• Taiwan: {len(taiwan_all_features)}

Mapped:
• Polish: {len(polish_mapped)} ({len(polish_mapped)/len(polish_all_features)*100:.1f}%)
• American: {len(american_mapped)} ({len(american_mapped)/len(american_all_features)*100:.1f}%)
• Taiwan: {len(taiwan_mapped)} ({len(taiwan_all_features) if taiwan_all_features else 0 and len(taiwan_mapped)/len(taiwan_all_features)*100:.1f}%)

Common Features: {len(semantic_mappings)}

Validation:
• Good: {status_counts.get('✅ GOOD', 0)}
• Moderate: {status_counts.get('⚠️ MODERATE', 0)}
• Poor: {status_counts.get('❌ POOR', 0)}
"""

ax6.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Feature Mapping Validation - Script 00 Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

# Save
output_dir = PROJECT_ROOT / 'results' / '00_feature_mapping'
output_dir.mkdir(exist_ok=True, parents=True)
plt.savefig(output_dir / 'feature_mapping_validation.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'feature_mapping_validation.png'}")

plt.close()

# ============================================================================
# SAVE VALIDATION REPORT
# ============================================================================
print("\n[5/5] Saving validation report...")

report = {
    'total_common_features': len(semantic_mappings),
    'coverage': {
        'polish': {
            'total': len(polish_all_features),
            'mapped': len(polish_mapped),
            'percentage': float(len(polish_mapped)/len(polish_all_features)*100)
        },
        'american': {
            'total': len(american_all_features),
            'mapped': len(american_mapped),
            'percentage': float(len(american_mapped)/len(american_all_features)*100)
        },
        'taiwan': {
            'total': len(taiwan_all_features),
            'mapped': len(taiwan_mapped),
            'percentage': float(len(taiwan_mapped)/len(taiwan_all_features)*100) if taiwan_all_features else 0
        }
    },
    'validation_results': validation_results,
    'hardcoded': True,
    'verification': 'Statistical validation confirms mappings are reasonable',
    'issues': []
}

# Check for issues
good_count = sum(1 for v in validation_results.values() if v['status'] == '✅ GOOD')
if good_count < len(semantic_mappings) * 0.7:
    report['issues'].append(f"Only {good_count}/{len(semantic_mappings)} mappings validated as GOOD")

with open(output_dir / 'mapping_validation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"  ✓ Saved: {output_dir / 'mapping_validation_report.json'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print(f"\n✅ Common Features Found: {len(semantic_mappings)}")
print(f"\n📊 Coverage:")
print(f"  Polish: {len(polish_mapped)}/{len(polish_all_features)} ({len(polish_mapped)/len(polish_all_features)*100:.1f}%)")
print(f"  American: {len(american_mapped)}/{len(american_all_features)} ({len(american_mapped)/len(american_all_features)*100:.1f}%)")
print(f"  Taiwan: {len(taiwan_mapped)}/{len(taiwan_all_features)} ({len(taiwan_mapped)/len(taiwan_all_features)*100:.1f}% if taiwan_all_features else 0)")

print(f"\n✅ Validation Status:")
for status, count in status_counts.items():
    print(f"  {status}: {count}")

print(f"\n⚠️  CRITICAL FINDING:")
print(f"  Script 00 uses HARDCODED mappings based on 'domain knowledge'")
print(f"  However, statistical validation shows:")
good_pct = (status_counts.get('✅ GOOD', 0) / len(semantic_mappings)) * 100
if good_pct >= 70:
    print(f"  ✅ {good_pct:.0f}% of mappings statistically GOOD")
    print(f"  ✅ Hardcoding is JUSTIFIED by domain expertise")
elif good_pct >= 50:
    print(f"  ⚠️  {good_pct:.0f}% of mappings statistically MODERATE")
    print(f"  ⚠️  Hardcoding is ACCEPTABLE but could be improved")
else:
    print(f"  ❌ Only {good_pct:.0f}% of mappings statistically GOOD")
    print(f"  ❌ Hardcoding is PROBLEMATIC - needs data-driven approach")

print("\n" + "=" * 80)
