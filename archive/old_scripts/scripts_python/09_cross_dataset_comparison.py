#!/usr/bin/env python3
"""
Cross-Dataset Comparison
Compare Polish, American, and Taiwan bankruptcy datasets
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import json

# Setup
project_root = Path(__file__).parent.parent
output_dir = project_root / 'results' / 'script_outputs' / '09_cross_dataset_comparison'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("CROSS-DATASET COMPARISON")
print("="*70)

# ========================================================================
# Part 1: Load Results from All Datasets
# ========================================================================
print("\n[1/5] Loading dataset results...")

# Polish results
polish_baseline = pd.read_csv(
    project_root / 'results' / 'script_outputs' / '04_baseline_models' / 'baseline_results.csv'
)
polish_advanced = pd.read_csv(
    project_root / 'results' / 'script_outputs' / '05_advanced_models' / 'advanced_results.csv'
)
polish_results = pd.concat([polish_baseline, polish_advanced], ignore_index=True)
polish_results['dataset'] = 'Polish'

# American results
american_results = pd.read_csv(
    project_root / 'results' / 'script_outputs' / 'american' / 'baseline_results.csv'
)
american_results['dataset'] = 'American'

# Taiwan results
taiwan_results = pd.read_csv(
    project_root / 'results' / 'script_outputs' / 'taiwan' / 'baseline_results.csv'
)
taiwan_results['dataset'] = 'Taiwan'

# Combine all results
all_results = pd.concat([polish_results, american_results, taiwan_results], ignore_index=True)
all_results.to_csv(output_dir / 'all_datasets_results.csv', index=False)

print(f"✓ Loaded results from 3 datasets")
print(f"  Total model runs: {len(all_results)}")

# ========================================================================
# Part 2: Dataset Characteristics Comparison
# ========================================================================
print("\n[2/5] Comparing dataset characteristics...")

# Load dataset summaries
with open(project_root / 'results' / 'script_outputs' / '01_data_understanding' / 'summary.json') as f:
    polish_summary = json.load(f)

with open(project_root / 'results' / 'script_outputs' / 'american' / 'cleaning_summary.json') as f:
    american_summary = json.load(f)

with open(project_root / 'results' / 'script_outputs' / 'taiwan' / 'cleaning_summary.json') as f:
    taiwan_summary = json.load(f)

# Calculate Polish overall stats from horizons
polish_total_samples = sum([h['Total_Samples'] for h in polish_summary['horizon_stats']])
polish_total_bankruptcies = sum([h['Bankruptcies'] for h in polish_summary['horizon_stats']])
polish_bankruptcy_rate = polish_total_bankruptcies / polish_total_samples

# Create comparison table
dataset_comparison = pd.DataFrame([
    {
        'Dataset': 'Polish',
        'Source': 'UCI ML Repository',
        'Companies': polish_total_samples,
        'Features': 64,
        'Feature_Type': 'Financial Ratios',
        'Bankruptcy_Rate_%': polish_bankruptcy_rate * 100,
        'Bankruptcies': polish_total_bankruptcies,
        'Horizons': 5,
        'Time_Period': '2000-2013'
    },
    {
        'Dataset': 'American',
        'Source': 'Kaggle (NYSE/NASDAQ)',
        'Companies': american_summary['companies'],
        'Features': american_summary['features'],
        'Feature_Type': 'Absolute Values',
        'Bankruptcy_Rate_%': american_summary['bankruptcy_rate'] * 100,
        'Bankruptcies': american_summary['bankruptcies'],
        'Horizons': 1,
        'Time_Period': '1999-2018'
    },
    {
        'Dataset': 'Taiwan',
        'Source': 'TEJ Database',
        'Companies': taiwan_summary['total_samples'],
        'Features': taiwan_summary['features'],
        'Feature_Type': 'Financial Ratios',
        'Bankruptcy_Rate_%': taiwan_summary['bankruptcy_rate'] * 100,
        'Bankruptcies': taiwan_summary['bankruptcies'],
        'Horizons': 1,
        'Time_Period': '1999-2009'
    }
])

dataset_comparison.to_csv(output_dir / 'dataset_characteristics.csv', index=False)

print("✓ Dataset Characteristics:")
for _, row in dataset_comparison.iterrows():
    print(f"\n  {row['Dataset']}:")
    print(f"    Companies: {row['Companies']:,}")
    print(f"    Features: {row['Features']} ({row['Feature_Type']})")
    print(f"    Bankruptcy Rate: {row['Bankruptcy_Rate_%']:.2f}%")

# ========================================================================
# Part 3: Model Performance Comparison
# ========================================================================
print("\n[3/5] Comparing model performance...")

# Get best model per dataset
best_models = all_results.loc[all_results.groupby('dataset')['roc_auc'].idxmax()]
best_models_df = best_models[['dataset', 'model_name', 'roc_auc', 'pr_auc']].copy()
best_models_df.to_csv(output_dir / 'best_models.csv', index=False)

print("\n✓ Best Model Performance:")
for _, row in best_models_df.iterrows():
    print(f"  {row['dataset']}: {row['model_name']} - ROC-AUC: {row['roc_auc']:.4f}")

# Performance by model type across datasets
model_types = ['Logistic Regression', 'Random Forest', 'CatBoost']
performance_matrix = []

for model_type in model_types:
    row_data = {'Model': model_type}
    for dataset in ['Polish', 'American', 'Taiwan']:
        subset = all_results[(all_results['dataset'] == dataset) & 
                            (all_results['model_name'] == model_type)]
        if len(subset) > 0:
            row_data[f'{dataset}_AUC'] = subset['roc_auc'].values[0]
        else:
            row_data[f'{dataset}_AUC'] = np.nan
    performance_matrix.append(row_data)

performance_df = pd.DataFrame(performance_matrix)
performance_df.to_csv(output_dir / 'model_performance_matrix.csv', index=False)

# ========================================================================
# Part 4: Statistical Analysis
# ========================================================================
print("\n[4/5] Statistical analysis...")

# Performance statistics
stats = {
    'best_overall_dataset': best_models_df.loc[best_models_df['roc_auc'].idxmax(), 'dataset'],
    'best_overall_model': best_models_df.loc[best_models_df['roc_auc'].idxmax(), 'model_name'],
    'best_overall_auc': float(best_models_df['roc_auc'].max()),
    'avg_auc_polish': float(all_results[all_results['dataset'] == 'Polish']['roc_auc'].mean()),
    'avg_auc_american': float(all_results[all_results['dataset'] == 'American']['roc_auc'].mean()),
    'avg_auc_taiwan': float(all_results[all_results['dataset'] == 'Taiwan']['roc_auc'].mean()),
    'performance_range': float(best_models_df['roc_auc'].max() - best_models_df['roc_auc'].min()),
    'total_models_compared': len(all_results)
}

with open(output_dir / 'comparison_summary.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n✓ Statistical Summary:")
print(f"  Best: {stats['best_overall_dataset']} - {stats['best_overall_model']} ({stats['best_overall_auc']:.4f})")
print(f"  Average AUC by dataset:")
print(f"    Polish: {stats['avg_auc_polish']:.4f}")
print(f"    Taiwan: {stats['avg_auc_taiwan']:.4f}")
print(f"    American: {stats['avg_auc_american']:.4f}")

# ========================================================================
# Part 5: Visualizations
# ========================================================================
print("\n[5/5] Creating visualizations...")

# 1. Dataset characteristics comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Companies
axes[0, 0].bar(dataset_comparison['Dataset'], dataset_comparison['Companies'],
              color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Number of Companies', fontweight='bold')
axes[0, 0].set_title('Dataset Size Comparison', fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(dataset_comparison['Companies']):
    axes[0, 0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

# Features
axes[0, 1].bar(dataset_comparison['Dataset'], dataset_comparison['Features'],
              color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Number of Features', fontweight='bold')
axes[0, 1].set_title('Feature Count Comparison', fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(dataset_comparison['Features']):
    axes[0, 1].text(i, v + 2, str(v), ha='center', fontweight='bold')

# Bankruptcy Rate
axes[1, 0].bar(dataset_comparison['Dataset'], dataset_comparison['Bankruptcy_Rate_%'],
              color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
axes[1, 0].set_title('Class Imbalance Comparison', fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(dataset_comparison['Bankruptcy_Rate_%']):
    axes[1, 0].text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold')

# Feature Types
feature_types = dataset_comparison.groupby('Feature_Type').size()
axes[1, 1].pie(feature_types.values, labels=feature_types.index, autopct='%1.0f%%',
              colors=['#2ca02c', '#ff7f0e'], startangle=90)
axes[1, 1].set_title('Feature Type Distribution', fontweight='bold')

plt.suptitle('Dataset Characteristics Comparison', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(figures_dir / 'dataset_characteristics.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved dataset characteristics")

# 2. Model performance comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Best models comparison
datasets_ordered = best_models_df.sort_values('roc_auc', ascending=False)
colors_perf = ['gold', 'silver', '#cd7f32']  # Gold, Silver, Bronze

axes[0].barh(range(len(datasets_ordered)), datasets_ordered['roc_auc'],
            color=colors_perf, alpha=0.7, edgecolor='black')
axes[0].set_yticks(range(len(datasets_ordered)))
axes[0].set_yticklabels([f"{row['dataset']}\n({row['model_name']})" 
                          for _, row in datasets_ordered.iterrows()])
axes[0].set_xlabel('ROC-AUC Score', fontweight='bold')
axes[0].set_title('Best Model Performance by Dataset', fontsize=14, fontweight='bold')
axes[0].set_xlim([0.8, 1.0])
axes[0].grid(axis='x', alpha=0.3)

for i, (_, row) in enumerate(datasets_ordered.iterrows()):
    axes[0].text(row['roc_auc'] + 0.005, i, f"{row['roc_auc']:.4f}",
                va='center', fontweight='bold')

# Model consistency across datasets
model_avg_performance = []
for model in model_types:
    avg_auc = all_results[all_results['model_name'] == model]['roc_auc'].mean()
    model_avg_performance.append({'Model': model, 'Avg_AUC': avg_auc})

model_avg_df = pd.DataFrame(model_avg_performance).sort_values('Avg_AUC', ascending=False)

axes[1].bar(range(len(model_avg_df)), model_avg_df['Avg_AUC'],
           color=['darkgreen', 'steelblue', 'crimson'], alpha=0.7, edgecolor='black')
axes[1].set_xticks(range(len(model_avg_df)))
axes[1].set_xticklabels(model_avg_df['Model'], rotation=15, ha='right')
axes[1].set_ylabel('Average ROC-AUC', fontweight='bold')
axes[1].set_title('Model Consistency Across Datasets', fontsize=14, fontweight='bold')
axes[1].set_ylim([0.8, 1.0])
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(model_avg_df['Avg_AUC']):
    axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved model performance comparison")

# 3. Performance heatmap
plt.figure(figsize=(10, 6))

# Create heatmap data
heatmap_data = []
for model in model_types:
    row = []
    for dataset in ['Polish', 'American', 'Taiwan']:
        subset = all_results[(all_results['dataset'] == dataset) & 
                            (all_results['model_name'] == model)]
        if len(subset) > 0:
            row.append(subset['roc_auc'].values[0])
        else:
            row.append(np.nan)
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, 
                         index=model_types,
                         columns=['Polish', 'American', 'Taiwan'])

sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdYlGn', center=0.9,
           vmin=0.8, vmax=1.0, linewidths=2, linecolor='black',
           cbar_kws={'label': 'ROC-AUC Score'})
plt.title('Model Performance Heatmap Across Datasets', fontsize=14, fontweight='bold')
plt.ylabel('Model Type', fontweight='bold')
plt.xlabel('Dataset', fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved performance heatmap")

# 4. Comprehensive comparison table visualization
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Dataset', 'Companies', 'Features', 'Feature Type', 
                  'Bankruptcy %', 'Best Model', 'Best AUC'])

for _, ds_row in dataset_comparison.iterrows():
    best_for_ds = best_models_df[best_models_df['dataset'] == ds_row['Dataset']].iloc[0]
    table_data.append([
        ds_row['Dataset'],
        f"{ds_row['Companies']:,}",
        str(ds_row['Features']),
        ds_row['Feature_Type'],
        f"{ds_row['Bankruptcy_Rate_%']:.2f}%",
        best_for_ds['model_name'],
        f"{best_for_ds['roc_auc']:.4f}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.12, 0.12, 0.10, 0.18, 0.14, 0.18, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(7):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows by performance
colors_rows = ['#E7E6E6', '#C6E0B4', '#FFC000']  # Gray, Green, Orange
for i in range(1, 4):
    color = colors_rows[i-1]
    for j in range(7):
        table[(i, j)].set_facecolor(color)

plt.title('Comprehensive Dataset & Performance Comparison', 
         fontsize=16, fontweight='bold', pad=20)
plt.savefig(figures_dir / 'comparison_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved comparison table")

print("\n" + "="*70)
print("✓ CROSS-DATASET COMPARISON COMPLETE")
print(f"  Datasets compared: 3")
print(f"  Total models: {stats['total_models_compared']}")
print(f"  Best overall: {stats['best_overall_dataset']} ({stats['best_overall_auc']:.4f})")
print(f"  Performance range: {stats['performance_range']:.4f}")
print("="*70)
