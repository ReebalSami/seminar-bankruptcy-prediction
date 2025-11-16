#!/usr/bin/env python3
"""
Data Understanding Script
Converted from 01_data_understanding.ipynb
Saves results to files instead of displaying inline
"""

import sys
from pathlib import Path

# Add project root to path (scripts/01_polish -> root is 2 levels up)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.bankruptcy_prediction.data import DataLoader, MetadataParser

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', 100)

# Create output directories
output_dir = PROJECT_ROOT / 'results' / 'script_outputs' / '01_data_understanding'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir = output_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

print("="*70)
print("SCRIPT 01: Data Understanding")
print("="*70)

# Track progress
results_summary = {}

# Step 1: Load data
print("\n[1/6] Loading data and metadata...")
try:
    loader = DataLoader()
    metadata = MetadataParser.from_default()
    
    df_full = loader.load_poland(horizon=None, dataset_type='full')
    df = loader.load_poland(horizon=1, dataset_type='full')
    
    results_summary['data_loaded'] = True
    results_summary['full_shape'] = df_full.shape
    results_summary['h1_shape'] = df.shape
    
    print(f"✓ Full dataset: {df_full.shape}")
    print(f"✓ Horizon 1: {df.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    results_summary['data_loaded'] = False
    sys.exit(1)

# Step 2: Dataset info
print("\n[2/6] Getting dataset info...")
try:
    info = loader.get_info(df)
    
    info_df = pd.DataFrame([info])
    info_df.to_csv(output_dir / 'dataset_info.csv', index=False)
    
    results_summary['dataset_info'] = info
    
    print(f"✓ Samples: {info['n_samples']:,}")
    print(f"✓ Features: {info['n_features']}")
    print(f"✓ Bankruptcy rate: {info['bankruptcy_rate']:.2%}")
    print(f"  Saved: dataset_info.csv")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Step 3: Class distribution by horizon
print("\n[3/6] Analyzing class distribution by horizon...")
try:
    horizon_stats = df_full.groupby('horizon')['y'].agg(['count', 'sum', 'mean']).reset_index()
    horizon_stats.columns = ['Horizon', 'Total_Samples', 'Bankruptcies', 'Bankruptcy_Rate']
    horizon_stats['Healthy'] = horizon_stats['Total_Samples'] - horizon_stats['Bankruptcies']
    
    horizon_stats.to_csv(output_dir / 'horizon_statistics.csv', index=False)
    
    results_summary['horizon_stats'] = horizon_stats.to_dict('records')
    
    print(f"✓ Analyzed {len(horizon_stats)} horizons")
    print(horizon_stats)
    print(f"  Saved: horizon_statistics.csv")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Step 4: Plot class distribution
print("\n[4/6] Creating class distribution plot...")
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stacked bar
    horizons = horizon_stats['Horizon']
    healthy_pct = (1 - horizon_stats['Bankruptcy_Rate']) * 100
    bankrupt_pct = horizon_stats['Bankruptcy_Rate'] * 100
    
    ax1.bar(horizons, healthy_pct, label='Healthy', color='#2ecc71', alpha=0.8)
    ax1.bar(horizons, bankrupt_pct, bottom=healthy_pct, label='Bankrupt', color='#e74c3c', alpha=0.8)
    
    for h, b_pct, h_pct in zip(horizons, bankrupt_pct, healthy_pct):
        ax1.text(h, h_pct + b_pct/2, f'{b_pct:.1f}%', ha='center', va='center', fontweight='bold')
    
    ax1.set_xlabel('Horizon (years ahead)', fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontweight='bold')
    ax1.set_title('Class Distribution by Horizon', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Sample size
    ax2.bar(horizons, horizon_stats['Total_Samples'], color='#3498db', alpha=0.8)
    ax2.set_xlabel('Horizon (years ahead)', fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontweight='bold')
    ax2.set_title('Sample Size by Horizon', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'class_distribution_by_horizon.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot created")
    print(f"  Saved: figures/class_distribution_by_horizon.png")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Feature categories
print("\n[5/6] Analyzing feature categories...")
try:
    categories = metadata.get_all_categories()
    category_counts = {cat: len(metadata.get_features_by_category(cat)) for cat in categories}
    
    category_df = pd.DataFrame([category_counts]).T
    category_df.columns = ['Count']
    category_df.index.name = 'Category'
    category_df = category_df.sort_values('Count', ascending=False)
    category_df.to_csv(output_dir / 'feature_categories.csv')
    
    results_summary['categories'] = category_counts
    
    print(f"✓ Analyzed {len(categories)} categories")
    print(category_df)
    print(f"  Saved: feature_categories.csv")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Step 6: Plot categories
print("\n[6/6] Creating categories plot...")
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'Profitability': '#3498db', 'Liquidity': '#2ecc71', 
              'Leverage': '#e74c3c', 'Activity': '#f39c12', 
              'Size': '#9b59b6', 'Other': '#95a5a6'}
    
    bar_colors = [colors.get(cat, '#95a5a6') for cat in category_df.index]
    
    ax.barh(range(len(category_df)), category_df['Count'], color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(category_df)))
    ax.set_yticklabels(category_df.index)
    ax.set_xlabel('Number of Features', fontweight='bold')
    ax.set_title('Features by Category', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, count in enumerate(category_df['Count']):
        ax.text(count + 0.3, i, str(count), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'features_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot created")
    print(f"  Saved: figures/features_by_category.png")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Save summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

import json
with open(output_dir / 'summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f"\n✓ Script completed successfully!")
print(f"✓ All outputs saved to: {output_dir}")
print(f"✓ Figures saved to: {figures_dir}")
print(f"✓ Summary saved to: summary.json")
print("\nFiles created:")
print(f"  - dataset_info.csv")
print(f"  - horizon_statistics.csv")
print(f"  - feature_categories.csv")
print(f"  - figures/class_distribution_by_horizon.png")
print(f"  - figures/features_by_category.png")
print(f"  - summary.json")
print("="*70)
