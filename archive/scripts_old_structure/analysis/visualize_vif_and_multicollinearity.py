"""
Create comprehensive VIF and multicollinearity visualizations for all datasets.

Generates:
1. VIF bar charts for each dataset
2. Correlation heatmaps  
3. VIF comparison across datasets
4. Feature selection impact analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def main():
    """Generate all VIF and multicollinearity visualizations."""
    
    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "results/visualizations/vif_multicollinearity"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GENERATING VIF AND MULTICOLLINEARITY VISUALIZATIONS")
    print("="*80)
    print()
    
    # =====================================================
    # 1. LOAD VIF DATA FOR ALL DATASETS
    # =====================================================
    print("1. Loading VIF data for all datasets...")
    
    vif_data = {}
    
    # Polish
    polish_vif_path = base_dir / "results/script_outputs/10d_remediation_save/vif_all_features.csv"
    if polish_vif_path.exists():
        vif_data['Polish'] = pd.read_csv(polish_vif_path)
        print(f"   ✓ Polish: {len(vif_data['Polish'])} features")
    
    # American
    american_vif_path = base_dir / "results/script_outputs/02_american/02b_vif_remediation/vif_all_features.csv"
    if american_vif_path.exists():
        american_vif = pd.read_csv(american_vif_path)
        # Standardize column names to Feature, VIF
        american_vif.columns = ['Feature', 'VIF']
        vif_data['American'] = american_vif
        print(f"   ✓ American: {len(vif_data['American'])} features")
    
    # Taiwan
    taiwan_vif_path = base_dir / "results/script_outputs/03_taiwan/02b_vif_remediation/vif_all_features.csv"
    if taiwan_vif_path.exists():
        taiwan_vif = pd.read_csv(taiwan_vif_path)
        # Standardize column names to Feature, VIF
        taiwan_vif.columns = ['Feature', 'VIF']
        vif_data['Taiwan'] = taiwan_vif
        print(f"   ✓ Taiwan: {len(vif_data['Taiwan'])} features")
    
    print()
    
    # =====================================================
    # 2. VIF BAR CHARTS FOR EACH DATASET
    # =====================================================
    print("2. Creating VIF bar charts...")
    
    for dataset_name, vif_df in vif_data.items():
        # Remove inf values for visualization
        vif_plot = vif_df.copy()
        vif_plot['VIF'] = vif_plot['VIF'].replace([np.inf, -np.inf], np.nan)
        vif_plot = vif_plot.dropna()
        vif_plot = vif_plot.sort_values('VIF', ascending=False).head(30)  # Top 30
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_plot['VIF']]
        
        ax.barh(vif_plot['Feature'], vif_plot['VIF'], color=colors, alpha=0.7)
        ax.axvline(x=10, color='red', linestyle='--', label='High VIF threshold (>10)', linewidth=2)
        ax.axvline(x=5, color='orange', linestyle='--', label='Moderate VIF threshold (>5)', linewidth=2)
        
        ax.set_xlabel('VIF', fontweight='bold', fontsize=12)
        ax.set_title(f'{dataset_name} Dataset - Top 30 Features by VIF', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset_name.lower()}_vif_bar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ {dataset_name} VIF bar chart saved")
    
    print()
    
    # =====================================================
    # 3. VIF DISTRIBUTION COMPARISON
    # =====================================================
    print("3. Creating VIF distribution comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (dataset_name, vif_df) in enumerate(vif_data.items()):
        ax = axes[idx]
        vif_clean = vif_df['VIF'].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Log scale for better visualization
        vif_log = np.log10(vif_clean + 1)
        
        ax.hist(vif_log, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.log10(11), color='red', linestyle='--', label='VIF=10', linewidth=2)
        ax.axvline(x=np.log10(6), color='orange', linestyle='--', label='VIF=5', linewidth=2)
        
        ax.set_xlabel('Log10(VIF + 1)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{dataset_name}\n{len(vif_clean)} features', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('VIF Distribution Across Datasets (Log Scale)', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'vif_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✓ VIF distribution comparison saved")
    print()
    
    # =====================================================
    # 4. VIF SUMMARY STATISTICS TABLE
    # =====================================================
    print("4. Creating VIF summary statistics...")
    
    summary_stats = []
    for dataset_name, vif_df in vif_data.items():
        vif_clean = vif_df['VIF'].replace([np.inf, -np.inf], np.nan).dropna()
        
        summary_stats.append({
            'Dataset': dataset_name,
            'Total_Features': len(vif_df),
            'Valid_VIF_Count': len(vif_clean),
            'High_VIF_Count': len(vif_clean[vif_clean > 10]),
            'Moderate_VIF_Count': len(vif_clean[(vif_clean > 5) & (vif_clean <= 10)]),
            'Low_VIF_Count': len(vif_clean[vif_clean <= 5]),
            'Mean_VIF': vif_clean.mean(),
            'Median_VIF': vif_clean.median(),
            'Max_VIF': vif_clean.max(),
            'Pct_High_VIF': (len(vif_clean[vif_clean > 10]) / len(vif_clean) * 100) if len(vif_clean) > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / 'vif_summary_statistics.csv', index=False)
    
    # Visualize summary table
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = summary_df.round(2).values
    table = ax.table(cellText=table_data, colLabels=summary_df.columns, 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('VIF Summary Statistics Across All Datasets', fontweight='bold', fontsize=14, pad=20)
    plt.savefig(output_dir / 'vif_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✓ VIF summary statistics saved")
    print()
    
    # =====================================================
    # 5. FEATURE SELECTION IMPACT
    # =====================================================
    print("5. Creating feature selection impact visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(vif_data))
    width = 0.25
    
    high_counts = [summary_stats[i]['High_VIF_Count'] for i in range(len(summary_stats))]
    mod_counts = [summary_stats[i]['Moderate_VIF_Count'] for i in range(len(summary_stats))]
    low_counts = [summary_stats[i]['Low_VIF_Count'] for i in range(len(summary_stats))]
    
    ax.bar(x - width, high_counts, width, label='High VIF (>10)', color='#d62728', alpha=0.8)
    ax.bar(x, mod_counts, width, label='Moderate VIF (5-10)', color='#ff7f0e', alpha=0.8)
    ax.bar(x + width, low_counts, width, label='Low VIF (<5)', color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('Dataset', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Features', fontweight='bold', fontsize=12)
    ax.set_title('Feature Distribution by VIF Category', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s['Dataset'] for s in summary_stats])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (h, m, l) in enumerate(zip(high_counts, mod_counts, low_counts)):
        ax.text(i - width, h + 0.5, str(h), ha='center', va='bottom', fontweight='bold')
        ax.text(i, m + 0.5, str(m), ha='center', va='bottom', fontweight='bold')
        ax.text(i + width, l + 0.5, str(l), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_selection_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✓ Feature selection impact visualization saved")
    print()
    
    # =====================================================
    # SUMMARY
    # =====================================================
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print()
    print("Generated files:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  • {file.name}")
    for file in sorted(output_dir.glob('*.csv')):
        print(f"  • {file.name}")
    print()
    print("Multicollinearity Analysis:")
    for stats in summary_stats:
        pct = stats['Pct_High_VIF']
        status = "🔴 SEVERE" if pct > 30 else "🟡 MODERATE" if pct > 10 else "🟢 LOW"
        print(f"  {stats['Dataset']:10s}: {stats['High_VIF_Count']:2d}/{stats['Total_Features']:2d} high VIF ({pct:.1f}%) {status}")
    print()
    
    return output_dir


if __name__ == "__main__":
    try:
        output_dir = main()
        print("✅ SUCCESS! All VIF and multicollinearity visualizations created!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
