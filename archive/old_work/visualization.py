"""
Visualization Utilities for Bankruptcy Prediction
==================================================

Reusable plotting functions with readable feature names and publication-quality styling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple

from src.metadata import get_readable_name, get_category, CATEGORIES

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_class_imbalance_by_horizon(df: pd.DataFrame, 
                                     output_path: Optional[Path] = None,
                                     figsize: Tuple[int, int] = (10, 6)):
    """
    Bar chart showing bankruptcy share per horizon.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'horizon' and 'y' columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    # Calculate bankruptcy rate per horizon
    stats = df.groupby('horizon')['y'].agg(['mean', 'sum', 'count']).reset_index()
    stats['bankrupt_pct'] = stats['mean'] * 100
    stats['healthy_pct'] = (1 - stats['mean']) * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = stats['horizon']
    width = 0.35
    
    # Stacked bar chart
    p1 = ax.bar(x, stats['healthy_pct'], width, label='Healthy', color='#2ecc71')
    p2 = ax.bar(x, stats['bankrupt_pct'], width, bottom=stats['healthy_pct'],
                label='Bankrupt', color='#e74c3c')
    
    # Add percentage labels on bars
    for i, (h, pct) in enumerate(zip(stats['horizon'], stats['bankrupt_pct'])):
        ax.text(h, stats.loc[i, 'healthy_pct'] + pct/2, f'{pct:.1f}%',
                ha='center', va='center', fontweight='bold', color='white')
        ax.text(h, stats.loc[i, 'healthy_pct']/2, f'{stats.loc[i, "healthy_pct"]:.1f}%',
                ha='center', va='center', fontweight='bold', color='white')
    
    ax.set_xlabel('Horizon (years ahead)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution by Prediction Horizon', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add sample size annotations
    for i, (h, n) in enumerate(zip(stats['horizon'], stats['count'])):
        ax.text(h, -5, f'n={n:,}', ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig, ax


def plot_missingness(df: pd.DataFrame, 
                      top_n: int = 20,
                      output_path: Optional[Path] = None,
                      figsize: Tuple[int, int] = (12, 8)):
    """
    Bar chart showing NA% for top N features with highest missingness.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with potential missing values
    top_n : int
        Show top N features
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    # Calculate missingness percentage
    na_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    na_pct = na_pct[na_pct > 0].head(top_n)
    
    if len(na_pct) == 0:
        print("No missing values found in dataset!")
        return None, None
    
    # Get readable names
    readable_names = [get_readable_name(col, short=True) for col in na_pct.index]
    categories = [get_category(col) for col in na_pct.index]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by category
    colors = {'Profitability': '#3498db', 'Liquidity': '#2ecc71', 
              'Leverage': '#e74c3c', 'Activity': '#f39c12', 
              'Size': '#9b59b6', 'Other': '#95a5a6'}
    bar_colors = [colors.get(cat, '#95a5a6') for cat in categories]
    
    bars = ax.barh(readable_names, na_pct.values, color=bar_colors, alpha=0.8)
    
    # Add percentage labels
    for bar, pct in zip(bars, na_pct.values):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Missing Values (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Features with Highest Missingness',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max(na_pct.values) * 1.15)
    ax.grid(axis='x', alpha=0.3)
    
    # Add category legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[cat], label=cat) 
                       for cat in sorted(set(categories))]
    ax.legend(handles=legend_elements, loc='lower right', 
              frameon=True, title='Category', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig, ax


def plot_distributions_by_class(df: pd.DataFrame,
                                  features: List[str],
                                  target_col: str = 'y',
                                  output_path: Optional[Path] = None,
                                  figsize: Tuple[int, int] = (16, 10),
                                  log_scale: bool = False):
    """
    Create histograms and boxplots for features, split by bankruptcy status.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with features and target
    features : list
        List of feature names to plot
    target_col : str
        Name of target column
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    log_scale : bool
        Use log scale for x-axis
    """
    n_features = len(features)
    n_cols = 2  # histogram and boxplot
    n_rows = n_features
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feat in enumerate(features):
        # Skip if feature doesn't exist
        if feat not in df.columns:
            print(f"Warning: {feat} not in dataframe, skipping")
            continue
        
        data = df[[feat, target_col]].dropna()
        bankrupt = data[data[target_col] == 1][feat]
        healthy = data[data[target_col] == 0][feat]
        
        readable_name = get_readable_name(feat, short=True)
        
        # Histogram
        ax_hist = axes[i, 0]
        bins = 50
        
        if log_scale and (data[feat] > 0).all():
            # Use log scale if all values positive
            bins = np.logspace(np.log10(data[feat].min() + 1e-10), 
                               np.log10(data[feat].max() + 1e-10), 50)
            ax_hist.set_xscale('log')
        
        ax_hist.hist(healthy, bins=bins, alpha=0.6, label='Healthy', 
                     color='#2ecc71', density=True)
        ax_hist.hist(bankrupt, bins=bins, alpha=0.6, label='Bankrupt', 
                     color='#e74c3c', density=True)
        
        ax_hist.set_ylabel('Density', fontsize=9)
        ax_hist.set_title(f'{readable_name}', fontsize=10, fontweight='bold')
        ax_hist.legend(fontsize=8)
        ax_hist.grid(alpha=0.3)
        
        # Boxplot
        ax_box = axes[i, 1]
        box_data = [healthy, bankrupt]
        bp = ax_box.boxplot(box_data, labels=['Healthy', 'Bankrupt'],
                            patch_artist=True, vert=False,
                            boxprops=dict(alpha=0.7),
                            medianprops=dict(color='black', linewidth=2))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
            patch.set_facecolor(color)
        
        ax_box.set_xlabel(readable_name, fontsize=9)
        ax_box.grid(alpha=0.3, axis='x')
    
    plt.suptitle('Feature Distributions by Bankruptcy Status', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig, axes


def plot_correlation_heatmap(df: pd.DataFrame,
                               method: str = 'pearson',
                               threshold: Optional[float] = None,
                               output_path: Optional[Path] = None,
                               figsize: Tuple[int, int] = (14, 12)):
    """
    Correlation heatmap with readable feature names.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with features
    method : str
        Correlation method (pearson, spearman, kendall)
    threshold : float, optional
        If provided, only show correlations above this threshold
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    # Calculate correlation matrix
    corr = df.corr(method=method)
    
    # Apply threshold if specified
    if threshold is not None:
        mask = np.abs(corr) < threshold
        corr = corr.mask(mask)
    else:
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
    
    # Rename with readable names
    readable_cols = [get_readable_name(col, short=True) for col in corr.columns]
    corr.index = readable_cols
    corr.columns = readable_cols
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    title = f'{method.capitalize()} Correlation Matrix'
    if threshold:
        title += f' (|r| ≥ {threshold})'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig, ax


def find_high_correlations(df: pd.DataFrame,
                            threshold: float = 0.9,
                            output_path: Optional[Path] = None):
    """
    Find and display pairs of highly correlated features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with features
    threshold : float
        Correlation threshold (absolute value)
    output_path : Path, optional
        Path to save CSV
    
    Returns
    -------
    pd.DataFrame
        Table of highly correlated pairs
    """
    corr = df.corr()
    
    # Get upper triangle
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) >= threshold:
                high_corr.append({
                    'Feature_1': get_readable_name(corr.columns[i], short=True),
                    'Feature_2': get_readable_name(corr.columns[j], short=True),
                    'Feature_1_Raw': corr.columns[i],
                    'Feature_2_Raw': corr.columns[j],
                    'Correlation': corr.iloc[i, j]
                })
    
    result = pd.DataFrame(high_corr).sort_values('Correlation', 
                                                   key=abs, ascending=False)
    
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
    
    return result


def plot_roc_pr_curves(y_true, y_probs_dict: dict,
                        output_path: Optional[Path] = None,
                        figsize: Tuple[int, int] = (14, 6)):
    """
    Plot ROC and Precision-Recall curves for multiple models.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_probs_dict : dict
        Dictionary of {model_name: predicted_probabilities}
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.Set2.colors
    
    for i, (name, y_prob) in enumerate(y_probs_dict.items()):
        color = colors[i % len(colors)]
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})',
                 color=color, linewidth=2)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, label=f'{name} (AUC={pr_auc:.3f})',
                 color=color, linewidth=2)
    
    # ROC plot styling
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    
    # PR plot styling
    baseline = y_true.mean()
    ax2.axhline(baseline, color='k', linestyle='--', 
                label=f'Random (={baseline:.3f})', linewidth=1)
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(alpha=0.3)
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig, (ax1, ax2)


def plot_calibration_curve(y_true, y_probs_dict: dict,
                             n_bins: int = 10,
                             output_path: Optional[Path] = None,
                             figsize: Tuple[int, int] = (10, 8)):
    """
    Plot calibration (reliability) curves for multiple models.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_probs_dict : dict
        Dictionary of {model_name: predicted_probabilities}
    n_bins : int
        Number of bins for calibration
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2.colors
    
    for i, (name, y_prob) in enumerate(y_probs_dict.items()):
        color = colors[i % len(colors)]
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins,
                                                  strategy='quantile')
        
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, 
                label=name, color=color, markersize=8)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Positives (True Rate)', fontsize=12, fontweight='bold')
    ax.set_title('Calibration (Reliability) Curve', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig, ax


def plot_feature_importance(importances: pd.Series,
                              top_n: int = 20,
                              output_path: Optional[Path] = None,
                              figsize: Tuple[int, int] = (12, 8)):
    """
    Plot feature importance from Random Forest.
    
    Parameters
    ----------
    importances : pd.Series
        Series with feature names as index and importance values
    top_n : int
        Show top N features
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    top_features = importances.nlargest(top_n).sort_values()
    
    # Get readable names and categories
    readable_names = [get_readable_name(feat, short=True) for feat in top_features.index]
    categories = [get_category(feat) for feat in top_features.index]
    
    # Color by category
    colors = {'Profitability': '#3498db', 'Liquidity': '#2ecc71', 
              'Leverage': '#e74c3c', 'Activity': '#f39c12', 
              'Size': '#9b59b6', 'Other': '#95a5a6'}
    bar_colors = [colors.get(cat, '#95a5a6') for cat in categories]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(readable_names, top_features.values, color=bar_colors, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, top_features.values):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances (Random Forest)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add category legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[cat], label=cat) 
                       for cat in sorted(set(categories))]
    ax.legend(handles=legend_elements, loc='lower right',
              frameon=True, title='Category', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig, ax


def plot_odds_ratios(coefs: pd.DataFrame,
                      top_n: int = 20,
                      output_path: Optional[Path] = None,
                      figsize: Tuple[int, int] = (12, 10)):
    """
    Lollipop plot of odds ratios from GLM with confidence intervals.
    
    Parameters
    ----------
    coefs : pd.DataFrame
        DataFrame with columns: 'feature', 'coef', 'std_err', 'pval'
    top_n : int
        Show top N features by absolute z-score
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    # Calculate odds ratios and CIs
    coefs = coefs.copy()
    coefs['odds_ratio'] = np.exp(coefs['coef'])
    coefs['ci_lower'] = np.exp(coefs['coef'] - 1.96 * coefs['std_err'])
    coefs['ci_upper'] = np.exp(coefs['coef'] + 1.96 * coefs['std_err'])
    coefs['abs_coef'] = np.abs(coefs['coef'])
    
    # Filter out constant and select top N
    coefs = coefs[coefs['feature'] != 'const'].nlargest(top_n, 'abs_coef')
    
    # Get readable names
    readable_names = [get_readable_name(feat, short=True) if feat != 'const' 
                      else 'Intercept' for feat in coefs['feature']]
    
    # Determine significance
    is_significant = coefs['pval'] < 0.05
    colors = ['#e74c3c' if sig else '#95a5a6' for sig in is_significant]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(coefs))
    
    # Plot confidence intervals as lines
    for i, (_, row) in enumerate(coefs.iterrows()):
        ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 
                color=colors[i], linewidth=2, alpha=0.6)
    
    # Plot point estimates as dots
    ax.scatter(coefs['odds_ratio'], y_pos, color=colors, s=100, zorder=3)
    
    # Add reference line at OR=1
    ax.axvline(1, color='black', linestyle='--', linewidth=1.5, 
               label='OR = 1 (no effect)')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(readable_names, fontsize=10)
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Features by Effect Size (Logistic Regression)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    
    # Add significance indicator
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='p < 0.05'),
        Patch(facecolor='#95a5a6', label='p ≥ 0.05')
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
              frameon=True, title='Significance')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig, ax


if __name__ == '__main__':
    print("Visualization module loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_class_imbalance_by_horizon")
    print("  - plot_missingness")
    print("  - plot_distributions_by_class")
    print("  - plot_correlation_heatmap")
    print("  - find_high_correlations")
    print("  - plot_roc_pr_curves")
    print("  - plot_calibration_curve")
    print("  - plot_feature_importance")
    print("  - plot_odds_ratios")
