"""
Cross-Horizon Robustness Testing
=================================

Test model generalization across different prediction horizons.
Train on one horizon, evaluate on others to assess robustness.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def prepare_horizon_data(df: pd.DataFrame, 
                          horizon: int,
                          drop_cols: list = ['y', 'horizon']) -> Tuple:
    """
    Extract features and target for a specific horizon.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all horizons
    horizon : int
        Horizon to extract (1-5)
    drop_cols : list
        Columns to drop from features
    
    Returns
    -------
    X : pd.DataFrame
        Features
    y : np.ndarray
        Target labels
    """
    df_h = df[df['horizon'] == horizon].copy()
    y = df_h['y'].astype(int).values
    X = df_h.drop(columns=drop_cols)
    return X, y


def cross_horizon_evaluation(model,
                              df_full: pd.DataFrame,
                              train_horizon: int,
                              eval_horizons: list,
                              model_name: str = "Model") -> pd.DataFrame:
    """
    Train on one horizon and evaluate on multiple horizons.
    
    Parameters
    ----------
    model : sklearn estimator
        Fitted or unfitted model with .fit() and .predict_proba()
    df_full : pd.DataFrame
        Full dataset with 'horizon' and 'y' columns
    train_horizon : int
        Horizon to train on
    eval_horizons : list
        List of horizons to evaluate on
    model_name : str
        Name for results table
    
    Returns
    -------
    pd.DataFrame
        Results table with metrics per horizon
    """
    from sklearn.base import clone
    
    # Train on specified horizon
    X_train, y_train = prepare_horizon_data(df_full, train_horizon)
    
    # Clone and fit model
    model_fitted = clone(model)
    model_fitted.fit(X_train, y_train)
    
    results = []
    
    for eval_h in eval_horizons:
        X_eval, y_eval = prepare_horizon_data(df_full, eval_h)
        
        # Ensure same features
        missing_cols = set(X_train.columns) - set(X_eval.columns)
        extra_cols = set(X_eval.columns) - set(X_train.columns)
        
        if missing_cols:
            print(f"Warning: Missing columns in h={eval_h}: {missing_cols}")
            for col in missing_cols:
                X_eval[col] = 0
        
        if extra_cols:
            X_eval = X_eval.drop(columns=list(extra_cols))
        
        # Reorder to match training
        X_eval = X_eval[X_train.columns]
        
        # Predict
        y_prob = model_fitted.predict_proba(X_eval)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model': model_name,
            'train_horizon': train_horizon,
            'eval_horizon': eval_h,
            'n_samples': len(y_eval),
            'bankruptcy_rate': y_eval.mean(),
            'roc_auc': roc_auc_score(y_eval, y_prob),
            'pr_auc': average_precision_score(y_eval, y_prob),
            'brier': brier_score_loss(y_eval, y_prob),
        }
        
        # Add recall at fixed FPR
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_eval, y_prob)
        
        # Recall at 1% FPR
        mask = fpr <= 0.01
        if np.any(mask):
            metrics['recall_1pct_fpr'] = float(np.max(tpr[mask]))
        else:
            metrics['recall_1pct_fpr'] = 0.0
        
        # Recall at 5% FPR
        mask = fpr <= 0.05
        if np.any(mask):
            metrics['recall_5pct_fpr'] = float(np.max(tpr[mask]))
        else:
            metrics['recall_5pct_fpr'] = 0.0
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def full_cross_horizon_matrix(models_dict: Dict,
                               df_full: pd.DataFrame,
                               horizons: list = [1, 2, 3, 4, 5],
                               output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Create full cross-horizon evaluation matrix for multiple models.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary of {model_name: model_instance}
    df_full : pd.DataFrame
        Full dataset with all horizons
    horizons : list
        List of horizons to test
    output_path : Path, optional
        Path to save results CSV
    
    Returns
    -------
    pd.DataFrame
        Complete results table
    """
    all_results = []
    
    total_combos = len(models_dict) * len(horizons) * len(horizons)
    counter = 0
    
    print(f"\n{'='*60}")
    print(f"Running Cross-Horizon Evaluation")
    print(f"Models: {len(models_dict)}, Horizons: {len(horizons)}")
    print(f"Total combinations: {total_combos}")
    print(f"{'='*60}\n")
    
    for model_name, model in models_dict.items():
        print(f"\n[{model_name}]")
        
        for train_h in horizons:
            print(f"  Training on horizon {train_h}...", end=" ")
            
            results = cross_horizon_evaluation(
                model=model,
                df_full=df_full,
                train_horizon=train_h,
                eval_horizons=horizons,
                model_name=model_name
            )
            
            all_results.append(results)
            counter += len(horizons)
            print(f"✓ ({counter}/{total_combos} complete)")
    
    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Sort for readability
    final_results = final_results.sort_values(['model', 'train_horizon', 'eval_horizon'])
    
    if output_path:
        final_results.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Cross-Horizon Evaluation Complete!")
    print(f"{'='*60}\n")
    
    return final_results


def plot_cross_horizon_heatmap(results: pd.DataFrame,
                                 model_name: str,
                                 metric: str = 'roc_auc',
                                 output_path: Optional[Path] = None,
                                 figsize: Tuple = (10, 8)):
    """
    Heatmap showing metric degradation across horizons.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results from cross_horizon_evaluation
    model_name : str
        Model to plot
    metric : str
        Metric to visualize
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Filter for specific model
    data = results[results['model'] == model_name].copy()
    
    # Pivot to matrix form
    matrix = data.pivot(index='train_horizon', 
                        columns='eval_horizon', 
                        values=metric)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, center=0.75,
                cbar_kws={'label': metric.upper().replace('_', ' ')},
                linewidths=0.5, ax=ax)
    
    ax.set_xlabel('Evaluation Horizon (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Horizon (years)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cross-Horizon {metric.upper()} - {model_name}',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Highlight diagonal (same horizon)
    for i in range(len(matrix)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                    edgecolor='blue', lw=3))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    return fig, ax


def summarize_degradation(results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize metric degradation when evaluating on different horizons.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results from cross_horizon_evaluation
    
    Returns
    -------
    pd.DataFrame
        Summary table showing degradation
    """
    summary = []
    
    for model in results['model'].unique():
        model_data = results[results['model'] == model].copy()
        
        for train_h in model_data['train_horizon'].unique():
            # Get same-horizon performance (baseline)
            same_h = model_data[
                (model_data['train_horizon'] == train_h) & 
                (model_data['eval_horizon'] == train_h)
            ]
            
            if len(same_h) == 0:
                continue
            
            baseline_roc = same_h['roc_auc'].values[0]
            baseline_pr = same_h['pr_auc'].values[0]
            
            # Get other horizons
            other_h = model_data[
                (model_data['train_horizon'] == train_h) & 
                (model_data['eval_horizon'] != train_h)
            ]
            
            for _, row in other_h.iterrows():
                summary.append({
                    'model': model,
                    'train_horizon': train_h,
                    'eval_horizon': row['eval_horizon'],
                    'horizon_gap': abs(row['eval_horizon'] - train_h),
                    'roc_auc_baseline': baseline_roc,
                    'roc_auc_transfer': row['roc_auc'],
                    'roc_auc_drop': baseline_roc - row['roc_auc'],
                    'pr_auc_baseline': baseline_pr,
                    'pr_auc_transfer': row['pr_auc'],
                    'pr_auc_drop': baseline_pr - row['pr_auc'],
                })
    
    df_summary = pd.DataFrame(summary)
    
    # Add percentage drops
    if len(df_summary) > 0:
        df_summary['roc_auc_drop_pct'] = (
            df_summary['roc_auc_drop'] / df_summary['roc_auc_baseline'] * 100
        )
        df_summary['pr_auc_drop_pct'] = (
            df_summary['pr_auc_drop'] / df_summary['pr_auc_baseline'] * 100
        )
    
    return df_summary


if __name__ == '__main__':
    print("Cross-horizon evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("  - prepare_horizon_data")
    print("  - cross_horizon_evaluation")
    print("  - full_cross_horizon_matrix")
    print("  - plot_cross_horizon_heatmap")
    print("  - summarize_degradation")
