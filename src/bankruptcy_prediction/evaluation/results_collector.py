"""
Results Collector
=================

Centralized system for collecting and comparing model results across all notebooks.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime


class ResultsCollector:
    """
    Collect and manage results from multiple models.
    
    All model notebooks save results to a central location.
    Master report auto-loads and displays everything.
    
    Example:
        # In modeling notebook
        >>> results = ResultsCollector()
        >>> results.add({
        ...     'model_name': 'Random Forest',
        ...     'horizon': 1,
        ...     'roc_auc': 0.90,
        ...     'pr_auc': 0.65,
        ... })
        >>> results.save()
        
        # In master report
        >>> results = ResultsCollector.load_all()
        >>> results.show_comparison()
        >>> results.plot_comparison()
    """
    
    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize results collector.
        
        Parameters:
            results_dir: Directory to save/load results. Auto-detects if None.
        """
        if results_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            self.results_dir = project_root / "results" / "models"
        else:
            self.results_dir = Path(results_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / "all_results.csv"
        self.metadata_file = self.results_dir / "metadata.json"
        
        self.results: List[Dict] = []
    
    def add(self, result: Dict):
        """
        Add a model result.
        
        Parameters:
            result: Dictionary with model metrics
        
        Required fields:
            - model_name: str
            - horizon: int
            - roc_auc: float
            - pr_auc: float
        
        Optional fields:
            - recall_1pct_fpr: float
            - recall_5pct_fpr: float
            - brier_score: float
            - precision: float
            - f1_score: float
            - training_time: float
            - n_features: int
            - ... any other metrics
        """
        # Validate required fields
        required = ['model_name', 'horizon', 'roc_auc', 'pr_auc']
        missing = [f for f in required if f not in result]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        self.results.append(result)
        print(f"✓ Added result: {result['model_name']} (horizon={result['horizon']})")
    
    def save(self):
        """Save results to CSV file."""
        if not self.results:
            print("⚠ No results to save")
            return
        
        df = pd.DataFrame(self.results)
        
        # Append to existing results if file exists
        if self.results_file.exists():
            existing = pd.read_csv(self.results_file)
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_csv(self.results_file, index=False)
        print(f"✓ Saved {len(self.results)} result(s) to: {self.results_file}")
        
        # Save metadata
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'total_results': len(df),
            'models': sorted(df['model_name'].unique().tolist()),
            'horizons': sorted(df['horizon'].unique().tolist()),
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load_all(cls, results_dir: Optional[Path] = None) -> 'ResultsCollector':
        """
        Load all saved results.
        
        Parameters:
            results_dir: Directory with results
        
        Returns:
            ResultsCollector with loaded results
        """
        collector = cls(results_dir)
        
        if not collector.results_file.exists():
            print("⚠ No results file found")
            return collector
        
        df = pd.read_csv(collector.results_file)
        collector.results = df.to_dict('records')
        
        print(f"✓ Loaded {len(collector.results)} results")
        return collector
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get results as pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)
    
    def show_comparison(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Show comparison table of all models.
        
        Parameters:
            metrics: List of metrics to display. If None, shows key metrics.
        
        Returns:
            Formatted comparison DataFrame
        """
        if not self.results:
            print("⚠ No results to display")
            return pd.DataFrame()
        
        df = self.get_dataframe()
        
        if metrics is None:
            metrics = ['roc_auc', 'pr_auc', 'recall_1pct_fpr', 'recall_5pct_fpr', 'brier_score']
            # Only include metrics that exist
            metrics = [m for m in metrics if m in df.columns]
        
        # Pivot to show models as rows
        display_cols = ['model_name', 'horizon'] + metrics
        display_cols = [c for c in display_cols if c in df.columns]
        
        comparison = df[display_cols].copy()
        
        # Sort by ROC-AUC
        comparison = comparison.sort_values(['horizon', 'roc_auc'], ascending=[True, False])
        
        return comparison
    
    def best_model(self, horizon: Optional[int] = None, 
                   metric: str = 'roc_auc') -> Dict:
        """
        Get best performing model.
        
        Parameters:
            horizon: Filter by horizon. If None, considers all.
            metric: Metric to use for ranking
        
        Returns:
            Dictionary with best model info
        """
        if not self.results:
            return {}
        
        df = self.get_dataframe()
        
        if horizon is not None:
            df = df[df['horizon'] == horizon]
        
        if df.empty:
            return {}
        
        best_idx = df[metric].idxmax()
        best = df.loc[best_idx].to_dict()
        
        return best
    
    def plot_comparison(self, output_path: Optional[Path] = None):
        """
        Create comparison plots.
        
        Parameters:
            output_path: Path to save figure
        """
        if not self.results:
            print("⚠ No results to plot")
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        df = self.get_dataframe()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC-AUC comparison
        ax1 = axes[0]
        models = df['model_name'].unique()
        horizons = sorted(df['horizon'].unique())
        
        x = np.arange(len(horizons))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            model_data = df[df['model_name'] == model].sort_values('horizon')
            aucs = model_data['roc_auc'].values
            ax1.bar(x + i*width, aucs, width, label=model, alpha=0.8)
        
        ax1.set_xlabel('Horizon (years)', fontweight='bold')
        ax1.set_ylabel('ROC-AUC', fontweight='bold')
        ax1.set_title('Model Comparison: ROC-AUC by Horizon', fontweight='bold')
        ax1.set_xticks(x + width * (len(models)-1) / 2)
        ax1.set_xticklabels(horizons)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0.5, 1.0])
        
        # Recall at 1% FPR comparison
        ax2 = axes[1]
        if 'recall_1pct_fpr' in df.columns:
            for i, model in enumerate(models):
                model_data = df[df['model_name'] == model].sort_values('horizon')
                recalls = model_data['recall_1pct_fpr'].values
                ax2.bar(x + i*width, recalls, width, label=model, alpha=0.8)
            
            ax2.set_xlabel('Horizon (years)', fontweight='bold')
            ax2.set_ylabel('Recall @ 1% FPR', fontweight='bold')
            ax2.set_title('Model Comparison: Recall at 1% FPR', fontweight='bold')
            ax2.set_xticks(x + width * (len(models)-1) / 2)
            ax2.set_xticklabels(horizons)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_ylim([0, 1.0])
        else:
            ax2.text(0.5, 0.5, 'Recall @ 1% FPR not available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to: {output_path}")
        
        return fig, axes
    
    def clear(self):
        """Clear all results from memory (doesn't delete saved file)."""
        self.results = []
        print("✓ Cleared results from memory")
    
    def delete_saved(self):
        """Delete saved results file (use with caution!)."""
        if self.results_file.exists():
            self.results_file.unlink()
            print(f"✓ Deleted: {self.results_file}")
        if self.metadata_file.exists():
            self.metadata_file.unlink()
            print(f"✓ Deleted: {self.metadata_file}")


if __name__ == '__main__':
    # Test results collector
    print("Testing ResultsCollector...\n")
    
    # Create dummy results
    results = ResultsCollector()
    
    results.add({
        'model_name': 'Logistic Regression',
        'horizon': 1,
        'roc_auc': 0.87,
        'pr_auc': 0.58,
        'recall_1pct_fpr': 0.50,
        'recall_5pct_fpr': 0.72,
        'brier_score': 0.045,
    })
    
    results.add({
        'model_name': 'Random Forest',
        'horizon': 1,
        'roc_auc': 0.90,
        'pr_auc': 0.65,
        'recall_1pct_fpr': 0.574,
        'recall_5pct_fpr': 0.80,
        'brier_score': 0.038,
    })
    
    # Save
    results.save()
    
    # Load and display
    print("\nLoading results...")
    loaded = ResultsCollector.load_all()
    
    print("\nComparison table:")
    print(loaded.show_comparison())
    
    print("\nBest model:")
    best = loaded.best_model(horizon=1)
    print(f"  {best['model_name']}: ROC-AUC = {best['roc_auc']:.3f}")
    
    print("\n✓ ResultsCollector test complete!")
