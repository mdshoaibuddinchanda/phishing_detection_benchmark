"""Plot accuracy comparison across models."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from .style import apply_publication_style

# Apply publication-quality styling once
apply_publication_style()


def plot_accuracy_comparison(
    results_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (10, 6),
    dpi: int = 300
) -> None:
    """
    Create bar chart comparing accuracy across models.
    
    Args:
        results_df: DataFrame with model results
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = results_df['model'].tolist()
    accuracy = results_df['accuracy'].tolist()
    f1 = results_df['f1_score'].tolist()
    
    x = range(len(models))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], accuracy, width, label='Accuracy', alpha=0.8)
    ax.bar([i + width/2 for i in x], f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: Accuracy vs F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved accuracy comparison to {output_path}")


def plot_all_metrics(
    results_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (12, 6),
    dpi: int = 300
) -> None:
    """
    Create grouped bar chart for all performance metrics.
    
    Args:
        results_df: DataFrame with model results
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = results_df['model'].tolist()
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    x = range(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = results_df[metric].tolist()
        offset = (i - len(metrics)/2 + 0.5) * width
        ax.bar([xi + offset for xi in x], values, width, label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Complete Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved all metrics plot to {output_path}")
