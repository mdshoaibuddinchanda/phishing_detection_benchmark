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
    
    x = list(range(len(models)))
    width = 0.35
    
    bars_acc = ax.bar([i - width/2 for i in x], accuracy, width, label='Accuracy', alpha=0.9)
    bars_f1 = ax.bar([i + width/2 for i in x], f1, width, label='F1-Score', alpha=0.9)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: Accuracy vs F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    # Dynamic y-limits to show differences
    min_val = min(min(accuracy), min(f1))
    max_val = max(max(accuracy), max(f1))
    pad = max(0.005, (max_val - min_val) * 0.2)
    y_min = max(0.95, min_val - pad)
    y_max = min(1.0, max_val + pad)
    ax.set_ylim([y_min, y_max])
    
    # Add numeric labels on bars (rotated to prevent overlap)
    for bar in bars_acc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + (y_max - y_min) * 0.01,
                f'{h:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
    for bar in bars_f1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + (y_max - y_min) * 0.01,
                f'{h:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
    
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
    
    x = list(range(len(models)))
    width = 0.18
    
    all_values = []
    bars_collection = []
    for i, metric in enumerate(metrics):
        values = results_df[metric].tolist()
        all_values.extend(values)
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], values, width, 
                      label=metric.replace('_', ' ').title(), alpha=0.85)
        bars_collection.append(bars)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Complete Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    
    # Dynamic y-limits to show differences
    min_val = min(all_values)
    max_val = max(all_values)
    pad = max(0.005, (max_val - min_val) * 0.2)
    y_min = max(0.95, min_val - pad)
    y_max = min(1.0, max_val + pad)
    ax.set_ylim([y_min, y_max])
    
    # Add numeric labels on all bars (rotated to prevent overlap)
    for bars in bars_collection:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + (y_max - y_min) * 0.01,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=6, rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved all metrics plot to {output_path}")
