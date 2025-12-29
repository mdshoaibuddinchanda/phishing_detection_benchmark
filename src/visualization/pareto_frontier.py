"""Pareto frontier plot for accuracy-energy trade-offs."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .style import apply_publication_style

# Apply publication-quality styling once
apply_publication_style()


def plot_pareto_frontier(
    results_df: pd.DataFrame,
    output_path: str,
    x_metric: str = 'energy_kwh',
    y_metric: str = 'f1_score',
    figsize: tuple = (10, 8),
    dpi: int = 300
) -> None:
    """
    Create Pareto frontier plot showing accuracy-energy trade-offs.
    
    Args:
        results_df: DataFrame with model results
        output_path: Path to save figure
        x_metric: Metric for x-axis (efficiency)
        y_metric: Metric for y-axis (performance)
        figsize: Figure size
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = results_df['model'].tolist()
    x_values = results_df[x_metric].tolist()
    y_values = results_df[y_metric].tolist()
    
    # Plot points
    colors = ['#000000', '#555555', '#888888']
    for i, (model, x, y) in enumerate(zip(models, x_values, y_values)):
        color = colors[i] if i < len(colors) else '#bbbbbb'
        ax.scatter(x, y, s=100, alpha=0.7, color=color, edgecolors='black', linewidth=1.5, label=model)
        
        # Add model name annotation
        ax.annotate(
            model,
            (x, y),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5)
        )
    
    # Labels and title
    x_label_map = {
        'energy_kwh': 'Energy Consumption (kWh)',
        'co2_grams': 'CO₂ Emissions (grams)',
        'latency_ms_per_sample': 'Latency (ms per sample)',
        'model_size_mb': 'Model Size (MB)'
    }
    
    y_label_map = {
        'f1_score': 'F1-Score',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    ax.set_xlabel(x_label_map.get(x_metric, x_metric))
    ax.set_ylabel(y_label_map.get(y_metric, y_metric))
    ax.set_title('Accuracy–Energy Trade-off (Pareto Frontier)')
    
    # Minimal grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Set y-axis limits for performance metrics
    if y_metric in ['accuracy', 'f1_score', 'precision', 'recall']:
        y_min = min(y_values) - 0.05
        y_max = 1.0
        ax.set_ylim([max(0, y_min), y_max])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Pareto frontier to {output_path}")


def plot_multi_objective_comparison(
    results_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (14, 6),
    dpi: int = 300
) -> None:
    """
    Create multi-panel plot showing multiple trade-offs.
    
    Args:
        results_df: DataFrame with model results
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    models = results_df['model'].tolist()
    colors = ['#000000', '#555555', '#888888']
    
    # Plot 1: F1-Score vs Energy
    ax1 = axes[0]
    for i, model in enumerate(models):
        row = results_df[results_df['model'] == model].iloc[0]
        color = colors[i] if i < len(colors) else '#bbbbbb'
        ax1.scatter(row['energy_kwh'], row['f1_score'], s=100, alpha=0.7, 
                   color=color, edgecolors='black', linewidth=1.5, label=model)
    
    ax1.set_xlabel('Energy Consumption (kWh)')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('Performance vs Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Plot 2: F1-Score vs Latency
    ax2 = axes[1]
    for i, model in enumerate(models):
        row = results_df[results_df['model'] == model].iloc[0]
        color = colors[i] if i < len(colors) else '#bbbbbb'
        ax2.scatter(row['latency_ms_per_sample'], row['f1_score'], s=100, alpha=0.7,
                   color=color, edgecolors='black', linewidth=1.5, label=model)
    
    ax2.set_xlabel('Latency (ms per sample)')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Performance vs Latency')
    ax2.legend()
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved multi-objective comparison to {output_path}")
