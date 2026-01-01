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
    
    # Distinct colors for each model (colorful palette)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (model, x, y) in enumerate(zip(models, x_values, y_values)):
        color = colors[i % len(colors)]
        ax.scatter(x, y, s=120, alpha=0.8, color=color, edgecolors='black', 
                  linewidth=1.5, label=model, zorder=3)
        
        # Smart label positioning to avoid overlap
        # Alternate label positions based on index and x-value
        if i % 2 == 0:
            xytext = (10, -15)  # Bottom-right
        else:
            xytext = (10, 10)   # Top-right
        
        # For points on the left side, place labels on the right
        if x < (max(x_values) + min(x_values)) / 2:
            xytext = (12, 0)
        
        # Add model name annotation WITHOUT box
        ax.annotate(
            model,
            (x, y),
            xytext=xytext,
            textcoords='offset points',
            fontsize=8,
            fontweight='bold',
            color=color
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
    
    # Set y-axis limits for performance metrics - zoom into high-performance range
    if y_metric in ['accuracy', 'f1_score', 'precision', 'recall']:
        y_min = min(y_values)
        y_max = max(y_values)
        # Add small padding
        padding = max(0.002, (y_max - y_min) * 0.15)
        y_min = max(0.98, y_min - padding)  # Don't go below 0.98
        y_max = min(1.0, y_max + padding)
        ax.set_ylim([y_min, y_max])
    
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
    # Distinct colors for each model (colorful palette)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot 1: F1-Score vs Energy
    ax1 = axes[0]
    for i, model in enumerate(models):
        row = results_df[results_df['model'] == model].iloc[0]
        color = colors[i % len(colors)]
        ax1.scatter(row['energy_kwh'], row['f1_score'], s=120, alpha=0.8, 
                   color=color, edgecolors='black', linewidth=1.5, zorder=3)
        # Add direct label near point
        ax1.annotate(model, (row['energy_kwh'], row['f1_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold', color=color)
    
    ax1.set_xlabel('Energy Consumption (kWh)')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('Performance vs Energy')
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    # Plot 2: F1-Score vs Latency
    ax2 = axes[1]
    for i, model in enumerate(models):
        row = results_df[results_df['model'] == model].iloc[0]
        color = colors[i % len(colors)]
        ax2.scatter(row['latency_ms_per_sample'], row['f1_score'], s=120, alpha=0.8,
                   color=color, edgecolors='black', linewidth=1.5, zorder=3)
        # Add direct label near point
        ax2.annotate(model, (row['latency_ms_per_sample'], row['f1_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold', color=color)
    
    ax2.set_xlabel('Latency (ms per sample)')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Performance vs Latency')
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved multi-objective comparison to {output_path}")
