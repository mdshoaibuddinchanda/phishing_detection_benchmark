"""Plot energy consumption and efficiency metrics."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from .style import apply_publication_style

# Apply publication-quality styling once
apply_publication_style()


def plot_energy_comparison(
    results_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (10, 6),
    dpi: int = 300
) -> None:
    """
    Create bar chart comparing energy consumption across models.
    
    Args:
        results_df: DataFrame with model results
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = results_df['model'].tolist()
    energy = results_df['energy_kwh'].tolist()
    
    # Distinct colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    bar_colors = [colors[i % len(colors)] for i in range(len(models))]
    
    bars = ax.bar(models, energy, alpha=0.85, color=bar_colors, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Energy Consumption (kWh)')
    ax.set_title('Energy Consumption Comparison')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved energy comparison to {output_path}")


def plot_co2_comparison(
    results_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (10, 6),
    dpi: int = 300
) -> None:
    """
    Create bar chart comparing CO2 emissions across models.
    
    Args:
        results_df: DataFrame with model results
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = results_df['model'].tolist()
    co2 = results_df['co2_grams'].tolist()
    
    # Distinct colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    bar_colors = [colors[i % len(colors)] for i in range(len(models))]
    
    bars = ax.bar(models, co2, alpha=0.85, color=bar_colors, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('CO₂ Emissions (grams)')
    ax.set_title('CO₂ Emissions Comparison')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved CO2 comparison to {output_path}")


def plot_latency_comparison(
    results_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (10, 6),
    dpi: int = 300
) -> None:
    """
    Create bar chart comparing inference latency across models.
    
    Args:
        results_df: DataFrame with model results
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = results_df['model'].tolist()
    latency = results_df['latency_ms_per_sample'].tolist()
    
    # Distinct colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    bar_colors = [colors[i % len(colors)] for i in range(len(models))]
    
    bars = ax.bar(models, latency, alpha=0.85, color=bar_colors, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (ms per sample)')
    ax.set_title('Inference Latency Comparison')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved latency comparison to {output_path}")


def plot_model_size_comparison(
    results_df: pd.DataFrame,
    output_path: str,
    figsize: tuple = (10, 6),
    dpi: int = 300
) -> None:
    """
    Create bar chart comparing model sizes.
    
    Args:
        results_df: DataFrame with model results
        output_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = results_df['model'].tolist()
    sizes = results_df['model_size_mb'].tolist()
    
    # Distinct colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    bar_colors = [colors[i % len(colors)] for i in range(len(models))]
    
    bars = ax.bar(models, sizes, alpha=0.85, color=bar_colors, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Model Size Comparison')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model size comparison to {output_path}")
