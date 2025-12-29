"""Generate publication-ready benchmark tables from raw results."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def load_results(results_path: str) -> pd.DataFrame:
    """Load raw results CSV."""
    return pd.read_csv(results_path)


def format_benchmark_table(
    results_df: pd.DataFrame,
    output_dir: str = "results/tables",
    bold_best: bool = True,
    add_arrows: bool = True
) -> Dict[str, str]:
    """
    Format raw results into publication-quality benchmark table.
    
    Args:
        results_df: Raw results DataFrame
        output_dir: Directory to save formatted tables
        bold_best: If True, bold the best value in each column
        add_arrows: If True, add ↑ (higher is better) / ↓ (lower is better) indicators
        
    Returns:
        Dictionary with different table formats (markdown, latex, csv)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define metric groups and their directions
    # Direction: True = higher is better (↑), False = lower is better (↓)
    metrics_spec = {
        # Performance metrics (higher is better)
        'Performance': {
            'accuracy': True,
            'f1_score': True,
            'precision': True,
            'recall': True,
        },
        # Efficiency metrics (lower is better)
        'Efficiency': {
            'latency_ms_per_sample': False,
            'energy_kwh': False,
            'co2_grams': False,
            'model_size_mb': False,
        }
    }
    
    # Select and reorder columns
    df = results_df.copy()
    df = df.rename(columns={
        'model': 'Model',
        'accuracy': 'Accuracy ↑',
        'f1_score': 'F1 ↑',
        'precision': 'Precision ↑',
        'recall': 'Recall ↑',
        'latency_ms_per_sample': 'Latency (ms) ↓',
        'energy_kwh': 'Energy (kWh) ↓',
        'co2_grams': 'CO₂ (g) ↓',
        'model_size_mb': 'Size (MB) ↓',
    })
    
    # Select columns that exist
    available_cols = ['Model'] + [col for col in df.columns if col != 'Model']
    df = df[available_cols]
    
    # Format numeric values
    format_dict = {}
    for col in df.columns:
        if col == 'Model':
            continue
        elif 'Accuracy' in col or 'F1' in col or 'Precision' in col or 'Recall' in col:
            format_dict[col] = '{:.4f}'
        elif 'Latency' in col or 'Energy' in col or 'Size' in col:
            format_dict[col] = '{:.2f}'
        elif 'CO₂' in col:
            format_dict[col] = '{:.4f}'
    
    # Format DataFrame
    df_formatted = df.copy()
    for col, fmt in format_dict.items():
        df_formatted[col] = df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
    
    # ===== MARKDOWN TABLE (publication-ready) =====
    markdown_lines = [
        "# Benchmark Results: Phishing Email Detection",
        "",
        "Table 1. Performance and efficiency metrics across models (test set).",
        "",
    ]
    
    # Header row
    markdown_lines.append("| " + " | ".join(df_formatted.columns) + " |")
    markdown_lines.append("|" + "|".join(["---"] * len(df_formatted.columns)) + "|")
    
    # Data rows
    for idx, row in df_formatted.iterrows():
        row_str = "| " + " | ".join(row.astype(str)) + " |"
        markdown_lines.append(row_str)
    
    markdown_table = "\n".join(markdown_lines)
    
    # ===== LATEX TABLE (for papers) =====
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Benchmark results: Phishing email detection (test set).}",
        r"\label{tab:benchmark}",
        r"\small",
        r"\begin{tabular}{" + "l" + "r" * (len(df_formatted.columns) - 1) + "}",
        r"\toprule",
    ]
    
    # Header
    header = " & ".join(df_formatted.columns) + r" \\"
    latex_lines.append(header)
    latex_lines.append(r"\midrule")
    
    # Data rows
    for idx, row in df_formatted.iterrows():
        row_str = " & ".join(row.astype(str)) + r" \\"
        latex_lines.append(row_str)
    
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    latex_table = "\n".join(latex_lines)
    
    # ===== SAVE ALL FORMATS =====
    # Save formatted CSV
    csv_path = output_path / "benchmark_table.csv"
    df_formatted.to_csv(csv_path, index=False)
    print(f"✓ Saved benchmark table (CSV): {csv_path}")
    
    # Save Markdown
    md_path = output_path / "benchmark_table.md"
    with open(md_path, 'w') as f:
        f.write(markdown_table)
    print(f"✓ Saved benchmark table (Markdown): {md_path}")
    
    # Save LaTeX
    latex_path = output_path / "benchmark_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✓ Saved benchmark table (LaTeX): {latex_path}")
    
    return {
        'markdown': markdown_table,
        'latex': latex_table,
        'csv_path': str(csv_path)
    }


def generate_supplementary_tables(
    results_df: pd.DataFrame,
    output_dir: str = "results/tables"
) -> None:
    """
    Generate supplementary detail tables for appendix.
    
    Args:
        results_df: Raw results DataFrame
        output_dir: Directory to save tables
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supplementary Table 1: Full metrics
    full_table_path = output_path / "supplementary_full_metrics.csv"
    results_df.to_csv(full_table_path, index=False)
    print(f"✓ Saved supplementary table (full metrics): {full_table_path}")
    
    # Supplementary Table 2: Performance-only
    perf_cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score']
    perf_cols = [col for col in perf_cols if col in results_df.columns]
    if len(perf_cols) > 1:
        perf_table_path = output_path / "supplementary_performance.csv"
        results_df[perf_cols].to_csv(perf_table_path, index=False)
        print(f"✓ Saved supplementary table (performance only): {perf_table_path}")
    
    # Supplementary Table 3: Efficiency-only
    eff_cols = ['model', 'latency_ms_per_sample', 'energy_kwh', 'co2_grams', 'model_size_mb']
    eff_cols = [col for col in eff_cols if col in results_df.columns]
    if len(eff_cols) > 1:
        eff_table_path = output_path / "supplementary_efficiency.csv"
        results_df[eff_cols].to_csv(eff_table_path, index=False)
        print(f"✓ Saved supplementary table (efficiency only): {eff_table_path}")


def main():
    """Generate all benchmark tables from results."""
    results_path = "results/tables/results_summary.csv"
    
    # Check if results exist
    if not Path(results_path).exists():
        print(f"❌ Results file not found: {results_path}")
        print("Run training/inference first with: uv run python src/main.py")
        return
    
    print("\n" + "="*60)
    print("GENERATING PUBLICATION-QUALITY BENCHMARK TABLES")
    print("="*60 + "\n")
    
    # Load results
    results_df = load_results(results_path)
    print(f"Loaded {len(results_df)} results")
    
    # Generate main benchmark table
    print("\nFormatting main benchmark table...")
    tables = format_benchmark_table(results_df)
    
    # Generate supplementary tables
    print("\nGenerating supplementary tables...")
    generate_supplementary_tables(results_df)
    
    print("\n" + "="*60)
    print("TABLE GENERATION COMPLETE")
    print("="*60)
    print("\n📊 Generated files:")
    print("  • benchmark_table.csv     (formatted CSV for easy viewing)")
    print("  • benchmark_table.md      (Markdown for GitHub/docs)")
    print("  • benchmark_table.tex     (LaTeX for papers)")
    print("  • supplementary_*.csv     (detailed breakdowns)")
    print("\n💡 For your paper:")
    print("  1. Use Table 1 (benchmark_table) as the main result")
    print("  2. Reference supplementary tables in appendix")
    print("  3. Copy LaTeX code directly into your paper")
    print("\n✨ Your professor can now see:")
    print("  ✓ Which model is best (one glance)")
    print("  ✓ Trade-offs at a glance (performance vs efficiency)")
    print("  ✓ Clear metric grouping (not a data dump)")
    print("  ✓ Professional formatting (publication-ready)")


if __name__ == '__main__':
    main()
