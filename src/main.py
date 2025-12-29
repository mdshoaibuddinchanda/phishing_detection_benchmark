"""
Main orchestration script for phishing detection benchmarking.
Runs the complete pipeline from data processing to visualization.
"""

import argparse
from pathlib import Path
import pandas as pd
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data import (
    load_raw_data,
    preprocess_dataframe,
    split_dataset,
    save_splits,
    load_processed_data
)
from src.training import train_model
from src.inference import benchmark_model
from src.metrics import compute_metrics_summary, print_metrics
from src.visualization import (
    plot_accuracy_comparison,
    plot_all_metrics,
    plot_energy_comparison,
    plot_co2_comparison,
    plot_latency_comparison,
    plot_model_size_comparison,
    plot_pareto_frontier,
    plot_multi_objective_comparison
)


def prepare_data(config):
    """Preprocess and split dataset."""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)
    
    data_config = config['data']
    
    # Load raw data
    print("\nLoading raw data...")
    df = load_raw_data(data_config['raw_path'])
    
    # Preprocess
    print("\nPreprocessing text...")
    df_clean = preprocess_dataframe(df, text_col=data_config['text_column'])
    
    # Split dataset
    print("\nSplitting dataset...")
    train_df, val_df, test_df = split_dataset(
        df_clean,
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio'],
        random_seed=config['random_seed'],
        stratify_col=data_config['label_column']
    )
    
    # Save splits
    save_splits(train_df, val_df, test_df, data_config['processed_dir'])
    
    print("\nData preparation complete!")
    return train_df, val_df, test_df


def train_all_models(config, train_df, val_df, force_train: bool = False, allow_resume: bool = True):
    """Train all models.

    Args:
        config: Full configuration dict
        train_df: Training split dataframe
        val_df: Validation split dataframe
        force_train: If True, do not auto-skip even if artifacts exist
        allow_resume: If True, resume training from latest checkpoint in output_dir
    """
    import gc
    import torch
    
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    models_config = config['models']
    
    for model_key, model_config in models_config.items():
        # Auto-skip HF models that already have trained artifacts unless forced
        out_dir = Path(model_config['output_dir'])
        trained_artifacts = [out_dir / 'model.safetensors', out_dir / 'training_args.bin']
        if (not force_train) and all(p.exists() for p in trained_artifacts):
            print(f"\n[Skip] {model_key} already trained (artifacts present); skipping training")
            continue

        print(f"Training: {model_key}")
        print(f"{'='*60}")
        
        # Merge model-specific training config with global training config
        training_config = config['training'].copy()
        if 'training' in model_config:
            training_config.update(model_config['training'])
        
        train_model(
            model_name=model_config['name'],
            train_df=train_df,
            val_df=val_df,
            output_dir=model_config['output_dir'],
            config={**model_config, 'training': training_config},
            random_seed=config['random_seed'],
            allow_resume=allow_resume
        )
        
        # Clear GPU memory between models to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\n[Memory] GPU cache cleared after {model_key}")
    
    print("\n" + "="*60)
    print("All models trained successfully!")
    print("="*60)


def benchmark_all_models(config, test_df):
    """Benchmark all models with energy tracking."""
    print("\n" + "="*60)
    print("STEP 3: BENCHMARKING WITH ENERGY TRACKING")
    print("="*60)
    
    models_config = config['models']
    all_results = []
    
    for model_key, model_config in models_config.items():
        # Benchmark HF models only
        model_dir = model_config['output_dir']
        
        # Benchmark model
        results = benchmark_model(
            model_dir=model_dir,
            test_df=test_df,
            model_key=model_key,
            config=config
        )
        
        # Compute comprehensive metrics
        metrics = compute_metrics_summary(
            predictions=results['predictions'],
            true_labels=results['true_labels'],
            model_name=model_key,
            model_size_mb=results['model_size_mb'],
            latency_ms=results['latency_ms_per_sample'],
            energy_kwh=results['energy_kwh'],
            co2_grams=results['co2_grams']
        )
        
        all_results.append(metrics)
        print_metrics(metrics, model_name=model_key)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_config = config['results']
    results_path = Path(results_config['summary_file'])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Saved results summary to {results_path}")
    print(f"{'='*60}")
    
    return results_df


def generate_visualizations(config, results_df):
    """Generate all plots for the paper."""
    print("\n" + "="*60)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*60)
    
    results_config = config['results']
    viz_config = config.get('visualization', {})
    figures_dir = Path(results_config['figures_dir'])
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    figsize = tuple(viz_config.get('figsize', [10, 6]))
    dpi = viz_config.get('dpi', 300)
    
    # Accuracy plots
    print("\nGenerating accuracy plots...")
    plot_accuracy_comparison(
        results_df,
        str(figures_dir / 'accuracy_comparison.png'),
        figsize=figsize,
        dpi=dpi
    )
    
    plot_all_metrics(
        results_df,
        str(figures_dir / 'all_metrics.png'),
        figsize=(12, 6),
        dpi=dpi
    )
    
    # Energy plots
    print("\nGenerating energy plots...")
    plot_energy_comparison(
        results_df,
        str(figures_dir / 'energy_comparison.png'),
        figsize=figsize,
        dpi=dpi
    )
    
    plot_co2_comparison(
        results_df,
        str(figures_dir / 'co2_comparison.png'),
        figsize=figsize,
        dpi=dpi
    )
    
    plot_latency_comparison(
        results_df,
        str(figures_dir / 'latency_comparison.png'),
        figsize=figsize,
        dpi=dpi
    )
    
    plot_model_size_comparison(
        results_df,
        str(figures_dir / 'model_size_comparison.png'),
        figsize=figsize,
        dpi=dpi
    )
    
    # Pareto frontier
    print("\nGenerating Pareto frontier...")
    plot_pareto_frontier(
        results_df,
        str(figures_dir / 'pareto_frontier.png'),
        x_metric='energy_kwh',
        y_metric='f1_score',
        figsize=(10, 8),
        dpi=dpi
    )
    
    plot_multi_objective_comparison(
        results_df,
        str(figures_dir / 'multi_objective.png'),
        figsize=(14, 6),
        dpi=dpi
    )
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to {figures_dir}/")
    print(f"{'='*60}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Phishing Detection Benchmarking Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip data preparation (use existing processed data)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (use existing models)'
    )
    parser.add_argument(
        '--only-visualize',
        action='store_true',
        help='Only generate visualizations from existing results'
    )
    parser.add_argument(
        '--force-train',
        action='store_true',
        help='Force training even if trained artifacts exist'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume-from-checkpoint during training'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    data_config = config['data']
    
    if args.only_visualize:
        # Load existing results and visualize
        print("Loading existing results...")
        results_df = pd.read_csv(config['results']['summary_file'])
        generate_visualizations(config, results_df)
        print("\n✅ Visualization complete!")
        return
    
    # Step 1: Data preparation
    if args.skip_data:
        print("Skipping data preparation, loading existing data...")
        train_df, val_df, test_df = load_processed_data(
            data_config['train_file'],
            data_config['val_file'],
            data_config['test_file']
        )
    else:
        train_df, val_df, test_df = prepare_data(config)
    
    # Step 2: Model training
    if not args.skip_training:
        train_all_models(
            config,
            train_df,
            val_df,
            force_train=args.force_train,
            allow_resume=(not args.no_resume)
        )
    else:
        print("\nSkipping model training, using existing models...")
    
    # Step 3: Benchmarking
    results_df = benchmark_all_models(config, test_df)
    
    # Step 4: Visualization
    generate_visualizations(config, results_df)
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETE!")
    print("="*60)
    print("\nResults summary:")
    print(results_df.to_string(index=False))
    print(f"\nAll outputs saved to results/")
    print(f"Figures ready for paper: results/figures/")


if __name__ == '__main__':
    main()
