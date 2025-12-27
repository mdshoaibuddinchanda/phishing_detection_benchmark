"""Benchmark inference with energy tracking."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from ..energy import EnergyTracker


def benchmark_model(
    model_dir: str,
    test_df: pd.DataFrame,
    model_key: str,
    config: Dict[str, Any],
    text_col: str = 'text',
    label_col: str = 'label'
) -> Dict[str, Any]:
    """
    Benchmark model with energy tracking.
    
    Args:
        model_dir: Directory containing saved model
        test_df: Test DataFrame
        model_key: Model identifier (bert_large, distilbert)
        config: Full configuration dictionary
        text_col: Name of text column
        label_col: Name of label column
        
    Returns:
        Dictionary with predictions, labels, and energy metrics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_key}")
    print(f"{'='*60}")
    
    # Load configuration
    inference_config = config.get('inference', {})
    energy_config = config.get('energy', {})
    model_config = config['models'][model_key]
    
    batch_size = inference_config.get('batch_size', 32)
    max_length = model_config.get('max_length', 256)
    device = inference_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize energy tracker
    tracker = EnergyTracker(
        project_name=f"{energy_config.get('project_name', 'benchmark')}_{model_key}",
        output_dir=energy_config.get('log_dir', 'results/logs'),
        country_iso_code=energy_config.get('country_iso_code', 'USA'),
        tracking_mode=energy_config.get('tracking_mode', 'process')
    )
    
    # Load model and tokenizer
    print(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Get model size
    model_size_mb = get_model_size_mb(model_dir)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    # Prepare data
    texts = test_df[text_col].tolist()
    labels = test_df[label_col].tolist()
    
    predictions = []
    
    # Start energy tracking
    tracker.start()
    
    # Run inference
    print(f"Running inference on {len(texts)} samples...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get predictions
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_preds)
    
    # Stop energy tracking
    energy_metrics = tracker.stop()
    
    # Calculate per-sample latency
    latency_ms = (energy_metrics['runtime_seconds'] / len(texts)) * 1000
    
    results = {
        'model': model_key,
        'predictions': np.array(predictions),
        'true_labels': np.array(labels),
        'model_size_mb': model_size_mb,
        'latency_ms_per_sample': latency_ms,
        'total_runtime_seconds': energy_metrics['runtime_seconds'],
        'energy_kwh': energy_metrics['energy_kwh'],
        'co2_grams': energy_metrics['co2_grams']
    }
    
    print(f"\nBenchmark complete:")
    print(f"  Model size: {model_size_mb:.2f} MB")
    print(f"  Latency: {latency_ms:.2f} ms/sample")
    print(f"  Energy: {energy_metrics['energy_kwh']:.6f} kWh")
    print(f"  CO2: {energy_metrics['co2_grams']:.4f} g")
    
    return results


def get_model_size_mb(model_dir: str) -> float:
    """
    Calculate model size in MB.
    
    Args:
        model_dir: Directory containing saved model
        
    Returns:
        Model size in MB
    """
    model_path = Path(model_dir)
    total_size = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    return size_mb
