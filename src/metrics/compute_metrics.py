"""Performance metrics computation."""

import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary')
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "") -> None:
    """
    Print metrics in formatted table.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Optional model name for display
    """
    if model_name:
        print(f"\nMetrics for {model_name}:")
    else:
        print("\nMetrics:")
    
    print("-" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("-" * 40)
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"True Negatives:  {metrics['true_negatives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print("-" * 40)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, target_names=['Legitimate', 'Phishing'])


def compute_metrics_summary(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    model_name: str,
    model_size_mb: float,
    latency_ms: float,
    energy_kwh: float,
    co2_grams: float
) -> Dict[str, float]:
    """
    Compute comprehensive metrics summary for paper.
    
    Args:
        predictions: Model predictions
        true_labels: True labels
        model_name: Model identifier
        model_size_mb: Model size in MB
        latency_ms: Latency per sample in ms
        energy_kwh: Energy consumption in kWh
        co2_grams: CO2 emissions in grams
        
    Returns:
        Complete metrics dictionary
    """
    # Performance metrics
    perf_metrics = compute_all_metrics(true_labels, predictions)
    
    # Combine with efficiency metrics (include confusion counts for printing)
    full_metrics = {
        'model': model_name,
        'accuracy': perf_metrics['accuracy'],
        'precision': perf_metrics['precision'],
        'recall': perf_metrics['recall'],
        'f1_score': perf_metrics['f1_score'],
        'true_positives': perf_metrics.get('true_positives'),
        'true_negatives': perf_metrics.get('true_negatives'),
        'false_positives': perf_metrics.get('false_positives'),
        'false_negatives': perf_metrics.get('false_negatives'),
        'model_size_mb': model_size_mb,
        'latency_ms_per_sample': latency_ms,
        'energy_kwh': energy_kwh,
        'co2_grams': co2_grams
    }
    
    return full_metrics
