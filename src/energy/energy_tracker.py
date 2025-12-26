"""Energy consumption tracking using CodeCarbon."""

import time
from pathlib import Path
from typing import Dict, Any, Optional
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
import pandas as pd


class EnergyTracker:
    """
    Wrapper for CodeCarbon energy tracking.
    Tracks energy consumption, CO2 emissions, and runtime.
    """
    
    def __init__(
        self,
        project_name: str,
        output_dir: str,
        country_iso_code: str = "USA",
        tracking_mode: str = "process"
    ):
        """
        Initialize energy tracker.
        
        Args:
            project_name: Name of the project/experiment
            output_dir: Directory to save energy logs
            country_iso_code: ISO code for carbon intensity
            tracking_mode: 'process' or 'machine'
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracker = OfflineEmissionsTracker(
            project_name=project_name,
            output_dir=str(self.output_dir),
            country_iso_code=country_iso_code,
            tracking_mode=tracking_mode,
            log_level="warning"
        )
        
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """Start tracking energy consumption."""
        self.start_time = time.time()
        self.tracker.start()
        print(f"Started energy tracking for {self.project_name}")
    
    def stop(self) -> Dict[str, float]:
        """
        Stop tracking and return metrics.
        
        Returns:
            Dictionary with energy metrics
        """
        self.end_time = time.time()
        emissions = self.tracker.stop()
        
        # Calculate runtime safely
        if self.start_time is not None and self.end_time is not None:
            runtime = self.end_time - self.start_time
        else:
            runtime = 0.0
        
        # Handle None emissions from tracker
        if emissions is None:
            emissions = 0.0
        
        metrics = {
            'runtime_seconds': runtime,
            'energy_kwh': emissions,
            'co2_grams': emissions * 1000  # Convert to grams
        }
        
        print(f"Energy tracking complete:")
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  Energy: {metrics['energy_kwh']:.6f} kWh")
        print(f"  CO2: {metrics['co2_grams']:.4f} g")
        
        return metrics


def track_inference_energy(
    model_name: str,
    inference_function,
    output_dir: str,
    project_name: str = "phishing_inference",
    **kwargs
) -> Dict[str, Any]:
    """
    Track energy consumption during inference.
    
    Args:
        model_name: Name of the model being benchmarked
        inference_function: Function to run inference (must return predictions)
        output_dir: Directory to save energy logs
        project_name: Project name for tracking
        **kwargs: Arguments to pass to inference_function
        
    Returns:
        Dictionary with predictions and energy metrics
    """
    tracker = EnergyTracker(
        project_name=f"{project_name}_{model_name}",
        output_dir=output_dir
    )
    
    # Track energy
    tracker.start()
    predictions = inference_function(**kwargs)
    energy_metrics = tracker.stop()
    
    # Add model name to metrics
    result = {
        'predictions': predictions,
        'energy_metrics': {**energy_metrics, 'model': model_name}
    }
    
    return result


def save_energy_log(
    energy_metrics: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save energy metrics to CSV.
    
    Args:
        energy_metrics: Dictionary with energy metrics
        output_path: Path to save CSV
    """
    df = pd.DataFrame([energy_metrics])
    
    # Append if file exists
    if Path(output_path).exists():
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved energy log to {output_path}")


def load_energy_logs(log_dir: str) -> pd.DataFrame:
    """
    Load all energy logs from directory.
    
    Args:
        log_dir: Directory containing energy logs
        
    Returns:
        Combined DataFrame of all energy logs
    """
    log_path = Path(log_dir)
    csv_files = list(log_path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No energy logs found in {log_dir}")
    
    dfs = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df
