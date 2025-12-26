"""Energy tracking module initialization."""

from .energy_tracker import (
    EnergyTracker,
    track_inference_energy,
    save_energy_log,
    load_energy_logs
)

__all__ = [
    'EnergyTracker',
    'track_inference_energy',
    'save_energy_log',
    'load_energy_logs'
]
