"""Configuration module for loading and managing experiment settings."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "src/config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Dictionary containing all configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        config: Full configuration dictionary
        model_name: Name of model (bert_large, distilbert, phi3_mini)
        
    Returns:
        Model-specific configuration
    """
    return config['models'][model_name]
