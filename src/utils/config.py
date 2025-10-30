"""Configuration utilities"""

import yaml
import json
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    config_path = Path(config_path)
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
    return config


def save_config(config, save_path):
    """
    Save configuration to file.
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save the configuration
    """
    save_path = Path(save_path)
    
    if save_path.suffix == '.yaml' or save_path.suffix == '.yml':
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif save_path.suffix == '.json':
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        raise ValueError(f"Unsupported config file format: {save_path.suffix}")


