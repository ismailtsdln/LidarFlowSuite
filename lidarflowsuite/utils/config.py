import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config

def save_config(config: Dict[str, Any], save_path: str):
    """Saves a dictionary as a YAML configuration file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
