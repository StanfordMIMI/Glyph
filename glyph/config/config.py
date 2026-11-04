"""
Hydra-based configuration management for Glyph.

Simple wrapper around Hydra for configuration loading.
No dataclasses needed - just initialize Hydra and access cfg as a dict-like object.
"""

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional, List


def get_config_path() -> Path:
    """Get the path to the packaged config directory.

    Returns:
        Path to the YAML config directory shipped inside the glyph package
    """
    return Path(__file__).parent.parent / "conf"


def load_config(config_name: str = "config", overrides: Optional[List[str]] = None) -> DictConfig:
    """
    Load configuration using Hydra's compose API.

    Args:
        config_name: Name of config file (without .yaml extension)
        overrides: List of config overrides (e.g., ["sae=custom", "seed=123"])

    Returns:
        DictConfig object (dict-like, supports dot notation and dict access)

    Examples:
        >>> # Load default config
        >>> cfg = load_config()
        >>> print(cfg.seed)
        42

        >>> # Load with overrides
        >>> cfg = load_config(overrides=["seed=999", "logging=wandb"])
        >>> print(cfg.seed)
        999
        >>> print(cfg.logging.use_wandb)
        True

        >>> # Access nested values
        >>> print(cfg.data.cache_dir)
        ./cache
        >>> print(cfg.metrics.threshold)
        0.5
    """
    from hydra import compose, initialize_config_dir

    config_dir = str(get_config_path())

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    return cfg


def print_config(cfg: DictConfig, resolve: bool = True):
    """Pretty print configuration.

    Args:
        cfg: Configuration to print
        resolve: Whether to resolve interpolations (default: True)
    """
    print(OmegaConf.to_yaml(cfg, resolve=resolve))


def save_config(cfg: DictConfig, output_dir: str, filename: str = "config.yaml"):
    """Save configuration to YAML file for reproducibility.

    Args:
        cfg: Configuration to save
        output_dir: Directory to save config file
        filename: Name of output file (default: config.yaml)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / filename
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f, resolve=True)

    print(f"Configuration saved to: {config_path}")
