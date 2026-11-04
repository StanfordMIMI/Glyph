"""
I/O utilities for saving and loading data.
"""

import pickle
from pathlib import Path
from typing import Any


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object for the directory

    Examples:
        >>> ensure_dir('./outputs/results')
        PosixPath('./outputs/results')
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_pickle(obj: Any, path: str):
    """Save object to pickle file.

    Args:
        obj: Object to save
        path: Path to save pickle file

    Examples:
        >>> data = {'embeddings': np.array([1, 2, 3])}
        >>> save_pickle(data, './cache/embeddings.pkl')
    """
    path_obj = Path(path)
    # Ensure parent directory exists
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    """Load object from pickle file.

    Args:
        path: Path to pickle file

    Returns:
        Loaded object

    Raises:
        FileNotFoundError: If pickle file doesn't exist

    Examples:
        >>> data = load_pickle('./cache/embeddings.pkl')
        >>> print(data['embeddings'])
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with open(path_obj, 'rb') as f:
        return pickle.load(f)


def file_exists(path: str) -> bool:
    """Check if file exists.

    Args:
        path: Path to check

    Returns:
        True if file exists, False otherwise
    """
    return Path(path).exists()


def get_cache_path(cache_dir: str, filename: str) -> str:
    """Get path for cached file.

    Args:
        cache_dir: Cache directory
        filename: Name of cached file

    Returns:
        Full path to cached file

    Examples:
        >>> path = get_cache_path('./cache', 'embeddings.pkl')
        >>> print(path)
        ./cache/embeddings.pkl
    """
    return str(Path(cache_dir) / filename)
