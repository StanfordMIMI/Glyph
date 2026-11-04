"""
Device management (GPU/CPU detection and assignment).
"""

import torch


def get_device(device: str = "auto") -> torch.device:
    """Get PyTorch device based on availability and preference.

    Args:
        device: Device preference ('auto', 'cuda', or 'cpu')
                - 'auto': Use CUDA if available, otherwise CPU
                - 'cuda': Force CUDA (will error if not available)
                - 'cpu': Force CPU

    Returns:
        torch.device object

    Raises:
        RuntimeError: If 'cuda' is requested but not available

    Examples:
        >>> device = get_device('auto')
        >>> print(device)
        cuda:0  # or cpu if CUDA not available

        >>> device = get_device('cpu')
        >>> print(device)
        cpu
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Invalid device: {device}. Must be 'auto', 'cuda', or 'cpu'")


def get_device_info() -> dict:
    """Get information about available devices.

    Returns:
        Dictionary with device information:
            - cuda_available: Whether CUDA is available
            - cuda_version: CUDA version if available
            - device_count: Number of CUDA devices
            - device_name: Name of current CUDA device (if available)
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)

    return info


def set_seed(seed: int, device: torch.device):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed
        device: Torch device (used to determine if CUDA seeding needed)
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
