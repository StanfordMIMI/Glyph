"""
Model registry for dynamic model loading.

Maps model names to their implementations and provides utilities
for loading models by name.
"""

import importlib
from typing import Dict, Any
from glyph.models.base import BaseEmbeddingModel


# Registry mapping model names to their module paths and class names
MODEL_REGISTRY = {
    'hf': {'module': 'glyph.models.huggingface', 'class': 'HFEmbeddingModel'},
    'dinov3': {'module': 'glyph.models.huggingface', 'class': 'DINOv3'},
    'qwen3-embedding-0.6b': {'module': 'glyph.models.huggingface', 'class': 'Qwen3Embedding06B'},
    'merlin': {'module': 'glyph.extraction.encoders', 'class': 'MerlinEncoder'},
    'qwen3': {'module': 'glyph.extraction.encoders', 'class': 'Qwen3Encoder'},
    'labs': {'module': 'glyph.extraction.labs', 'class': 'LabsEncoder'},
    'resnet18': {
        'module': 'glyph.models.image.resnet18',
        'class': 'ResNet18',
    },
    'siglip_embed': {
        'module': 'glyph.models.image.siglip_embed',
        'class': 'SiglipEmbed',
    },
    'precomputed': {
        'module': 'glyph.models.precomputed',
        'class': 'PrecomputedModel',
    },
    # Future models can be added here:
    # 'vit': {
    #     'module': 'glyph.models.image.vit',
    #     'class': 'VisionTransformer',
    # },
    # 'clip': {
    #     'module': 'glyph.models.image.clip',
    #     'class': 'CLIP',
    # },
}


def get_model(model_name: str, device: str = 'cpu', **kwargs) -> BaseEmbeddingModel:
    """
    Load a model by name from the registry.

    Args:
        model_name: Name of the model (e.g., 'resnet18')
        device: Device to load model on ('cpu', 'cuda', etc.)
        **kwargs: Additional keyword arguments to pass to model constructor

    Returns:
        Instantiated model that inherits from BaseEmbeddingModel

    Raises:
        ValueError: If model_name is not in registry
        ImportError: If model module cannot be imported
        AttributeError: If model class cannot be found in module

    Example:
        >>> model = get_model('resnet18', device='cuda')
        >>> embedding = model.embed(image)
    """
    if model_name.startswith('hf:'):
        from glyph.models.huggingface import HFEmbeddingModel
        if not model_name[3:]:
            raise ValueError('hf: must be followed by a Hugging Face model ID')
        return HFEmbeddingModel(model_id=model_name[3:], device=device, **kwargs)

    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {available}"
        )

    model_info = MODEL_REGISTRY[model_name]
    module_path = model_info['module']
    class_name = model_info['class']

    # Dynamically import module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{module_path}' for model '{model_name}': {e}"
        )

    # Get class from module
    try:
        model_class = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Could not find class '{class_name}' in module '{module_path}': {e}"
        )

    # Instantiate model
    model = model_class(device=device, **kwargs)

    # Verify it's a BaseEmbeddingModel
    if not isinstance(model, BaseEmbeddingModel):
        raise TypeError(
            f"Model '{model_name}' (class {class_name}) must inherit from "
            f"BaseEmbeddingModel, but got {type(model)}"
        )

    return model


def list_available_models() -> Dict[str, Dict[str, str]]:
    """
    List all available models in the registry.

    Returns:
        Dictionary mapping model names to their info (module, class)

    Example:
        >>> models = list_available_models()
        >>> print(models.keys())
        dict_keys(['resnet18'])
    """
    return MODEL_REGISTRY.copy()
