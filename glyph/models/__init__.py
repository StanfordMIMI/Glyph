"""Model architectures and components."""

from glyph.models.base import BaseEmbeddingModel
from glyph.models.adapters import EmbeddingAdapter
from glyph.models.registry import get_model, list_available_models
from glyph.models.embedding_extractor import EmbeddingExtractor

# from glyph.models.sae import SparseAutoencoder

__all__ = [
    "BaseEmbeddingModel",
    "EmbeddingAdapter",
    "get_model",
    "list_available_models",
    "EmbeddingExtractor",
    # "SparseAutoencoder",
]
