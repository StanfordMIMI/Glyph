"""
Base classes for embedding models.

All embedding models must inherit from BaseEmbeddingModel and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
import torch
from typing import Any, List, Union


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for all embedding models.

    All models that extract embeddings from data must inherit from this class
    and implement the required methods. This ensures a consistent interface
    across different modalities (images, text, tabular, etc.).
    """

    @abstractmethod
    def embed(self, data: Any) -> torch.Tensor:
        """
        Extract embeddings from input data.

        Args:
            data: Input data (type depends on modality)
                  - For images: PIL.Image or path string
                  - For text: string
                  - For tabular: numpy array or tensor

        Returns:
            torch.Tensor: Embedding vector of shape (embedding_dim,)
        """
        pass

    @abstractmethod
    def embed_batch(self, data_list: List[Any]) -> torch.Tensor:
        """
        Extract embeddings from a batch of input data.

        Args:
            data_list: List of input data items

        Returns:
            torch.Tensor: Batch of embeddings of shape (batch_size, embedding_dim)
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Return the dimensionality of the embedding vectors.

        Returns:
            int: Embedding dimension
        """
        pass
