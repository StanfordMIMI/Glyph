"""
Precomputed embedding model.

Looks up embeddings from a parquet file by sample key instead of running inference.
Useful when embeddings are already computed externally (e.g. VISTA CT/EHR embeddings).
"""

import torch
import pyarrow.parquet as pq
from typing import Any, List
from glyph.models.base import BaseEmbeddingModel


class PrecomputedModel(BaseEmbeddingModel):
    """
    Returns precomputed embeddings looked up by sample key.

    Args:
        parquet_path: Path to parquet file containing embeddings.
        key_col: Column name used to look up samples (e.g. 'accession_number').
        embedding_col: Column name containing the embedding vector (default: 'embedding').
        device: Torch device string (unused for lookup, kept for API compatibility).
    """

    def __init__(self, parquet_path: str, key_col: str, embedding_col: str = 'embedding', device: str = 'cpu'):
        table = pq.read_table(parquet_path)
        keys = [str(v.as_py()) for v in table.column(key_col)]
        vecs = [torch.tensor(v.as_py(), dtype=torch.float32) for v in table.column(embedding_col)]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Precomputed embedding keys must be nonempty and unique")
        if any(v.ndim != 1 or v.shape != vecs[0].shape or not torch.isfinite(v).all() for v in vecs):
            raise ValueError("Precomputed embeddings must be finite vectors of one consistent dimension")
        self._lookup = dict(zip(keys, vecs))
        self._dim = vecs[0].shape[0]
        self.device = device

    def embed(self, key: Any) -> torch.Tensor:
        return self._lookup[str(key)]

    def embed_batch(self, keys: List[Any]) -> torch.Tensor:
        return torch.stack([self._lookup[str(k)] for k in keys])

    def get_embedding_dim(self) -> int:
        return self._dim
