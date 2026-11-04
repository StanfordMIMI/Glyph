"""
Embedding extraction using PyTorch Lightning.

Handles extraction of embeddings from raw data using configured models,
with caching to avoid recomputation.
"""

import torch
import hashlib
import json
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, Optional, List, Any
from tqdm import tqdm

from omegaconf import OmegaConf
from glyph.models.registry import get_model
from glyph.models.base import BaseEmbeddingModel


class EmbeddingExtractor(pl.LightningModule):
    """
    Extract and cache embeddings from datasets using configured models.

    This class handles:
    1. Loading appropriate models for each data modality
    2. Extracting embeddings for all samples in a dataset split
    3. Caching embeddings to disk to avoid recomputation
    4. Automatic GPU/CPU device handling via PyTorch Lightning

    Example:
        >>> from glyph.config import load_config
        >>> cfg = load_config(overrides=['data.metadata=data.csv'])
        >>> embedding_map = {'image_path': 'resnet18'}
        >>> extractor = EmbeddingExtractor(cfg, embedding_map)
        >>> embeddings = extractor.extract_embeddings(train_loader, split='train')
    """

    def __init__(
        self,
        cfg,
        embedding_map: Dict[str, str],
        cache_embeddings: bool = True
    ):
        """
        Initialize the embedding extractor.

        Args:
            cfg: Hydra config with data settings
            embedding_map: Maps data column names to model names
                          e.g., {'image_path': 'resnet18'}
            cache_embeddings: Whether to cache embeddings to disk (default: True)
        """
        super().__init__()
        self.cfg = cfg
        self.embedding_map = embedding_map
        self.cache_embeddings = cache_embeddings

        # Set up cache directory
        self.cache_dir = Path(cfg.data.cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load models for each modality
        self.models = self._load_models()

    def _load_models(self) -> Dict[str, BaseEmbeddingModel]:
        """
        Load all models specified in embedding_map.

        Returns:
            Dictionary mapping column names to model instances
        """
        models = {}
        device = self._get_device()

        model_kwargs_map = {}
        if hasattr(self.cfg.data, 'embedding_model_kwargs') and self.cfg.data.embedding_model_kwargs:
            model_kwargs_map = OmegaConf.to_container(self.cfg.data.embedding_model_kwargs, resolve=True)

        for modality_col, model_name in self.embedding_map.items():
            print(f"Loading model '{model_name}' for column '{modality_col}'...")
            kwargs = dict(model_kwargs_map.get(modality_col, {}))
            model = get_model(model_name, device=device, **kwargs)
            models[modality_col] = model

        return models

    def _get_device(self) -> str:
        """
        Determine device from config.

        Returns:
            Device string ('cpu', 'cuda', etc.)
        """
        if self.cfg.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.cfg.device

    def _get_cache_path(self, split: str) -> Path:
        """
        Get cache file path for a specific split.

        Args:
            split: Dataset split ('train', 'val', 'test')

        Returns:
            Path to cache file
        """
        payload = {
            "schema": 2, "split": split, "models": dict(self.embedding_map),
            "dataset_context": getattr(self, "_dataset_context", {}).get(split),
            "seed": int(self.cfg.seed), "split_ratios": list(self.cfg.data.split_ratios),
            "model_kwargs": OmegaConf.to_container(self.cfg.data.embedding_model_kwargs, resolve=True)
                if self.cfg.data.get("embedding_model_kwargs") else {},
        }
        metadata_path = self.cfg.data.get("metadata")
        import inspect
        import importlib.metadata
        payload["runtime"] = {name: importlib.metadata.version(name) for name in ["torch", "torchvision"]}
        payload["implementation"] = {}
        for name, model in self.models.items():
            source = inspect.getsourcefile(type(model))
            if source and Path(source).is_file():
                payload["implementation"][name] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            config = getattr(getattr(model, "model", None), "config", None)
            if config is not None and getattr(config, "_commit_hash", None):
                payload.setdefault("encoder_revisions", {})[name] = config._commit_hash
        files = []
        if metadata_path and Path(metadata_path).is_file():
            import pandas as pd
            source = Path(metadata_path)
            payload["metadata_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            frame = pd.read_csv(source)
            for col in self.embedding_map:
                if col in frame:
                    for value in frame[col].dropna().astype(str).unique():
                        path = Path(value)
                        if path.is_file():
                            stat = path.stat()
                            files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        def assets(value):
            if isinstance(value, dict):
                for child in value.values(): assets(child)
            elif isinstance(value, str) and Path(value).is_file():
                path = Path(value)
                payload.setdefault("asset_sha256", {})[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        assets(payload["model_kwargs"])
        payload["raw_files"] = sorted(files)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return self.cache_dir / f"{split}_{digest}.pt"

    def _load_from_cache(self, split: str) -> Optional[Dict[str, Any]]:
        """
        Try to load embeddings from cache.

        Args:
            split: Dataset split

        Returns:
            Cached embeddings dict or None if not found
        """
        if not self.cache_embeddings:
            return None

        cache_path = self._get_cache_path(split)
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        return None

    def _save_to_cache(self, embeddings: Dict[str, Any], split: str):
        """
        Save embeddings to cache.

        Args:
            embeddings: Embeddings dictionary
            split: Dataset split
        """
        if not self.cache_embeddings:
            return

        cache_path = self._get_cache_path(split)
        print(f"Saving embeddings to cache: {cache_path}")
        torch.save(embeddings, cache_path)

    def extract_embeddings(
        self,
        dataloader,
        split: str = 'train',
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Extract embeddings for all samples in a dataloader.

        Args:
            dataloader: PyTorch DataLoader
            split: Dataset split name ('train', 'val', 'test')
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with:
                - 'sample_ids': List of sample IDs
                - 'embeddings': Dict mapping modality to embedding tensors
                  e.g., {'image_path': tensor of shape (N, 512)}

        Example:
            >>> result = extractor.extract_embeddings(train_loader, split='train')
            >>> print(result['sample_ids'])  # [0, 1, 2, ...]
            >>> print(result['embeddings']['image_path'].shape)  # (N, 512)
        """
        metadata = getattr(getattr(dataloader, 'dataset', None), 'metadata', None)
        if metadata is not None:
            if not hasattr(self, '_dataset_context'):
                self._dataset_context = {}
            self._dataset_context[split] = hashlib.sha256(metadata.to_json(orient='split', date_format='iso').encode()).hexdigest()
        # Check cache first
        cached = self._load_from_cache(split)
        if cached is not None:
            return cached

        # Extract embeddings
        all_sample_ids = []
        all_embeddings = {col: [] for col in self.embedding_map.keys()}

        iterator = tqdm(dataloader, desc=f"Extracting {split} embeddings") if show_progress else dataloader

        for batch in iterator:
            # Get sample IDs
            sample_ids = batch['sample_id']
            if isinstance(sample_ids, torch.Tensor):
                sample_ids = sample_ids.tolist()
            all_sample_ids.extend(sample_ids)

            # Extract embeddings for each modality
            batch_embeddings = self._embed_batch(batch)
            for col, emb in batch_embeddings.items():
                all_embeddings[col].append(emb)

        # Concatenate all embeddings
        result = {
            'sample_ids': all_sample_ids,
            'embeddings': {
                col: torch.cat(emb_list, dim=0)
                for col, emb_list in all_embeddings.items()
            }
        }

        # Save to cache
        self._save_to_cache(result, split)

        return result

    def _embed_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings for a single batch.

        Args:
            batch: Batch dictionary from dataloader

        Returns:
            Dictionary mapping modality to embeddings tensor
        """
        embeddings = {}

        for modality_col, model in self.models.items():
            # Get data from batch
            data = batch['inputs'][modality_col]

            # Handle tensor batch (if data is already in tensor form)
            if isinstance(data, torch.Tensor):
                data = data.tolist()

            # Convert single item to list
            if not isinstance(data, list):
                data = [data]

            # Extract embeddings
            emb = model.embed_batch(data)  # (batch_size, embedding_dim)
            emb = emb.detach().cpu()
            if hasattr(emb, "as_tensor"):
                emb = emb.as_tensor()
            embeddings[modality_col] = emb

        return embeddings

    def extract_all_splits(
        self,
        dataloaders: Dict[str, Any],
        show_progress: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract embeddings for all splits (train/val/test).

        Args:
            dataloaders: Dictionary mapping split names to DataLoaders
            show_progress: Whether to show progress bars

        Returns:
            Dictionary mapping split names to embedding dictionaries

        Example:
            >>> from glyph.dataloader import get_dataloaders
            >>> dataloaders = get_dataloaders(cfg)
            >>> all_embeddings = extractor.extract_all_splits(dataloaders)
            >>> print(all_embeddings.keys())  # ['train', 'val', 'test']
        """
        results = {}
        for split, dataloader in dataloaders.items():
            print(f"\n{'='*50}")
            print(f"Processing {split} split")
            print(f"{'='*50}")
            results[split] = self.extract_embeddings(
                dataloader,
                split=split,
                show_progress=show_progress
            )
        return results
