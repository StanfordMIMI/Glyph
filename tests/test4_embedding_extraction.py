"""
Unit tests for embedding extraction functionality.

Tests base models, model registry, and embedding extraction with real pneumonia data.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from glyph.config import load_config
from glyph.models.base import BaseEmbeddingModel
from glyph.models.image.resnet18 import ResNet18
from glyph.models.registry import get_model, list_available_models
from glyph.models.embedding_extractor import EmbeddingExtractor
from glyph.dataloader import get_dataloader, get_dataloaders


# Path to real pneumonia data
PNEUMONIA_DATA_PATH = "/dataNAS/people/rholland/pneumonia_data"
METADATA_PATH = f"{PNEUMONIA_DATA_PATH}/metadata.csv"
TEST_IMAGE_PATH = f"{PNEUMONIA_DATA_PATH}/images/0.png"


class TestBaseEmbeddingModel:
    """Test cases for base embedding model interface."""

    def test_resnet18_inherits_from_base(self):
        """Test that ResNet18 inherits from BaseEmbeddingModel."""
        model = ResNet18(device='cpu')
        assert isinstance(model, BaseEmbeddingModel)

    def test_resnet18_implements_embed(self):
        """Test that ResNet18 implements embed method."""
        model = ResNet18(device='cpu')
        assert hasattr(model, 'embed')
        assert callable(model.embed)

    def test_resnet18_implements_embed_batch(self):
        """Test that ResNet18 implements embed_batch method."""
        model = ResNet18(device='cpu')
        assert hasattr(model, 'embed_batch')
        assert callable(model.embed_batch)

    def test_resnet18_implements_get_embedding_dim(self):
        """Test that ResNet18 implements get_embedding_dim method."""
        model = ResNet18(device='cpu')
        assert hasattr(model, 'get_embedding_dim')
        assert callable(model.get_embedding_dim)
        assert model.get_embedding_dim() == 512


class TestResNet18Model:
    """Test cases for ResNet18 model functionality."""

    def test_resnet18_single_image_embed(self):
        """Test embedding a single image from path."""
        model = ResNet18(device='cpu')

        # Load and embed a real image
        embedding = model.embed(TEST_IMAGE_PATH)

        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (512,)
        assert not torch.isnan(embedding).any()
        assert not torch.isinf(embedding).any()

    def test_resnet18_pil_image_embed(self):
        """Test embedding a PIL Image object."""
        model = ResNet18(device='cpu')

        # Load image as PIL
        pil_image = Image.open(TEST_IMAGE_PATH)
        embedding = model.embed(pil_image)

        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (512,)

    def test_resnet18_batch_embed(self):
        """Test embedding a batch of images."""
        model = ResNet18(device='cpu')

        # Create batch of 3 images (same image repeated)
        image_paths = [TEST_IMAGE_PATH] * 3
        embeddings = model.embed_batch(image_paths)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (3, 512)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_resnet18_batch_with_pil_images(self):
        """Test embedding a batch of PIL Images."""
        model = ResNet18(device='cpu')

        # Load images as PIL
        pil_images = [Image.open(TEST_IMAGE_PATH) for _ in range(3)]
        embeddings = model.embed_batch(pil_images)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (3, 512)


class TestModelRegistry:
    """Test cases for model registry functionality."""

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()

        assert isinstance(models, dict)
        assert 'resnet18' in models
        assert 'module' in models['resnet18']
        assert 'class' in models['resnet18']

    def test_get_model_resnet18(self):
        """Test loading ResNet18 from registry."""
        model = get_model('resnet18', device='cpu')

        assert isinstance(model, ResNet18)
        assert isinstance(model, BaseEmbeddingModel)

    def test_get_model_invalid_name_raises_error(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="not found in registry"):
            get_model('nonexistent_model', device='cpu')

    def test_get_model_returns_base_embedding_model(self):
        """Test that all models from registry inherit from BaseEmbeddingModel."""
        model = get_model('resnet18', device='cpu')
        assert isinstance(model, BaseEmbeddingModel)


class TestEmbeddingExtractor:
    """Test cases for EmbeddingExtractor class."""

    def test_extractor_initialization(self):
        """Test that EmbeddingExtractor initializes correctly."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.embedding_map={image_path: resnet18}",
            "seed=42"
        ])

        embedding_map = {'image_path': 'resnet18'}
        extractor = EmbeddingExtractor(cfg, embedding_map)

        assert extractor is not None
        assert extractor.embedding_map == embedding_map
        assert 'image_path' in extractor.models
        assert isinstance(extractor.models['image_path'], ResNet18)

    def test_extractor_cache_directory_created(self):
        """Test that cache directory is created."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.embedding_map={image_path: resnet18}",
            "data.cache_dir=./test_cache",
            "seed=42"
        ])

        embedding_map = {'image_path': 'resnet18'}
        extractor = EmbeddingExtractor(cfg, embedding_map)

        assert extractor.cache_dir.exists()
        assert extractor.cache_dir == Path("./test_cache/embeddings")

        # Cleanup
        extractor.cache_dir.rmdir()
        extractor.cache_dir.parent.rmdir()

    def test_extract_embeddings_single_split(self):
        """Test extracting embeddings for a single split."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.batch_size=16",
            "data.num_workers=0",
            "data.embedding_map={image_path: resnet18}",
            "seed=42"
        ])

        # Create dataloader
        dataloader = get_dataloader(cfg, split='train')

        # Create extractor
        embedding_map = {'image_path': 'resnet18'}
        extractor = EmbeddingExtractor(cfg, embedding_map, cache_embeddings=False)

        # Extract embeddings
        result = extractor.extract_embeddings(dataloader, split='train', show_progress=False)

        # Check result structure
        assert 'sample_ids' in result
        assert 'embeddings' in result
        assert isinstance(result['sample_ids'], list)
        assert isinstance(result['embeddings'], dict)

        # Check embeddings
        assert 'image_path' in result['embeddings']
        embeddings = result['embeddings']['image_path']
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[1] == 512  # ResNet18 dimension
        assert embeddings.shape[0] == len(result['sample_ids'])
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_extract_embeddings_correct_number_samples(self):
        """Test that correct number of samples are extracted."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.batch_size=16",
            "data.num_workers=0",
            "data.split_ratios=[0.7,0.15,0.15]",
            "data.embedding_map={image_path: resnet18}",
            "seed=42"
        ])

        dataloader = get_dataloader(cfg, split='train')

        embedding_map = {'image_path': 'resnet18'}
        extractor = EmbeddingExtractor(cfg, embedding_map, cache_embeddings=False)

        result = extractor.extract_embeddings(dataloader, split='train', show_progress=False)

        # Should have ~70% of ~4708 samples = ~3295 samples
        assert 3200 < len(result['sample_ids']) < 3400

    def test_extract_all_splits(self):
        """Test extracting embeddings for all splits."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.batch_size=16",
            "data.num_workers=0",
            "data.embedding_map={image_path: resnet18}",
            "seed=42"
        ])

        # Create dataloaders
        dataloaders = get_dataloaders(cfg)

        # Create extractor
        embedding_map = {'image_path': 'resnet18'}
        extractor = EmbeddingExtractor(cfg, embedding_map, cache_embeddings=False)

        # Extract all splits
        results = extractor.extract_all_splits(dataloaders, show_progress=False)

        # Check all splits present
        assert 'train' in results
        assert 'val' in results
        assert 'test' in results

        # Check each split has correct structure
        for split, result in results.items():
            assert 'sample_ids' in result
            assert 'embeddings' in result
            assert 'image_path' in result['embeddings']
            assert len(result['sample_ids']) > 0

    def test_caching_works(self):
        """Test that caching saves and loads embeddings correctly."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.batch_size=16",
            "data.num_workers=0",
            "data.cache_dir=./test_cache_embeddings",
            "data.embedding_map={image_path: resnet18}",
            "seed=999"
        ])

        dataloader = get_dataloader(cfg, split='train')

        embedding_map = {'image_path': 'resnet18'}
        extractor = EmbeddingExtractor(cfg, embedding_map, cache_embeddings=True)

        # First extraction (should compute)
        result1 = extractor.extract_embeddings(dataloader, split='train', show_progress=False)

        # Check cache file exists
        cache_path = extractor._get_cache_path('train')
        assert cache_path.exists()

        # Second extraction (should load from cache)
        extractor2 = EmbeddingExtractor(cfg, embedding_map, cache_embeddings=True)
        result2 = extractor2.extract_embeddings(dataloader, split='train', show_progress=False)

        # Results should be identical
        assert result1['sample_ids'] == result2['sample_ids']
        assert torch.allclose(
            result1['embeddings']['image_path'],
            result2['embeddings']['image_path']
        )

        # Cleanup
        cache_path.unlink()
        cache_path.parent.rmdir()
        cache_path.parent.parent.rmdir()

    def test_embedding_dimensions_match_model(self):
        """Test that embedding dimensions match model specification."""
        cfg = load_config(overrides=[
            f"data.metadata={METADATA_PATH}",
            "data.input_cols=[image_path]",
            "data.outcome_cols=[outcome]",
            "data.batch_size=8",
            "data.num_workers=0",
            "data.embedding_map={image_path: resnet18}",
            "seed=42"
        ])

        dataloader = get_dataloader(cfg, split='train')

        embedding_map = {'image_path': 'resnet18'}
        extractor = EmbeddingExtractor(cfg, embedding_map, cache_embeddings=False)

        result = extractor.extract_embeddings(dataloader, split='train', show_progress=False)

        # Check dimension matches ResNet18
        model = extractor.models['image_path']
        expected_dim = model.get_embedding_dim()
        actual_dim = result['embeddings']['image_path'].shape[1]

        assert actual_dim == expected_dim
        assert actual_dim == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
