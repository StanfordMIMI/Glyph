"""
Unit tests for Sparse Autoencoder (SAE) implementation.
"""

import pytest
import torch
import numpy as np

from glyph.mechinterp.sae import TopKSAE, SAE
from glyph.mechinterp.sae_trainer import train_sae, extract_sae_features
from glyph.config import load_config


class TestTopKSAE:
    """Test cases for TopKSAE model."""

    def test_initialization(self):
        """Test SAE initializes correctly."""
        sae = TopKSAE(input_dim=512, dict_size=2048, top_k=20)
        
        assert sae.input_dim == 512
        assert sae.dict_size == 2048
        assert sae.top_k == 20
        assert sae.W_enc.shape == (512, 2048)
        assert sae.W_dec.shape == (2048, 512)
        assert sae.b_enc.shape == (2048,)
        assert sae.b_dec.shape == (512,)

    def test_forward_output_shapes(self):
        """Test forward pass returns correct shapes."""
        sae = TopKSAE(input_dim=512, dict_size=2048, top_k=20)
        x = torch.randn(32, 512)
        
        outputs = sae(x)
        
        assert outputs['x'].shape == (32, 512)
        assert outputs['z'].shape == (32, 2048)
        assert outputs['z_sparse'].shape == (32, 2048)
        assert outputs['x_hat'].shape == (32, 512)

    def test_topk_sparsity(self):
        """Test that TopK produces exactly k non-zero activations."""
        sae = TopKSAE(input_dim=512, dict_size=2048, top_k=20)
        x = torch.randn(32, 512)
        
        outputs = sae(x)
        z_sparse = outputs['z_sparse']
        
        # Count non-zeros per sample
        non_zeros_per_sample = (z_sparse != 0).sum(dim=1)
        
        # Each sample should have exactly top_k non-zero values
        assert (non_zeros_per_sample == 20).all()

    def test_embed_method(self):
        """Test embed method returns sparse features."""
        sae = TopKSAE(input_dim=512, dict_size=2048, top_k=20)
        x = torch.randn(32, 512)
        
        features = sae.embed(x)
        
        assert features.shape == (32, 2048)
        non_zeros = (features != 0).sum(dim=1)
        assert (non_zeros == 20).all()

    def test_decoder_normalization(self):
        """Test decoder weight normalization."""
        sae = TopKSAE(input_dim=512, dict_size=2048, top_k=20)
        
        # Initially normalized
        norms = torch.norm(sae.W_dec, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        
        # Mess up the weights
        with torch.no_grad():
            sae.W_dec.data *= 2.0
        
        # Normalize again
        sae.normalize_decoder()
        
        norms = torch.norm(sae.W_dec, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_reconstruct_with_prefix(self):
        """Test Matryoshka-style prefix reconstruction."""
        sae = TopKSAE(input_dim=512, dict_size=2048, top_k=20)
        x = torch.randn(32, 512)
        
        outputs = sae(x)
        z_sparse = outputs['z_sparse']
        
        # Reconstruct with different prefix sizes
        x_hat_256 = sae.reconstruct_with_prefix(z_sparse, prefix_size=256)
        x_hat_512 = sae.reconstruct_with_prefix(z_sparse, prefix_size=512)
        x_hat_full = sae.reconstruct_with_prefix(z_sparse, prefix_size=2048)
        
        assert x_hat_256.shape == (32, 512)
        assert x_hat_512.shape == (32, 512)
        assert x_hat_full.shape == (32, 512)
        
        # Full prefix should match normal reconstruction
        assert torch.allclose(x_hat_full, outputs['x_hat'], atol=1e-5)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        sae = TopKSAE(input_dim=512, dict_size=2048, top_k=20)
        x = torch.randn(32, 512, requires_grad=True)
        
        outputs = sae(x)
        loss = ((outputs['x_hat'] - x) ** 2).mean()
        loss.backward()
        
        assert sae.W_enc.grad is not None
        assert sae.W_dec.grad is not None
        assert sae.b_enc.grad is not None
        assert sae.b_dec.grad is not None


class TestSAETrainer:
    """Test cases for SAE training functions."""

    def test_train_sae_runs(self):
        """Test that train_sae runs without errors on synthetic data."""
        # Create synthetic embeddings
        np.random.seed(42)
        torch.manual_seed(42)
        
        train_emb = torch.randn(200, 64)
        val_emb = torch.randn(50, 64)
        
        # Create minimal config
        cfg = load_config(overrides=[
            "sae.learning_rate=0.001",
            "sae.max_steps=100",
            "sae.sparsity_coefficient=0.001",
            "sae.use_ghost_grads=false",
            "sae.dead_feature_threshold=1e-8",
            "sae.aux_scale=1.0",
            "sae.kaux=32",
        ])
        
        model, alive_mask = train_sae(
            train_embeddings=train_emb,
            val_embeddings=val_emb,
            input_dim=64,
            dict_size=128,
            top_k=10,
            cfg=cfg,
            device='cpu',
        )
        
        assert isinstance(model, TopKSAE)
        assert alive_mask.shape == (128,)
        assert alive_mask.dtype == torch.bool

    def test_extract_sae_features(self):
        """Test feature extraction from trained SAE."""
        sae = TopKSAE(input_dim=64, dict_size=128, top_k=10)
        embeddings = torch.randn(100, 64)
        
        features = extract_sae_features(sae, embeddings, batch_size=32)
        
        assert features.shape == (100, 128)
        # Check sparsity
        non_zeros = (features != 0).sum(dim=1)
        assert (non_zeros == 10).all()


class TestSAEWrapper:
    """Test cases for high-level SAE wrapper."""

    def test_sae_wrapper_initialization(self):
        """Test SAE wrapper initializes correctly."""
        cfg = load_config(overrides=[
            "sae.top_ks=[10,20]",
            "sae.matryoshka=[128]",
        ])
        
        sae = SAE(cfg, input_dim=64, device='cpu')
        
        assert sae.input_dim == 64
        assert sae.device == 'cpu'
        assert len(sae.models) == 0  # Not trained yet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

