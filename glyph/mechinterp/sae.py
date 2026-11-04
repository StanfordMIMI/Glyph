"""
Sparse Autoencoder (SAE) with TopK activation.

Learns a sparse, interpretable representation of embeddings.
"""

import math
import warnings
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple


def reconstruction_metrics(model, embeddings, batch_size=256):
    if len(embeddings) == 0:
        raise ValueError('Reconstruction evaluation requires nonempty embeddings')
    device = next(model.parameters()).device
    mean = embeddings.mean(dim=0).to(device)
    squared_error = 0.0
    total_variation = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            x = embeddings[start:start + batch_size].to(device)
            x_hat = model(x)['x_hat']
            squared_error += ((x - x_hat) ** 2).sum(dtype=torch.float64).item()
            total_variation += ((x - mean) ** 2).sum(dtype=torch.float64).item()
    return {
        'mse': squared_error / embeddings.numel(),
        'r2': 1 - squared_error / total_variation if total_variation > 0 else None,
    }


class TopKSAE(nn.Module):
    """
    Sparse Autoencoder with TopK activation constraint.
    
    Architecture:
        z = x @ W_enc + b_enc           # Encode to dictionary size
        z_sparse = topk(z, k)           # Keep only top-k activations
        x_hat = z_sparse @ W_dec + b_dec # Decode back to embedding size
    
    Args:
        input_dim: Dimension of input embeddings
        dict_size: Size of the learned dictionary (number of concepts)
        top_k: Number of active features per sample
        
    Example:
        >>> sae = TopKSAE(input_dim=512, dict_size=2048, top_k=20)
        >>> x = torch.randn(32, 512)
        >>> outputs = sae(x)
        >>> print(outputs['z_sparse'].shape)  # (32, 2048) with only 20 non-zero per row
    """
    
    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        top_k: int = 20,
        init_encoder_as_decoder_transpose: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.top_k = top_k
        
        # Decoder weights (dictionary): dict_size x input_dim
        self.W_dec = nn.Parameter(
            torch.empty(dict_size, input_dim)
        )
        nn.init.kaiming_uniform_(self.W_dec)
        
        # Normalize decoder weights
        with torch.no_grad():
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        
        # Encoder weights: input_dim x dict_size
        if init_encoder_as_decoder_transpose:
            self.W_enc = nn.Parameter(self.W_dec.data.t().clone())
        else:
            self.W_enc = nn.Parameter(torch.empty(input_dim, dict_size))
            nn.init.kaiming_uniform_(self.W_enc)
        
        # Biases
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to dense latent representation."""
        return x @ self.W_enc + self.b_enc
    
    def apply_topk(
        self, 
        z: torch.Tensor, 
        k: Optional[int] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply TopK sparsity constraint.
        
        Args:
            z: Dense latent activations (batch_size, dict_size)
            k: Number of top activations to keep (default: self.top_k)
            mask: Optional mask for ghost gradient computation
            
        Returns:
            Sparse activations with only top-k non-zero values per sample
        """
        if k is None:
            k = self.top_k
        
        if mask is None:
            # Standard TopK on all features
            topk_vals, topk_idx = torch.topk(z, k=min(k, z.shape[1]), dim=1)
            z_sparse = torch.zeros_like(z).scatter_(1, topk_idx, topk_vals)
        else:
            # TopK only on masked features (for ghost gradients)
            z_sub = z[:, mask]
            topk_vals, topk_idx = torch.topk(z_sub, k=min(k, z_sub.shape[1]), dim=1)
            z_sparse_sub = torch.zeros_like(z_sub).scatter_(1, topk_idx, topk_vals)
            z_sparse = torch.zeros_like(z)
            z_sparse[:, mask] = z_sparse_sub
        
        return z_sparse
    
    def decode(self, z_sparse: torch.Tensor) -> torch.Tensor:
        """Decode sparse latent back to input space."""
        return z_sparse @ self.W_dec + self.b_dec
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: Input embeddings (batch_size, input_dim)
            
        Returns:
            Dict with:
                - x: Original input
                - z: Dense latent (before TopK)
                - z_sparse: Sparse latent (after TopK)
                - x_hat: Reconstruction
        """
        z = self.encode(x)
        z_sparse = self.apply_topk(z)
        x_hat = self.decode(z_sparse)
        
        return {
            'x': x,
            'z': z,
            'z_sparse': z_sparse,
            'x_hat': x_hat
        }
    
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract sparse features from input (for downstream use).
        
        Args:
            x: Input embeddings (batch_size, input_dim)
            
        Returns:
            Sparse activations (batch_size, dict_size)
        """
        z = self.encode(x)
        return self.apply_topk(z)
    
    def reconstruct_with_prefix(self, z_sparse: torch.Tensor, prefix_size: int) -> torch.Tensor:
        """
        Reconstruct using only the first prefix_size dictionary elements.
        Used for Matryoshka-style nested evaluation.
        
        Args:
            z_sparse: Sparse latent activations (batch_size, dict_size)
            prefix_size: Number of dictionary elements to use
            
        Returns:
            Reconstruction using only first prefix_size elements
        """
        z_prefix = z_sparse[:, :prefix_size]
        W_dec_prefix = self.W_dec[:prefix_size, :]
        return z_prefix @ W_dec_prefix + self.b_dec
    
    def normalize_decoder(self):
        """Normalize decoder weight rows to unit norm (call during training)."""
        with torch.no_grad():
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
    
    def remove_parallel_gradient_component(self):
        """
        Remove gradient component parallel to decoder weights.
        This prevents the decoder from just scaling up.
        Call after loss.backward() and before optimizer.step().
        """
        if self.W_dec.grad is not None:
            parallel = (self.W_dec.grad * self.W_dec.data).sum(dim=1, keepdim=True)
            self.W_dec.grad -= parallel * self.W_dec.data


class SAE:
    """
    High-level SAE wrapper for Glyph pipeline.
    
    Greedily selects one sparsity using validation reconstruction quality.
    
    Args:
        cfg: Hydra config with sae settings
        input_dim: Dimension of input embeddings
        device: Device to use ('cuda', 'cpu', or 'auto')
        
    Example:
        >>> sae = SAE(cfg, input_dim=512)
        >>> sae.fit(train_embeddings, val_embeddings)
        >>> features = sae.embed(test_embeddings)
    """
    
    def __init__(self, cfg, input_dim: int, device: str = 'auto'):
        self.cfg = cfg
        self.input_dim = input_dim
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Will hold trained models for each top_k
        self.models: Dict[int, TopKSAE] = {}
        self.alive_features: Dict[int, torch.Tensor] = {}
        self.selected_top_k = None
        self.selection_report = None
    
    def fit(
        self, 
        train_embeddings: torch.Tensor,
        val_embeddings: torch.Tensor,
        top_ks: Optional[List[int]] = None,
        dict_size: Optional[int] = None,
    ) -> Dict[int, TopKSAE]:
        """
        Try increasing sparsities, then explore bounded quality upgrades.
        
        Args:
            train_embeddings: Training embeddings (n_train, input_dim)
            val_embeddings: Validation embeddings (n_val, input_dim)
            top_ks: List of top_k values to train (default from config)
            dict_size: Dictionary size (default from config)
            
        Returns:
            Dict containing the selected top_k and its trained model
        """
        from glyph.mechinterp.sae_trainer import train_sae
        
        top_ks = sorted(set(top_ks if top_ks is not None else self.cfg.sae.top_ks))
        if dict_size is None:
            dict_size = max(self.cfg.sae.matryoshka)
        if not top_ks or any(not isinstance(k, int) or k < 1 or k > dict_size for k in top_ks):
            raise ValueError('Candidate top_ks must be positive integers no larger than dict_size')
        settings = self.cfg.sae.get('selection', {})
        targets = list(settings.get('r2_targets', [0.85, 0.90, 0.95]))
        extra = settings.get('extra_sparsity_per_target', 1.0)
        step = settings.get('lookahead_step', 1)
        if not targets or targets != sorted(set(targets)) or not all(0 < x <= 1 for x in targets):
            raise ValueError('selection.r2_targets must be strictly increasing and in (0, 1]')
        if not math.isfinite(extra) or extra < 0 or not isinstance(step, int) or step < 1:
            raise ValueError('Selection requires a nonnegative extra allowance and positive integer step')
        if len(val_embeddings) < 2 or not torch.isfinite(val_embeddings).all():
            raise ValueError('SAE selection requires finite validation embeddings with at least two rows')
        if not torch.any(val_embeddings != val_embeddings[0]):
            raise ValueError('Validation reconstruction R² is undefined for constant embeddings')

        self.models = {}
        self.alive_features = {}
        self.selected_top_k = None
        self.selection_report = None
        history = []
        baseline_k = None
        budgets = None
        selected_level = -1
        selected_score = -math.inf
        selected_model = None
        selected_mask = None
        pending = list(top_ks)
        while pending:
            top_k = pending.pop(0)
            print(f"\n  Trying SAE with top_k={top_k}, dict_size={dict_size}")
            model, alive_mask = train_sae(
                train_embeddings=train_embeddings,
                val_embeddings=val_embeddings,
                input_dim=self.input_dim,
                dict_size=dict_size,
                top_k=top_k,
                cfg=self.cfg,
                device=self.device,
            )
            metrics = reconstruction_metrics(model, val_embeddings)
            score = metrics['r2']
            if score is None or not math.isfinite(score):
                raise ValueError('SAE produced an undefined/nonfinite validation reconstruction R²')
            history.append(dict(top_k=top_k, validation_r2=score, validation_mse=metrics['mse'],
                encoder_initialization_scale=getattr(model,'encoder_initialization_scale',None)))
            print(f"  Validation reconstruction R² = {score:.4f}")
            if baseline_k is None and score >= targets[0]:
                baseline_k = top_k
                budgets = [min(dict_size, max(top_ks), math.floor(
                    baseline_k * (1 + extra * level) + 1e-9)) for level in range(len(targets))]
                limit = budgets[-1]
                # Include exact budget boundaries even with a coarser search step.
                pending = sorted(set(range(top_k + step, limit + 1, step))
                                 | {k for k in top_ks if top_k < k <= limit}
                                 | {k for k in budgets if k > top_k})

            eligible_level = -1 if baseline_k is None else max(
                (level for level, target in enumerate(targets)
                 if score >= target and top_k <= budgets[level]), default=-1)
            choose = (score > selected_score if baseline_k is None
                      else eligible_level > selected_level)
            if choose:
                selected_model = model.to('cpu')
                selected_mask = alive_mask.cpu()
                self.selected_top_k = top_k
                selected_score = score
                selected_level = eligible_level
            del model
            if selected_level == len(targets) - 1:
                break

        self.models = {self.selected_top_k: selected_model.to(self.device)}
        self.alive_features = {self.selected_top_k: selected_mask.to(self.device)}
        reached_minimum = baseline_k is not None
        self.selection_report = dict(
            dictionary_size=dict_size, candidate_top_ks=top_ks, r2_targets=targets,
            encoder_initialization=getattr(selected_model,'encoder_initialization','legacy_transpose'),
            extra_sparsity_per_target=extra, lookahead_step=step,
            baseline_top_k=baseline_k, target_top_k_limits=budgets,
            selected_top_k=self.selected_top_k, selected_validation_r2=selected_score,
            reached_minimum=reached_minimum,
            selected_target=targets[selected_level] if reached_minimum else None,
            reason='highest_qualifying_target_with_smallest_tested_k' if reached_minimum
                   else 'best_reconstruction_below_minimum',
            tried=history,
        )
        print(f"  Selected SAE: k={self.selected_top_k}, validation R²={selected_score:.4f}")
        if not reached_minimum:
            warnings.warn(
                f'SAE validation reconstruction R²={selected_score:.4f} is below '
                f'{targets[0]:.2f}; using the best tested model (k={self.selected_top_k}). '
                'Consider more training or a different dictionary configuration.',
                UserWarning, stacklevel=2)
        return self.models
    
    def embed(
        self, 
        embeddings: torch.Tensor, 
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract sparse features using trained SAE.
        
        Args:
            embeddings: Input embeddings (n_samples, input_dim)
            top_k: Which retained model to use (default: selected model)
            
        Returns:
            Sparse activations (n_samples, dict_size)
        """
        if top_k is None:
            top_k = self.selected_top_k
        
        model = self.models[top_k]
        model.eval()
        
        with torch.no_grad():
            embeddings = embeddings.to(self.device)
            return model.embed(embeddings)
    
    def get_alive_features(self, top_k: Optional[int] = None) -> torch.Tensor:
        """Get mask of alive features for specified top_k."""
        if top_k is None:
            top_k = self.selected_top_k
        return self.alive_features[top_k]
