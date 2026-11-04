"""
SAE training loop with ghost gradients and weight normalization.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from glyph.mechinterp.sae import TopKSAE


@torch.no_grad()
def calibrate_encoder_initialization(model, train_embeddings, prefix_sizes=None, batch_size=256):
    """Fit a positive starting gain using training reconstruction loss only."""
    prefixes=list(prefix_sizes) if prefix_sizes else [model.dict_size]
    dot=0.;prediction_square=0.;target_square=0.;before=0.;count=0
    device=model.W_enc.device
    for start in range(0,len(train_embeddings),batch_size):
        x=train_embeddings[start:start+batch_size].to(device)
        sparse=model(x)['z_sparse']
        for size in prefixes:
            prediction=model.reconstruct_with_prefix(sparse,size).double()
            target=x.double()
            dot+=float((prediction*target).sum())
            prediction_square+=float(prediction.square().sum())
            target_square+=float(target.square().sum())
            before+=float((prediction-target).square().sum())
            count+=target.numel()
    scale=dot/prediction_square if prediction_square>0 else 1.
    if not np.isfinite(scale) or scale<=0:scale=1.
    # Biases start at zero. Positive gain preserves the TopK support exactly.
    model.W_enc.mul_(scale)
    after=max(0.,scale*scale*prediction_square-2*scale*dot+target_square)
    return dict(scale=scale,mse_before=before/max(count,1),mse_after=after/max(count,1))


def train_sae(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    input_dim: int,
    dict_size: int,
    top_k: int,
    cfg,
    device: str = 'cuda',
) -> Tuple[TopKSAE, torch.Tensor]:
    """
    Train a TopK Sparse Autoencoder.
    
    Args:
        train_embeddings: Training embeddings (n_train, input_dim)
        val_embeddings: Validation embeddings (n_val, input_dim)
        input_dim: Dimension of input embeddings
        dict_size: Size of learned dictionary
        top_k: Number of active features per sample
        cfg: Hydra config with sae settings
        device: Device to train on
        
    Returns:
        Tuple of (trained model, alive_features_mask)
    """
    # SAE fitting must also be reproducible when no outcome probes precede it.
    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.get("seed", 42)))

    # Create model
    model = TopKSAE(
        input_dim=input_dim,
        dict_size=dict_size,
        top_k=top_k,
    ).to(device)
    
    # Move data to device
    train_embeddings = train_embeddings.to(device)
    val_embeddings = val_embeddings.to(device)
    
    model.encoder_initialization = 'legacy_transpose'
    model.encoder_initialization_scale = 1.0
    if cfg.sae.get('calibrate_encoder_init', True):
        diagnostic = calibrate_encoder_initialization(
            model, train_embeddings, cfg.sae.matryoshka,
            batch_size=max(1,min(cfg.sae.get('batch_size',256),len(train_embeddings))))
        model.encoder_initialization = 'training_calibrated_transpose'
        model.encoder_initialization_scale = diagnostic['scale']
        print(f"  Initial encoder gain: {diagnostic['scale']:.6g}; "
              f"training reconstruction MSE {diagnostic['mse_before']:.6g} -> {diagnostic['mse_after']:.6g}")

    # Training settings from config
    lr = cfg.sae.learning_rate
    max_steps = cfg.sae.max_steps
    sparsity_coef = cfg.sae.sparsity_coefficient
    use_ghost_grads = cfg.sae.use_ghost_grads
    aux_scale = cfg.sae.aux_scale
    dead_threshold = cfg.sae.dead_feature_threshold
    kaux = cfg.sae.kaux
    matryoshka_sizes = cfg.sae.matryoshka  # List of prefix sizes for Matryoshka loss
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss function
    mse_loss_fn = nn.MSELoss()
    
    # Create data loader
    batch_size = min(cfg.sae.get('batch_size', 256), len(train_embeddings))
    n_batches = (len(train_embeddings) + batch_size - 1) // batch_size
    
    # Training state
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 20  # Early stopping patience (in validation checks)
    val_check_interval = max(1, max_steps // 50)  # Check validation ~50 times
    
    # Feature activation tracking
    feature_activations = torch.zeros(dict_size, device=device)
    
    step = 0
    epoch = 0
    
    pbar = tqdm(total=max_steps, desc=f"Training SAE (k={top_k})")
    
    while step < max_steps:
        # Shuffle training data each epoch
        perm = torch.randperm(len(train_embeddings))
        train_embeddings_shuffled = train_embeddings[perm]
        
        for batch_idx in range(n_batches):
            if step >= max_steps:
                break
            
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_embeddings))
            x = train_embeddings_shuffled[start_idx:end_idx]
            
            # Normalize decoder weights
            model.normalize_decoder()
            
            # Forward pass
            model.train()
            outputs = model(x)
            
            x_hat = outputs['x_hat']
            z_sparse = outputs['z_sparse']
            
            # Reconstruction loss (MSE) - with Matryoshka if enabled
            if matryoshka_sizes is not None and len(matryoshka_sizes) > 0:
                # Matryoshka loss: compute nested reconstruction losses at each prefix size
                total_mse = 0.0
                for prefix_size in matryoshka_sizes:
                    x_hat_prefix = model.reconstruct_with_prefix(z_sparse, prefix_size)
                    total_mse += mse_loss_fn(x_hat_prefix, x)
                mse_loss = total_mse / len(matryoshka_sizes)
            else:
                # Standard reconstruction loss
                mse_loss = mse_loss_fn(x_hat, x)
            
            # Sparsity loss (L1)
            l1_loss = sparsity_coef * z_sparse.abs().mean()
            
            # Ghost gradient loss for dead features
            ghost_loss = torch.tensor(0.0, device=device)
            if use_ghost_grads:
                # Identify dead features (low average activation)
                dead_mask = z_sparse.abs().mean(dim=0) < dead_threshold
                
                if dead_mask.sum() > 0:
                    # Forward pass on dead features only
                    dead_indices = torch.where(dead_mask)[0]
                    z = outputs['z']
                    
                    # Apply topk only to dead features
                    z_sub = z[:, dead_indices]
                    k_aux = min(kaux, len(dead_indices))
                    if k_aux > 0:
                        topk_vals, topk_idx = torch.topk(z_sub, k=k_aux, dim=1)
                        z_sparse_aux = torch.zeros_like(z_sub).scatter_(1, topk_idx, topk_vals)
                        
                        # Reconstruct using dead features
                        W_dec_dead = model.W_dec[dead_indices, :]
                        x_hat_aux = z_sparse_aux @ W_dec_dead + model.b_dec
                        
                        # Ghost loss: encourage dead features to explain residual
                        residual = (x - x_hat).detach()
                        ghost_loss = aux_scale * mse_loss_fn(x_hat_aux, residual)
            
            # Total loss
            loss = mse_loss + l1_loss + ghost_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Remove parallel gradient component from decoder
            model.remove_parallel_gradient_component()
            
            optimizer.step()
            
            # Track feature activations
            with torch.no_grad():
                active = (z_sparse.abs() > dead_threshold).float().mean(0)
                feature_activations = 0.99 * feature_activations + 0.01 * active
            
            step += 1
            pbar.update(1)
            
            # Validation check
            if step % val_check_interval == 0:
                val_loss = evaluate_sae(model, val_embeddings, mse_loss_fn, batch_size)
                
                pbar.set_postfix({
                    'train_loss': f'{loss.item():.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'alive': f'{(feature_activations > 0).sum().item()}'
                })
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\n  Early stopping at step {step}")
                    break
        
        epoch += 1
        
        if patience_counter >= patience:
            break
    
    pbar.close()
    
    # Load best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    # Compute alive features on full training set
    model.eval()
    with torch.no_grad():
        all_activations = []
        for i in range(0, len(train_embeddings), batch_size):
            batch = train_embeddings[i:i+batch_size]
            z_sparse = model.embed(batch)
            all_activations.append(z_sparse)
        
        all_activations = torch.cat(all_activations, dim=0)
        alive_mask = (all_activations.abs() > dead_threshold).any(dim=0)
    
    n_alive = alive_mask.sum().item()
    print(f"  Training complete. Alive features: {n_alive}/{dict_size}")
    
    return model, alive_mask


def evaluate_sae(
    model: TopKSAE,
    embeddings: torch.Tensor,
    loss_fn: nn.Module,
    batch_size: int = 256,
) -> float:
    """Evaluate SAE on validation/test set."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            outputs = model(batch)
            loss = loss_fn(outputs['x_hat'], batch)
            total_loss += loss.item() * len(batch)
            n_samples += len(batch)
    
    return total_loss / n_samples


def extract_sae_features(
    model: TopKSAE,
    embeddings: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    """Extract sparse features from embeddings using trained SAE."""
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            features = model.embed(batch)
            all_features.append(features.cpu())
    
    return torch.cat(all_features, dim=0)
