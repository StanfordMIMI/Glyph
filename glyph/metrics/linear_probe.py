"""
Linear probe implementations for evaluating embeddings and sparse features.

Trains a linear classifier on embeddings with grid search over learning rate
and weight decay hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional
from glyph.metrics.statistics import compute_auc, compute_r2, compute_rmse


class UnestimableProbe(ValueError):
    pass


def _set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device(device: str = 'auto') -> str:
    """Determine device (cuda/cpu) from string or auto-detect."""
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def _prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    device: str
) -> tuple:
    """Convert numpy arrays to torch tensors on device."""
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    return X_tensor, y_tensor


def run_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    outcome_type: str = 'binary',
    lr_grid: List[float] = None,
    wd_grid: List[float] = None,
    max_epochs: int = 400,
    patience: int = 20,
    metric_tol: float = 1e-4,
    device: str = 'auto',
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train linear probe with grid search over learning rate and weight decay.

    Uses PyTorch with early stopping based on validation metric (AUC for binary, R2 for regression).

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        outcome_type: Type of outcome ('binary' or 'regression')
        lr_grid: Learning rates to search (default: [1e-4, 3e-4, 1e-3, 3e-3])
        wd_grid: Weight decays to search (default: [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3])
        max_epochs: Maximum training epochs per configuration
        patience: Early stopping patience (epochs without improvement)
        metric_tol: Minimum metric improvement to reset patience
        device: Device to use ('auto', 'cuda', 'cpu')
        seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
            - 'model': Trained torch.nn.Linear model
            - 'best_hparams': Dict with 'lr' and 'wd'
            - 'val_metric': Best validation metric (AUC or R2)
            - 'test_metric': Final test metric (AUC or R2)
            - 'test_preds': Test set predictions (n_test,)
            - 'all_metrics': Dict with train/val/test metrics
            - 'outcome_type': Type of outcome used
    """
    # Set defaults
    if lr_grid is None:
        lr_grid = [1e-4, 3e-4, 1e-3, 3e-3]
    if wd_grid is None:
        wd_grid = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3]

    if outcome_type not in {"binary", "regression"}:
        raise ValueError("Linear probes support binary or regression outcomes")
    if max_epochs < 1 or patience < 1 or not lr_grid or not wd_grid:
        raise ValueError("Provide nonempty parameter grids and positive epochs/patience")
    arrays = {}
    masks = {}
    input_dim = None
    for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError(f"{split}: expected aligned N x D features and N labels")
        if input_dim is None:
            input_dim = X.shape[1]
        if X.shape[1] != input_dim:
            raise ValueError("Probe feature dimensions differ between splits")
        finite_X = np.isfinite(X).all(axis=1)
        valid = finite_X & np.isfinite(y) & (y != -1)
        if outcome_type == "binary" and not np.isin(y[valid], [0, 1]).all():
            raise ValueError(f"{split}: binary probe labels must be 0/1, -1, or NaN")
        if split in {"train", "val"}:
            if valid.sum() < 2:
                raise UnestimableProbe(f"{split}: fewer than two eligible probe observations")
            if outcome_type == "binary" and len(np.unique(y[valid])) != 2:
                raise UnestimableProbe(f"{split}: binary probes require both classes")
            if split == "val" and outcome_type == "regression" and np.ptp(y[valid]) == 0:
                raise UnestimableProbe("val: regression selection needs nonconstant labels")
        clean_X = X.copy()
        clean_X[~finite_X] = 0
        arrays[split] = (clean_X, y, finite_X)
        masks[split] = valid
    X_train, y_train, _ = arrays["train"]
    X_val, y_val, _ = arrays["val"]
    X_test, y_test, _ = arrays["test"]
    _set_seed(seed)
    device = _get_device(device)
    metric_name = 'AUC' if outcome_type == 'binary' else 'R²'
    if verbose:
        print(f"Training linear probe on {device} ({outcome_type})")

    # Prepare data
    X_train_t, y_train_t = _prepare_data(X_train, y_train, device)
    X_val_t, y_val_t = _prepare_data(X_val, y_val, device)
    X_test_t, y_test_t = _prepare_data(X_test, y_test, device)

    # Precompute valid masks (outcome != -1) for each split
    train_valid = torch.as_tensor(masks["train"], device=device)
    val_valid = torch.as_tensor(masks["val"], device=device)
    test_valid = torch.as_tensor(masks["test"], device=device)

    in_dim = X_train.shape[1]

    # Grid search
    best_global_metric = -np.inf
    best_state = None
    best_hparams = None

    for lr in lr_grid:
        for wd in wd_grid:
            # Initialize model
            model = nn.Linear(in_dim, 1).to(device)

            # Choose loss function based on outcome type
            if outcome_type == 'binary':
                criterion = nn.BCEWithLogitsLoss()
            else:  # regression
                criterion = nn.MSELoss()

            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

            # Track best for this config
            best_val_metric = -np.inf
            best_local_state = None
            bad_epochs = 0

            # Training loop
            for epoch in range(max_epochs):
                # Train
                model.train()
                optimizer.zero_grad()
                logits = model(X_train_t).squeeze(1)
                loss = criterion(logits[train_valid], y_train_t[train_valid])
                loss.backward()
                optimizer.step()

                # Validate
                model.eval()
                with torch.no_grad():
                    v_logits = model(X_val_t).squeeze(1)

                    # Compute metric based on outcome type
                    if outcome_type == 'binary':
                        v_preds = torch.sigmoid(v_logits)
                        val_metric = compute_auc(
                            y_val_t[val_valid].cpu().numpy(),
                            v_preds[val_valid].cpu().numpy()
                        )
                    else:  # regression
                        v_preds = v_logits
                        val_metric = compute_r2(
                            y_val_t[val_valid].cpu().numpy(),
                            v_preds[val_valid].cpu().numpy()
                        )

                # Early stopping check
                if val_metric > best_val_metric + metric_tol:
                    best_val_metric = val_metric
                    best_local_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if bad_epochs >= patience:
                    break

            # Update global best
            if best_local_state is not None and best_val_metric > best_global_metric:
                best_global_metric = best_val_metric
                best_state = best_local_state
                best_hparams = {'lr': lr, 'wd': wd}

    if best_state is None:
        raise UnestimableProbe("No finite validation metric was obtained during probe training")
    if verbose:
        print(f"Best hyperparameters: lr={best_hparams['lr']}, wd={best_hparams['wd']}")
        print(f"Best validation {metric_name}: {best_global_metric:.4f}")

    # Load best model
    best_model = nn.Linear(in_dim, 1).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    predictions, metrics = {}, {}
    with torch.no_grad():
        for split, (X, y, finite_X) in arrays.items():
            logits = best_model(torch.as_tensor(X, device=device)).squeeze(1)
            preds = (torch.sigmoid(logits) if outcome_type == 'binary' else logits).cpu().numpy()
            preds[~finite_X] = np.nan
            predictions[split] = preds
            valid = masks[split]
            if valid.sum() < 2 or (outcome_type == 'binary' and len(np.unique(y[valid])) < 2):
                metrics[split] = np.nan
            else:
                score = compute_auc if outcome_type == 'binary' else compute_r2
                metrics[split] = float(score(y[valid], preds[valid]))
    train_metric, val_metric, test_metric = (metrics[k] for k in ('train', 'val', 'test'))
    test_preds = predictions['test']

    if verbose:
        print(f"Final {metric_name}s - Train: {train_metric:.4f}, Val: {val_metric:.4f}, Test: {test_metric:.4f}")

    return {
        'model': best_model,
        'best_hparams': best_hparams,
        'val_metric': val_metric,
        'test_metric': test_metric,
        'test_preds': test_preds,
        'all_metrics': {
            'train': train_metric,
            'val': val_metric,
            'test': test_metric
        },
        'outcome_type': outcome_type,
        'metric_name': metric_name,
        'eligible_counts': {key: int(value.sum()) for key, value in masks.items()},
        'predictions': predictions,
    }


def fit_linear_probe_from_embeddings(
    embeddings_dict: Dict[str, Dict[str, Any]],
    outcomes_dict: Dict[str, np.ndarray],
    outcome_type: str = 'binary',
    modality: str = 'image_path',
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience wrapper for linear probe that accepts EmbeddingExtractor output.

    Args:
        embeddings_dict: Output from EmbeddingExtractor.extract_all_splits()
                        Format: {'train': {'sample_ids': [...], 'embeddings': {...}}, ...}
        outcomes_dict: Dict mapping split names to outcome arrays
                      Format: {'train': np.ndarray, 'val': np.ndarray, 'test': np.ndarray}
        outcome_type: Type of outcome ('binary' or 'regression')
        modality: Which modality from embeddings to use (default: 'image_path')
        **kwargs: Additional arguments passed to run_linear_probe()

    Returns:
        Same as run_linear_probe()

    Example:
        >>> from glyph.models import EmbeddingExtractor
        >>> from glyph.metrics.linear_probe import fit_linear_probe_from_embeddings
        >>>
        >>> # Extract embeddings
        >>> extractor = EmbeddingExtractor(cfg, {'image_path': 'resnet18'})
        >>> embeddings = extractor.extract_all_splits(dataloaders)
        >>>
        >>> # Prepare outcomes
        >>> outcomes = {'train': y_train, 'val': y_val, 'test': y_test}
        >>>
        >>> # Fit linear probe
        >>> results = fit_linear_probe_from_embeddings(embeddings, outcomes, outcome_type='binary')
        >>> print(f"Test metric: {results['test_metric']:.4f}")
    """
    # Extract embeddings for specified modality
    X_train = embeddings_dict['train']['embeddings'][modality].cpu().numpy()
    X_val = embeddings_dict['val']['embeddings'][modality].cpu().numpy()
    X_test = embeddings_dict['test']['embeddings'][modality].cpu().numpy()

    # Get outcomes
    y_train = outcomes_dict['train']
    y_val = outcomes_dict['val']
    y_test = outcomes_dict['test']

    try:
        result = run_linear_probe(
            X_train, y_train, X_val, y_val, X_test, y_test,
            outcome_type=outcome_type, **kwargs)
        result['status'] = 'ok'
        return result
    except UnestimableProbe as error:
        return {'model': None, 'best_hparams': None, 'val_metric': np.nan,
                'test_metric': np.nan, 'test_preds': np.full(len(y_test), np.nan),
                'all_metrics': {split: np.nan for split in ('train', 'val', 'test')},
                'outcome_type': outcome_type, 'status': 'unestimable', 'reason': str(error)}


def compute_aucs(
    model: nn.Module,
    X_dict: Dict[str, np.ndarray],
    y_dict: Dict[str, np.ndarray],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute AUC for multiple dataset splits.

    Args:
        model: Trained linear probe model
        X_dict: Dict mapping split names to feature arrays
        y_dict: Dict mapping split names to label arrays
        device: Device to use

    Returns:
        Dict mapping split names to AUC scores

    Example:
        >>> aucs = compute_aucs(model,
        ...                     {'train': X_train, 'test': X_test},
        ...                     {'train': y_train, 'test': y_test})
        >>> print(f"Train AUC: {aucs['train']:.4f}")
    """
    model.eval()
    device = _get_device(device)

    auc_results = {}

    with torch.no_grad():
        for split_name in X_dict.keys():
            X = torch.tensor(X_dict[split_name], dtype=torch.float32).to(device)
            y = y_dict[split_name]

            logits = model(X).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            y = np.asarray(y)
            valid = np.isfinite(y) & (y != -1) & np.isfinite(probs) & np.isfinite(X_dict[split_name]).all(axis=1)
            auc = compute_auc(y[valid], probs[valid]) if len(np.unique(y[valid])) == 2 else np.nan
            auc_results[split_name] = auc

    return auc_results
