"""Load pretrained SAEs and apply them to embeddings."""

import math
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from glyph.zoo.registry import resolve

ASSET_DIR = Path(__file__).parent / "assets"
NORMALIZATION_ASSET = "normalization.yaml"
INTERPRETATIONS_ASSET = "interpretations.csv"

# Normalization maps every modality onto this row-norm regardless of its
# dimensionality; deviation means the embeddings are off-distribution relative
# to what the SAE was trained on, and concept identities stop being meaningful.
TARGET_NORM = math.sqrt(2048)
NORM_TOLERANCE = 0.35


def _asset(filename, entry, cache_dir):
    packaged = ASSET_DIR / filename
    if packaged.exists():
        return packaged
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=entry.repo,
            filename=filename,
            revision=entry.revision,
            local_dir=cache_dir,
        )
    )


def _download_weights(entry, cache_dir):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=entry.repo,
        filename=entry.weights,
        revision=entry.revision,
        local_dir=cache_dir,
    )


class PretrainedSAE:
    """A pretrained SAE plus the interpretations of its concepts."""

    def __init__(self, entry, W_enc, b_enc, norm, interpretations, decoder=None):
        self.decoder = decoder
        self.entry = entry
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.norm = norm
        self.interpretations = interpretations

    @property
    def name(self):
        return self.entry.name

    def normalize(self, embeddings, strict=True):
        """Apply the SAE's training-set normalization and check distribution."""
        if embeddings.shape[1] != self.entry.input_dim:
            raise ValueError(
                f"{self.name} expects {self.entry.input_dim}-d embeddings from "
                f"{self.entry.encoder}, got {embeddings.shape[1]}-d"
            )

        x = embeddings.float()
        mean = torch.tensor(self.norm["mean"], dtype=torch.float32)
        x = ((x - mean) / self.norm["std"]) * self.norm["dim_scale"]

        observed = x.norm(dim=1).mean().item()
        deviation = observed / TARGET_NORM - 1
        self.last_normalization_report = dict(mean_row_norm=observed, reference_row_norm=TARGET_NORM, relative_deviation=deviation, tolerance=NORM_TOLERANCE, within_reference_range=abs(deviation) <= NORM_TOLERANCE)
        if abs(deviation) > NORM_TOLERANCE:
            if strict:
                raise ValueError(
                    f"{self.name}: embeddings are off-distribution "
                    f"(mean row-norm {observed:.1f}, expected ~{TARGET_NORM:.1f}, "
                    f"{deviation:+.0%}). Check these embeddings came "
                    f"from {self.entry.encoder}, or pass strict=False to override."
                )
            warnings.warn(
                f"{self.name}: normalized embedding magnitude differs from the reference by {deviation:+.0%}; "
                "check the encoder recipe and assess concept transfer on this cohort.",
                UserWarning, stacklevel=2,
            )
        return x

    @torch.no_grad()
    def encode(self, embeddings, strict=True, *, device="cpu", batch_size=256):
        """Apply published normalization and sparse encoding in bounded batches."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        x = self.normalize(embeddings.detach().cpu(), strict=strict)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        weights, bias = self.W_enc.to(device), self.b_enc.to(device)
        chunks = []
        for start in range(0, len(x), batch_size):
            acts = F.relu(x[start:start + batch_size].to(device) @ weights + bias)
            values, indices = acts.topk(self.entry.top_k, dim=-1)
            chunks.append(torch.zeros_like(acts).scatter_(-1, indices, values).cpu())
        return torch.cat(chunks) if chunks else torch.empty(0, self.entry.n_concepts)

    @property
    def concepts(self):
        """Interpretations for this SAE's modality, indexed by concept id."""
        return self.interpretations

    def __repr__(self):
        return (
            f"PretrainedSAE({self.name!r}, encoder={self.entry.encoder!r}, "
            f"{self.entry.input_dim}d -> {self.entry.n_concepts} concepts, "
            f"top_k={self.entry.top_k})"
        )


def load_pretrained(name, cache_dir=None):
    """Load a pretrained SAE from the glyph zoo, downloading weights if needed."""
    import pandas as pd

    entry = resolve(name)
    cache_dir = cache_dir or Path.home() / ".cache" / "glyph" / "zoo"

    weights = torch.load(
        _download_weights(entry, cache_dir), map_location="cpu", weights_only=True
    )
    # The trained encoder is z = x @ W_enc + b_enc; b_dec is decoder-only and
    # must not be subtracted here.
    W_enc = weights["W_enc"].float()
    b_enc = weights["b_enc"].float()

    with open(_asset(NORMALIZATION_ASSET, entry, cache_dir)) as f:
        norm = yaml.safe_load(f)[entry.modality]

    interpretations = pd.read_csv(_asset(INTERPRETATIONS_ASSET, entry, cache_dir))
    interpretations = interpretations[interpretations["inputs"] == entry.modality]

    decoder = {key: weights[key].float() for key in ("W_dec", "b_dec")} if "W_dec" in weights and "b_dec" in weights else None
    return PretrainedSAE(entry, W_enc, b_enc, norm, interpretations, decoder=decoder)
