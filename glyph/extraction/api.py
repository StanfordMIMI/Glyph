from pathlib import Path
import pandas as pd
import torch

from glyph.models.registry import get_model
from glyph.models.base import BaseEmbeddingModel


def extract_embeddings(metadata, embedding_map, *, model_kwargs=None, device="auto", batch_size=1):
    """Extract from raw metadata; return tensors and IDs grouped by split.

    No implicit cache is read. Models and preprocessing are selected explicitly.
    Missing split labels default to train for standalone extraction.
    """
    frame = pd.read_csv(metadata, dtype={"sample_id": str}) if isinstance(metadata, (str, Path)) else metadata.copy()
    if "sample_id" not in frame or frame.sample_id.astype(str).duplicated().any():
        raise ValueError("Extraction requires unique sample_id values")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if "split" not in frame:
        frame["split"] = "train"
    if not set(frame.split).issubset({"train", "val", "test"}):
        raise ValueError("split must contain train, val, or test")
    missing = set(embedding_map) - set(frame.columns)
    if missing:
        raise ValueError(f"Raw extraction metadata lacks input columns: {sorted(missing)}")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    result = {split: {"sample_ids": sub.sample_id.tolist(), "embeddings": {}}
              for split, sub in frame.groupby("split", sort=False)}
    for modality, name in embedding_map.items():
        if isinstance(name, str):
            model = get_model(name, device=device, **(model_kwargs or {}).get(modality, {}))
        else:
            model = name
            if not isinstance(model, BaseEmbeddingModel):
                raise TypeError('Custom encoders must implement BaseEmbeddingModel; wrap your model with EmbeddingAdapter')
        for split, sub in frame.groupby("split", sort=False):
            values = sub[modality].tolist()
            chunks = []
            for start in range(0, len(values), batch_size):
                with torch.inference_mode():
                    batch = values[start:start + batch_size]
                    embedding = model.embed_batch(batch).detach().cpu()
                    if embedding.ndim != 2 or len(embedding) != len(batch) or not torch.isfinite(embedding).all():
                        raise ValueError(f"{modality}: encoder returned invalid or nonfinite embeddings")
                    chunks.append(embedding)
            result[split]["embeddings"][modality] = torch.cat(chunks)
        del model
    return result


def load_embeddings(paths, metadata, modalities, *, validate_splits=True):
    """Load explicit .pt paths and align every modality to metadata sample IDs.

    Files accept {sample_ids, embeddings} (HypothesisExtractor) or
    {sample_id, <modality>_embedding, split} (Glyph export).
    A directory resolves Glyph's exported filenames. All tensors are raw FM
    embeddings; a published SAE applies its own training normalization later.
    """
    models = dict(modalities) if hasattr(modalities, "items") else None
    modalities = list(modalities)
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        if path.is_dir():
            paths = {m: path / (m.replace("_path", "") + "_embeddings.pt") for m in modalities}
        elif len(modalities) == 1:
            paths = {modalities[0]: path}
        else:
            raise ValueError("For multiple modalities provide a directory or modality-to-path mapping")
    if set(paths) != set(modalities):
        raise ValueError("Embedding paths must cover exactly the configured modalities")
    if metadata.sample_id.astype(str).duplicated().any():
        raise ValueError("Metadata sample_id values must be unique")
    result = {split: {"sample_ids": sub.sample_id.tolist(), "embeddings": {}}
              for split, sub in metadata.groupby("split", sort=False)}
    for modality, path in paths.items():
        data = torch.load(path, map_location="cpu", weights_only=True)
        if models is not None and data.get("encoder") is not None and models[modality] != "precomputed" and data["encoder"] != models[modality]:
            raise ValueError(f"{modality}: saved encoder does not match the configured encoder")
        ids = data.get("sample_ids", data.get("sample_id"))
        tensor = data.get("embeddings", data.get(modality.replace("_path", "") + "_embedding"))
        if ids is None or tensor is None:
            raise ValueError(f"{modality}: expected sample_ids and embeddings, or Glyph export keys")
        if not torch.is_tensor(tensor) or tensor.ndim != 2 or len(ids) != len(tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"{modality}: embeddings must be a finite N x D tensor matching sample IDs")
        keys = [str(i) for i in ids]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{modality}: duplicate embedding sample IDs")
        index = {key: i for i, key in enumerate(keys)}
        missing = set(metadata.sample_id.astype(str)) - set(index)
        if missing:
            raise ValueError(f"{modality}: {len(missing)} metadata samples lack embeddings")
        if validate_splits and "split" in data:
            saved_splits = dict(zip(keys, data["split"]))
            if any(saved_splits[str(row.sample_id)] != row.split for row in metadata.itertuples()):
                raise ValueError(f"{modality}: saved splits differ from metadata")
        for split, sub in metadata.groupby("split", sort=False):
            result[split]["embeddings"][modality] = tensor[[index[str(i)] for i in sub.sample_id]].float()
    return result
