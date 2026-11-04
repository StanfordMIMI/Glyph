"""Versioned fitted runs and pretrained dictionary integration."""
import json
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from glyph.mechinterp.sae import SAE, TopKSAE, reconstruction_metrics


class PublishedTopKSAE(torch.nn.Module):
    def __init__(self, input_dim, dict_size, top_k):
        super().__init__()
        self.input_dim, self.dict_size, self.top_k = input_dim, dict_size, top_k
        self.W_enc = torch.nn.Parameter(torch.empty(input_dim, dict_size))
        self.b_enc = torch.nn.Parameter(torch.empty(dict_size))

    def embed(self, x):
        acts = torch.relu(x @ self.W_enc + self.b_enc)
        values, indices = acts.topk(self.top_k, dim=-1)
        return torch.zeros_like(acts).scatter_(-1, indices, values)


def _device(device):
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


def _redact(value):
    if isinstance(value, dict):
        return {k: (None if (k.lower() in {"token", "hf_token", "huggingface_token", "password", "secret", "api_key", "api_token"} or k.lower().endswith(("_api_key", "_api_token", "_password", "_secret")))
                    else _redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _metadata(obj):
    frames = []
    for split, loader in obj.dataloaders.items():
        frame = loader.dataset.metadata.copy()
        if "sample_id" not in frame:
            frame = frame.rename(columns={loader.dataset.sample_id_col: "sample_id"})
        frame["split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def start_run(obj, pretrained, embeddings_path, recompute_embeddings):
    if pretrained is not None and set(pretrained) != set(obj.cfg.data.embedding_map):
        raise ValueError("pretrained must map every configured input column to a published bundle")
    explicit = embeddings_path if embeddings_path is not None else obj.cfg.get("path_to_embeddings")
    if not recompute_embeddings and explicit is None:
        raise ValueError("recompute_embeddings=False requires an explicit embeddings_path")
    if pretrained:
        from glyph.zoo.registry import resolve
        expected = {"image": "merlin", "findings": "qwen3", "labs": "labs"}
        # Choose the encoder belonging to each dictionary before loading any model.
        for modality, name in pretrained.items():
            obj.cfg.data.embedding_map[modality] = expected[resolve(name).modality]
    root = getattr(obj, "_output_root", Path(obj.cfg.data.output_dir))
    obj._output_root = Path(root)
    obj.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    obj.run_dir = Path(root) / obj.run_id
    obj.run_dir.mkdir(parents=True, exist_ok=False)
    obj.cfg.data.output_dir = str(obj.run_dir)
    obj.phewas_results = None
    obj.autointerp_results = None
    manifest = {
        "schema_version": 1, "run_id": obj.run_id, "status": "running",
        "glyph_version": __import__("glyph").__version__,
        "device": _device(obj.cfg.device), "seed": int(obj.cfg.seed),
        "outcome_types": dict(obj.outcome_types),
        "pretrained": dict(pretrained or {}),
        "embedding_source": "explicit_files" if explicit is not None else "raw_inputs",
        "embeddings_path": {k: str(v) for k, v in explicit.items()} if isinstance(explicit, dict) else str(explicit) if explicit else None,
    }
    (obj.run_dir / "run.json").write_text(json.dumps(manifest, indent=2))
    config = _redact(OmegaConf.to_container(obj.cfg, resolve=False))
    OmegaConf.save(OmegaConf.create(config), obj.run_dir / "config.yaml")


def fit_embeddings(obj, pretrained, embeddings_path, recompute_embeddings):
    from glyph.extraction import extract_embeddings, load_embeddings
    if not obj.dataloaders and obj.embedding_extractor is None:
        from glyph.dataloader import get_dataloaders
        obj.dataloaders = get_dataloaders(obj.cfg, allow_missing_inputs=True)
    explicit = embeddings_path if embeddings_path is not None else obj.cfg.get("path_to_embeddings")
    if explicit is not None:
        obj.embeddings = load_embeddings(explicit, _metadata(obj), obj.cfg.data.embedding_map)
    elif obj.embedding_extractor is not None:
        # Compatibility for callers supplying their own extractor; no cache reuse.
        obj.embedding_extractor.cache_embeddings = False
        obj.embeddings = obj.embedding_extractor.extract_all_splits(obj.dataloaders, show_progress=True)
    else:
        obj.embeddings = extract_embeddings(
            _metadata(obj), dict(obj.cfg.data.embedding_map), device=obj.cfg.device,
            batch_size=obj.cfg.data.get("extraction_batch_size", 1),
            model_kwargs=OmegaConf.to_container(obj.cfg.data.embedding_model_kwargs) if obj.cfg.data.embedding_model_kwargs else None)


def fit_dictionaries(obj, pretrained):
    obj.sae_normalization = {}
    existing = getattr(obj, 'pretrained_bundles', {}) or {}
    obj.pretrained_bundles = {}
    obj.published_interpretations = None
    interps = []
    for modality in obj.cfg.data.embedding_map:
        if pretrained:
            from glyph.zoo import load_pretrained
            from glyph.zoo.registry import resolve
            bundle = existing.get(modality)
            if bundle is None or bundle.name != resolve(pretrained[modality]).name:
                bundle = load_pretrained(pretrained[modality], cache_dir=Path(obj.cfg.data.cache_dir) / "zoo")
            obj.pretrained_bundles[modality] = bundle
            obj.sae_normalization[modality] = {"kind": "published", **bundle.norm}
            sae = SAE(obj.cfg, input_dim=bundle.entry.input_dim, device=obj.cfg.device)
            model = PublishedTopKSAE(bundle.entry.input_dim, bundle.entry.n_concepts, bundle.entry.top_k)
            with torch.no_grad():
                model.W_enc.copy_(bundle.W_enc)
                model.b_enc.copy_(bundle.b_enc)
            sae.selected_top_k = bundle.entry.top_k
            sae.models = {sae.selected_top_k: model.to(sae.device).eval()}
            obj.sae_features[modality] = {
                split: bundle.encode(data["embeddings"][modality], device=sae.device, strict=getattr(obj, "_measurement_strict", True)).cpu()
                for split, data in obj.embeddings.items()}
            reference = "train" if "train" in obj.sae_features[modality] else next(iter(obj.sae_features[modality]))
            mask = (obj.sae_features[modality][reference] > 0).any(dim=0)
            sae.alive_features = {sae.selected_top_k: mask.to(sae.device)}
            sae.selection_report = {"dictionary_size": bundle.entry.n_concepts, "selected_top_k": sae.selected_top_k, "pretrained": bundle.name,
                                    "repo": bundle.entry.repo, "revision": bundle.entry.revision}
            interp = bundle.interpretations.copy()
            source = "feature_idx" if "feature_idx" in interp else "feature_name" if "feature_name" in interp else "feature"
            interp["feature_idx"] = interp[source].astype(str).str.extract(r"(\d+)$")[0].astype(int)
            interp["interpretation"] = interp.get("extracted_interpretation", interp.get("interpretation", ""))
            interp["test_accuracy"] = interp.get("test_interpretation_discrimination_accuracy", np.nan)
            interp["modality"] = modality
            interp["top_k"] = sae.selected_top_k
            interps.append(interp)
        else:
            raw = obj.embeddings["train"]["embeddings"][modality]
            mean = raw.mean(dim=0, keepdim=True)
            std = raw.std(dim=0, keepdim=True).clamp(min=1e-8)
            obj.sae_normalization[modality] = {"kind": "trained", "mean": mean.cpu(), "std": std.cpu()}
            normalized = {split: (data["embeddings"][modality] - mean) / std for split, data in obj.embeddings.items()}
            sae = SAE(obj.cfg, input_dim=raw.shape[1], device=obj.cfg.device)
            sae.fit(normalized["train"], normalized["val"])
            obj.sae_features[modality] = {split: sae.embed(x).cpu() for split, x in normalized.items()}
            for split, x in normalized.items():
                print(f"{modality} {split} reconstruction: {reconstruction_metrics(sae.models[sae.selected_top_k], x)}")
        obj.sae_models[modality] = sae
        obj.sae_alive_features[(modality, sae.selected_top_k)] = sae.get_alive_features().cpu()
    if interps:
        obj.published_interpretations = pd.concat(interps, ignore_index=True)
        obj.autointerp_results = obj.published_interpretations


def encode_embeddings(obj, embeddings, modality):
    stats = obj.sae_normalization[modality]
    x = embeddings.float().cpu()
    mean, std = torch.as_tensor(stats["mean"]), torch.as_tensor(stats["std"])
    x = (x - mean) / std
    if stats["kind"] == "published":
        x = x * stats["dim_scale"]
    return obj.sae_models[modality].embed(x).detach().cpu()


def save_run(obj):
    bundles = {}
    for modality, sae in obj.sae_models.items():
        model = sae.models[sae.selected_top_k]
        bundles[modality] = {
            "normalization": obj.sae_normalization[modality],
            "input_dim": model.input_dim, "dict_size": model.dict_size,
            "top_k": sae.selected_top_k, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "alive_mask": sae.get_alive_features().cpu(), "selection": sae.selection_report,
        }
    torch.save(bundles, obj.run_dir / "fitted_saes.pt")
    torch.save({split: {key: torch.as_tensor(value) for key, value in columns.items()}
                for split, columns in obj.outcomes.items()}, obj.run_dir / "outcomes.pt")
    if obj.published_interpretations is not None:
        obj.published_interpretations.to_csv(obj.run_dir / "published_interpretations.csv", index=False)


def complete_run(obj):
    path = obj.run_dir / "run.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(manifest, indent=2))


def load_run(cls, path, device="auto"):
    path = Path(path)
    manifest = json.loads((path / "run.json").read_text())
    if manifest["schema_version"] != 1 or manifest["status"] != "complete":
        raise ValueError("Run is incomplete or uses an unsupported schema")
    obj = cls.__new__(cls)
    obj.cfg = OmegaConf.load(path / "config.yaml")
    obj.cfg.device = device
    obj.cfg.data.output_dir = str(path)
    obj.run_dir, obj.run_id = path, manifest["run_id"]
    obj._output_root = path.parent
    obj.dataloaders = {}
    obj.embedding_extractor = None
    obj.sae_models, obj.sae_normalization, obj.sae_alive_features = {}, {}, {}
    bundles = torch.load(path / "fitted_saes.pt", map_location="cpu", weights_only=True)
    for modality, bundle in bundles.items():
        sae = SAE(obj.cfg, input_dim=bundle["input_dim"], device=device)
        stats = bundle["normalization"]
        model_cls = PublishedTopKSAE if stats["kind"] == "published" else TopKSAE
        model = model_cls(bundle["input_dim"], bundle["dict_size"], bundle["top_k"])
        model.load_state_dict(bundle["state_dict"])
        sae.selected_top_k = bundle["top_k"]
        sae.models = {sae.selected_top_k: model.to(_device(device)).eval()}
        sae.alive_features = {sae.selected_top_k: bundle["alive_mask"]}
        sae.selection_report = bundle["selection"]
        obj.sae_models[modality], obj.sae_normalization[modality] = sae, stats
        obj.sae_alive_features[(modality, sae.selected_top_k)] = bundle["alive_mask"]
    # Pickles are Glyph's existing result format: only load runs you trust.
    with (path / "sae_features.pkl").open("rb") as f:
        obj.sae_features = pickle.load(f)
    obj.embeddings = {}
    for modality in bundles:
        data = torch.load(path / (modality.replace("_path", "") + "_embeddings.pt"), map_location="cpu", weights_only=True)
        tensor = data[modality.replace("_path", "") + "_embedding"]
        for split in dict.fromkeys(data["split"]):
            indices = [i for i, value in enumerate(data["split"]) if value == split]
            obj.embeddings.setdefault(split, {"sample_ids": [data["sample_id"][i] for i in indices], "embeddings": {}})
            obj.embeddings[split]["embeddings"][modality] = tensor[indices]
    obj.outcomes = {split: {k: v.numpy() for k, v in columns.items()}
                    for split, columns in torch.load(path / "outcomes.pt", weights_only=True).items()}
    obj.outcome_types = manifest["outcome_types"] if "outcome_types" in manifest else {key: "binary" if set(np.unique(value[np.isfinite(value) & (value != -1)])) <= {0, 1}
                         else "regression" for key, value in obj.outcomes.get("train", next(iter(obj.outcomes.values()), {})).items()}
    for attr, filename in [("phewas_results", "hypotheses.pkl"), ("linear_probe_results", "linear_probe_results.pkl"),
                           ("sae_linear_probe_results", "sae_linear_probe_results.pkl")]:
        file = path / filename
        value = None
        if file.exists():
            with file.open("rb") as f:
                value = pickle.load(f)
        setattr(obj, attr, value)
    interp = path / "published_interpretations.csv"
    obj.published_interpretations = pd.read_csv(interp) if interp.exists() else None
    cached = path / "autointerp_results.pkl"
    obj.autointerp_results = pd.read_pickle(cached) if cached.exists() else obj.published_interpretations
    if manifest.get("mode") == "pretrained_measurement":
        from glyph.core.measurement import restore
        restore(obj, manifest)
    return obj
