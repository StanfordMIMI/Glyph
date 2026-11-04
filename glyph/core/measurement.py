"""Apply published dictionaries to reader cohorts without SAE training or splits."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import uuid

import pandas as pd
import torch
from omegaconf import OmegaConf

from glyph.config import load_config

COLUMNS = {'image': ('image_path', 'ct_path'), 'findings': ('report_path', 'findings_path'), 'labs': ('labs_path',)}
ENCODERS = {'image': 'merlin', 'findings': 'qwen3', 'labs': 'labs'}


def from_pretrained(cls, name, *, device='auto', cache_dir='cache', output_dir='outputs',
                    input_col=None, encoder_kwargs=None, batch_size=1, strict=False):
    from glyph.zoo import load_pretrained
    from glyph.zoo.registry import resolve
    entry = resolve(name)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    obj = cls.__new__(cls)
    obj.cfg = load_config()
    column = input_col or COLUMNS[entry.modality][0]
    obj.cfg.device = device
    obj.cfg.data.metadata = None
    obj.cfg.data.input_cols = [column]
    obj.cfg.data.outcome_cols = []
    obj.cfg.data.embedding_map = {column: ENCODERS[entry.modality]}
    obj.cfg.data.embedding_model_kwargs = {column: encoder_kwargs or {}}
    obj.cfg.data.extraction_batch_size = batch_size
    obj.cfg.data.cache_dir, obj.cfg.data.output_dir = str(cache_dir), str(output_dir)
    obj.cfg.autointerp.enabled = False
    obj._output_root = Path(output_dir)
    obj.dataloaders, obj.outcome_types = {}, {}
    obj.embedding_extractor = None
    for attr in ['embeddings', 'sae_features', 'sae_models', 'sae_alive_features', 'sae_normalization',
                 'outcomes', 'linear_probe_results', 'sae_linear_probe_results', 'phewas_results',
                 'autointerp_results', 'published_interpretations']:
        setattr(obj, attr, None)
    obj.pretrained_bundles = {column: load_pretrained(entry.name, cache_dir=Path(cache_dir) / 'zoo')}
    obj._measurement_bundle = entry.name
    obj._measurement_input_col = input_col
    obj._measurement_strict = strict
    obj._measurement_frame = None
    obj._measurement_key = None
    obj._measurement_encoder = None
    obj._measurement_embeddings_path = None
    return obj


def validate_code(bundle, code):
    import re
    from numbers import Integral
    from glyph.zoo.registry import resolve
    if code is None:
        return
    entry = resolve(bundle)
    if isinstance(code, str):
        match = re.fullmatch(r'([IRL])(\d+)', code.strip().upper())
        if not match:
            raise ValueError('Use a paper concept code such as I87, R64 or L127')
        expected = {'image': 'I', 'findings': 'R', 'labs': 'L'}[entry.modality]
        if match[1] != expected:
            raise KeyError(f'{code} is not in {entry.name}; expected {expected}-coded concepts')
        index = int(match[2])
    elif isinstance(code, Integral) and not isinstance(code, bool):
        index = int(code)
    else:
        raise ValueError('Use a paper code or an integer concept index')
    if not 0 <= index < entry.n_concepts:
        raise KeyError(f'Concept index must be between 0 and {entry.n_concepts - 1}')


def _fingerprint(frame, column, embeddings_path):
    value = frame.astype(str).to_json(orient='split') + str(embeddings_path)
    sources = list(frame[column]) if column in frame else []
    if embeddings_path is not None:
        paths = list(embeddings_path.values()) if isinstance(embeddings_path, dict) else [embeddings_path]
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                sources.extend(str(value) for value in path.glob('*_embeddings.pt'))
            else:
                sources.append(str(path))
    for raw in sources:
        if isinstance(raw, str):
            path = Path(raw)
            if path.is_file():
                stats = path.stat()
                value += str((path, stats.st_size, stats.st_mtime_ns))
    return hashlib.sha256(value.encode()).hexdigest()


def prepare(obj, metadata=None, *, sample_id=None, split=None, embeddings_path=None, recompute=False):
    from glyph.zoo.registry import resolve
    entry = resolve(obj._measurement_bundle)
    if metadata is None:
        if obj._measurement_frame is None:
            raise ValueError('Supply metadata on the first measure() call')
        frame = obj._measurement_frame.copy()
    else:
        frame = pd.read_csv(metadata, dtype={'sample_id': str}) if isinstance(metadata, (str, Path)) else metadata.copy()
    frame = frame.reset_index(drop=True)
    if 'sample_id' not in frame or frame.sample_id.isna().any():
        raise ValueError('Metadata needs nonmissing sample_id values')
    frame['sample_id'] = frame.sample_id.astype(str)
    if frame.sample_id.duplicated().any() or frame.empty:
        raise ValueError('Metadata sample_id values must be unique and the cohort nonempty')
    selected = frame
    if split is not None:
        if 'split' not in frame:
            raise ValueError('A split filter requires a split column in metadata')
        selected = selected[selected.split == split]
    if sample_id is not None:
        selected = selected[selected.sample_id == str(sample_id)]
    if selected.empty:
        raise KeyError('No studies match the requested sample_id/split')
    explicit = embeddings_path if embeddings_path is not None else obj._measurement_embeddings_path
    column = obj._measurement_input_col
    if column is None:
        matches = [name for name in COLUMNS[entry.modality] if name in frame]
        if len(matches) > 1:
            raise ValueError(f'Multiple input columns {matches}; specify input_col in from_pretrained')
        column = matches[0] if matches else COLUMNS[entry.modality][0]
    if explicit is None and column not in frame:
        raise ValueError(f'Metadata needs {column!r}; or supply embeddings_path')
    if explicit is None and selected[column].isna().any():
        raise ValueError('Requested studies have missing raw inputs')
    key = _fingerprint(frame, column, explicit)
    fresh = key != obj._measurement_key or recompute
    if fresh:
        obj.embeddings, obj.sae_features, obj.sae_models, obj.sae_alive_features = {}, {}, {}, {}
        obj.phewas_results = None
        obj._measurement_key = key
        obj._measurement_frame = frame.copy()
        obj._measurement_embeddings_path = explicit
        obj.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:10]
        obj.run_dir = obj._output_root / obj.run_id
        obj.run_dir.mkdir(parents=True, exist_ok=False)
        obj.cfg.data.output_dir = str(obj.run_dir)
        old = next(iter(obj.pretrained_bundles.values()), None) if getattr(obj, 'pretrained_bundles', None) else None
        obj.pretrained_bundles = {column: old} if old else {}
        obj.cfg.data.input_cols = [column]
        obj.cfg.data.embedding_map = {column: ENCODERS[entry.modality]}
        kwargs = next(iter(obj.cfg.data.embedding_model_kwargs.values()), {})
        obj.cfg.data.embedding_model_kwargs = {column: kwargs}
        frame.to_csv(obj.run_dir / 'measurement_metadata.csv', index=False)
        obj.cfg.data.metadata = str(obj.run_dir / 'measurement_metadata.csv')
    ids = obj.embeddings.get('cohort', {}).get('sample_ids', [])
    missing = selected[~selected.sample_id.isin(ids)].copy()
    if not missing.empty:
        manifest = dict(schema_version=1, run_id=obj.run_id, status='running',
                        mode='pretrained_measurement', bundle=entry.name, input_col=column,
                        strict=obj._measurement_strict, metadata_fingerprint=key, embeddings_path={key: str(value) for key, value in explicit.items()} if isinstance(explicit, dict) else str(explicit) if explicit is not None else None, outcome_types={}, embedding_source='explicit_files' if explicit is not None else 'raw_inputs')
        (obj.run_dir / 'run.json').write_text(json.dumps(manifest, indent=2))
        try:
            if explicit is not None:
                from glyph.extraction import load_embeddings
                missing['split'] = 'train'
                raw = load_embeddings(explicit, missing, obj.cfg.data.embedding_map, validate_splits=False)['train']['embeddings'][column]
            else:
                from glyph.models import get_model
                if obj._measurement_encoder is None:
                    options = OmegaConf.to_container(obj.cfg.data.embedding_model_kwargs[column], resolve=True)
                    obj._measurement_encoder = get_model(ENCODERS[entry.modality], device=('cuda' if torch.cuda.is_available() else 'cpu') if obj.cfg.device == 'auto' else obj.cfg.device, **options)
                chunks = []
                values = missing[column].tolist()
                size = obj.cfg.data.extraction_batch_size
                for start in range(0, len(values), size):
                    with torch.inference_mode():
                        chunks.append(obj._measurement_encoder.embed_batch(values[start:start + size]).detach().cpu())
                raw = torch.cat(chunks)
            if raw.shape != (len(missing), entry.input_dim) or not torch.isfinite(raw).all():
                raise ValueError(f'Expected one finite {entry.input_dim}-dimensional embedding per study')
            prior = obj.embeddings.get('cohort', {}).get('embeddings', {}).get(column)
            combined = raw if prior is None else torch.cat([prior, raw])
            combined_ids = ids + missing.sample_id.tolist()
            positions = {value: index for index, value in enumerate(combined_ids)}
            ordered_ids = [value for value in frame.sample_id if value in positions]
            obj.embeddings = {'cohort': dict(sample_ids=ordered_ids, embeddings={column: combined[[positions[value] for value in ordered_ids]]})}
            obj.outcomes = {'cohort': {}}
            obj.sae_features, obj.sae_models, obj.sae_alive_features = {}, {}, {}
            from glyph.core.runs import fit_dictionaries, save_run, complete_run, _redact
            fit_dictionaries(obj, {column: entry.name})
            obj._save_embeddings()
            obj._save_sae_features()
            save_run(obj)
            OmegaConf.save(OmegaConf.create(_redact(OmegaConf.to_container(obj.cfg, resolve=False))), obj.run_dir / 'config.yaml')
            manifest = json.loads((obj.run_dir / 'run.json').read_text())
            encoder = obj._measurement_encoder
            config = getattr(getattr(encoder, 'model', None), 'config', None)
            manifest['encoder'] = dict(name=ENCODERS[entry.modality], revision=getattr(config, '_commit_hash', None))
            manifest['embedding_distribution'] = {key: value.last_normalization_report for key, value in obj.pretrained_bundles.items()}
            (obj.run_dir / 'run.json').write_text(json.dumps(manifest, indent=2))
            complete_run(obj)
        except Exception:
            obj._measurement_key = None
            manifest['status'] = 'failed'
            (obj.run_dir / 'run.json').write_text(json.dumps(manifest, indent=2))
            raise
    return selected.sample_id.tolist()


def restore(obj, manifest):
    obj._measurement_bundle = manifest['bundle']
    obj._measurement_input_col = manifest['input_col']
    obj._measurement_strict = manifest.get('strict', True)
    obj._measurement_frame = pd.read_csv(obj.run_dir / 'measurement_metadata.csv', dtype={'sample_id': str})
    obj._measurement_embeddings_path = manifest.get('embeddings_path')
    obj._measurement_key = manifest.get('metadata_fingerprint') or _fingerprint(obj._measurement_frame, obj._measurement_input_col, obj._measurement_embeddings_path)
    obj._measurement_encoder = None
    obj.pretrained_bundles = {}
