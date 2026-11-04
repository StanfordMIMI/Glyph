"""Look up saved concept activations without losing dictionary identity."""
import re
from numbers import Integral

import numpy as np
import pandas as pd


PREFIXES = {'I': 'image', 'R': 'findings', 'L': 'labs'}


def _frame(obj, column, split):
    parts = obj.sae_features[column]
    if split is not None and split not in parts:
        raise KeyError(f'Unknown split {split!r}; available: {list(parts)}')
    names = [split] if split is not None else list(parts)
    arrays, ids = [], []
    for name in names:
        values = parts[name]
        arrays.append(values.detach().cpu().numpy() if hasattr(values, 'detach') else np.asarray(values))
        ids.extend(obj.embeddings[name]['sample_ids'])
    values = arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError('Saved sample IDs and concept activations must be aligned and unique')
    return pd.DataFrame(values, index=pd.Index(ids, name='sample_id'))


def measure(obj, code=None, *, split=None, modality=None):
    features = getattr(obj, 'sae_features', None)
    if not features:
        raise RuntimeError('Run fit() or load a saved run before looking up a concept')
    models = getattr(obj, 'sae_models', None) or {}
    if code is None:
        columns = [modality] if modality is not None else list(features)
        frames = []
        from glyph.zoo.registry import resolve
        for column in columns:
            if column not in features:
                raise KeyError(f'Unknown modality {column!r}; available: {list(features)}')
            frame = _frame(obj, column, split)
            bundle = getattr(models.get(column), 'selection_report', {}).get('pretrained')
            if bundle:
                kind = resolve(bundle).modality
                prefix = next(prefix for prefix, value in PREFIXES.items() if value == kind)
                frame.columns = [f'{prefix}{i}' for i in frame.columns]
            else:
                frame.columns = [f'{column}:Concept_{i}' for i in frame.columns]
            if frames and set(frame.index) != set(frames[0].index):
                raise ValueError('Modalities must have matching study IDs')
            frames.append(frame)
        result = pd.concat(frames, axis=1)
        if result.columns.duplicated().any():
            raise ValueError('Multiple dictionaries have the same concept codes; select modality= explicitly')
        return result.copy()
    if isinstance(code, str):
        match = re.fullmatch(r'([IRL])(\d+)', code.strip().upper())
        if not match:
            raise ValueError('Use a paper concept code such as I87, R64 or L92, or an integer feature index')
        prefix, number = match.groups()
        index = int(number)
        name = f'{prefix}{index}'
        from glyph.zoo.registry import resolve
        candidates = []
        for column, model in models.items():
            bundle = getattr(model, 'selection_report', {}).get('pretrained')
            if bundle and resolve(bundle).modality == PREFIXES[prefix]:
                candidates.append(column)
        if modality is not None:
            candidates = [column for column in candidates if column == modality]
        if not candidates:
            raise KeyError(
                f'{name} requires its published dictionary in this run; use '
                'fit(pretrained={input_column: bundle_name}). For a newly learned '
                'dictionary use an integer index and modality instead.')
    elif isinstance(code, Integral) and not isinstance(code, bool):
        index = int(code)
        if index < 0:
            raise ValueError('Concept indices must be nonnegative')
        name = f'Concept_{index}'
        if modality is None:
            candidates = list(features)
        else:
            candidates = [modality] if modality in features else []
        if not candidates:
            raise KeyError(f'Unknown modality {modality!r}; available: {list(features)}')
    else:
        raise ValueError('Use a paper concept code or an integer feature index')
    if len(candidates) != 1:
        raise ValueError(f'Concept lookup is ambiguous across {candidates}; specify modality= explicitly')
    column = candidates[0]
    frame = _frame(obj, column, split)
    if index >= frame.shape[1]:
        raise KeyError(f'{column!r} contains concept indices 0--{frame.shape[1] - 1}, not {index}')
    if column not in models:
        # Filtered CSV resumes fill omitted columns with zeros; those are unknown,
        # unlike actual zeros from a complete fitted dictionary.
        masks = [mask for (mod, _), mask in (getattr(obj, 'sae_alive_features', None) or {}).items() if mod == column]
        if len(masks) != 1 or index >= len(masks[0]) or not bool(masks[0][index]):
            raise KeyError(f'Concept {index} was not retained in the loaded concept CSV')
    activation = frame[index].copy()
    activation.name = name
    return activation
