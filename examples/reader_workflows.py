"""Public-data reader checks. Run with python -m examples.reader_workflows."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

from glyph import Glyph

UCI_URL = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"


def split_labels(y):
    indices = np.arange(len(y))
    train, held = train_test_split(indices, test_size=.30, stratify=y, random_state=42)
    val, test = train_test_split(held, test_size=.5, stratify=np.asarray(y)[held], random_state=42)
    splits = np.full(len(y), 'train', dtype=object)
    splits[val] = 'val'; splits[test] = 'test'
    return splits


def prepare_cancer(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    raw = urllib.request.urlopen(UCI_URL, timeout=60).read()
    (output / 'uci_wdbc.zip').write_bytes(raw)
    archive = zipfile.ZipFile(io.BytesIO(raw))
    name = next(name for name in archive.namelist() if name.endswith('wdbc.data'))
    data = pd.read_csv(io.BytesIO(archive.read(name)), header=None)
    reference = load_breast_cancer()
    x = data.iloc[:, 2:].to_numpy(dtype=np.float32)
    np.testing.assert_allclose(x, reference.data, rtol=1e-6)
    y = (data.iloc[:, 1] == 'M').astype(int).to_numpy()
    np.testing.assert_array_equal(y, 1 - reference.target)
    ids = [f'uci-{value}' for value in data.iloc[:, 0]]
    metadata = pd.DataFrame(dict(sample_id=ids, malignant=y, split=split_labels(y)))
    metadata.to_csv(output / 'metadata.csv', index=False)
    torch.save(dict(sample_ids=ids, embeddings=torch.from_numpy(x)), output / 'measurements.pt')
    pd.DataFrame(x, columns=reference.feature_names).assign(sample_id=ids).to_csv(output / 'measurements.csv', index=False)
    (output / 'provenance.json').write_text(json.dumps(dict(dataset='UCI Wisconsin Diagnostic Breast Cancer',
        url=UCI_URL, sha256=hashlib.sha256(raw).hexdigest(), rows=len(y), features=x.shape[1],
        representation='30 measured features supplied through the precomputed-vector interface; no foundation model',
        expected_checks=['cell size', 'concavity'], label='1=malignant; 0=benign'), indent=2))
    return output / 'metadata.csv', output / 'measurements.pt'


def prepare_digits(output, labelled=False):
    from PIL import Image
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    data = load_digits()
    selected, _ = train_test_split(np.arange(len(data.target)), train_size=400, stratify=data.target, random_state=42)
    y = (data.target[selected] >= 5).astype(int)
    paths = []
    for i, index in enumerate(selected):
        path = output / f'digit-{i:04}.png'
        Image.fromarray((data.images[index] * 255 / 16).astype(np.uint8)).save(path)
        paths.append(str(path))
    frame = pd.DataFrame(dict(sample_id=[f'digit-{i:04}' for i in range(len(selected))],
        image_path=paths, split=split_labels(y)))
    if labelled: frame['digit_ge_5'] = y
    frame.to_csv(output / 'metadata.csv', index=False)
    (output / 'provenance.json').write_text(json.dumps(dict(dataset='scikit-learn / UCI optical digits',
        source='https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html',
        rows=len(frame), selected_indices=selected.tolist(), labelled=labelled), indent=2))
    return output / 'metadata.csv'



def prepare_synthetic_labs(output):
    from glyph.extraction.labs import LabsEncoder, prepare_lab_inputs
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    encoder = LabsEncoder(device='cpu', cache_dir=str(Path('cache/huggingface').resolve()), token=False)
    stats = encoder.metadata
    numeric = stats[stats.retained & stats.is_numeric & (stats['std'] > 0)].copy()
    # Select numeric tests with valid positive ranges, solely from released model metadata.
    numeric = numeric[(numeric['mean'] > 0) & numeric.training_unit.notna()].head(24)
    rng = np.random.default_rng(42)
    latent = rng.normal(size=240)
    y = (latent + rng.normal(scale=.5,size=240) > 0).astype(int)
    metadata = pd.DataFrame(dict(sample_id=[f'synthetic-labs-{i:04}' for i in range(240)],
        patient_id=[f'new-synthetic-person-{i:04}' for i in range(240)], date='2026-07-01',
        synthetic_outcome=y, split=split_labels(y)))
    rows=[]
    for i, patient in enumerate(metadata.patient_id):
        for j, row in enumerate(numeric.itertuples()):
            z=.8*latent[i]*(1 if j%2 else -1)+rng.normal(scale=.5)
            value=max(.01,float(row.mean + row.std*np.clip(z,-2,2)))
            rows.append(dict(patient_id=patient,date='2026-05-15',lab_id=int(row.lab_id),
                value=round(value,4),unit=row.training_unit))
    measurements=pd.DataFrame(rows)
    metadata.to_csv(output/'study_metadata.csv',index=False)
    measurements.to_csv(output/'measurements.csv',index=False)
    path=prepare_lab_inputs(metadata,measurements,encoder.bundle_dir,output/'prepared')
    (output/'provenance.json').write_text(json.dumps(dict(dataset='Entirely synthetic lab histories',seed=42,
        encoder_repo='RobbieHolland/Glyph-labs-encoder',patients=240,measurements=len(rows),
        note='No research cohort rows. Outcomes are artificial. Released concept interpretations are suggestions only.'),indent=2))
    return path


def make_glyph(metadata, root, modality, encoder, outcomes):
    return Glyph('config', [f'data.metadata={metadata}', f'data.output_dir={root}',
        f'data.cache_dir={Path("cache").resolve()}', f'data.input_cols=[{modality}]',
        'data.outcome_cols=[' + ','.join(outcomes) + ']',
        f'data.embedding_map={{{modality}:{encoder}}}', 'data.num_workers=0',
        'data.extraction_batch_size=32', 'device=cpu', 'sae.top_ks=[4,8,16]',
        'sae.matryoshka=[64]', 'sae.max_steps=1000', 'sae.learning_rate=0.001',
        'sae.selection.lookahead_step=2', 'phewas.min_activations=15',
        'autointerp.llm.gemini_api_key=null'])


def check_known_signals(obj, measurements):
    """Describe concepts using training correlations; evaluate these on held-out data."""
    frame = pd.read_csv(measurements).set_index('sample_id')
    k = obj.sae_models['measurements'].selected_top_k
    rows = []
    for feature in np.where(np.asarray(obj.sae_alive_features[('measurements', k)]))[0]:
        correlations = {}
        for split in ['train', 'test']:
            x = frame.loc[obj.embeddings[split]['sample_ids']].to_numpy()
            f = obj.sae_features['measurements'][split][:, feature].numpy()
            correlations[split] = np.array([np.corrcoef(f, x[:, j])[0, 1] if np.std(f) > 0 else np.nan for j in range(x.shape[1])])
        best = int(np.nanargmax(np.abs(correlations['train'])))
        rows.append(dict(feature_idx=int(feature), strongest_training_measurement=frame.columns[best],
            train_correlation=float(correlations['train'][best]), test_correlation=float(correlations['test'][best])))
    result = pd.DataFrame(rows)
    result.to_csv(obj.run_dir / 'known_signal_checks.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', choices=['cancer', 'images', 'images-labelled', 'labs'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if args.workflow == 'cancer':
        metadata, embeddings = prepare_cancer(root / 'inputs')
        obj = make_glyph(metadata, root / 'runs', 'measurements', 'precomputed', ['malignant'])
        obj.fit(embeddings_path=embeddings)
        obj.hypothesis_search()
        check_known_signals(obj, root / 'inputs/measurements.csv')
    elif args.workflow == 'labs':
        metadata=prepare_synthetic_labs(root/'inputs')
        obj=make_glyph(metadata,root/'runs','labs_path','labs',['synthetic_outcome'])
        obj.fit(pretrained={'labs_path':'labs-loinc'})
        obj.hypothesis_search()
    else:
        labelled = args.workflow == 'images-labelled'
        metadata = prepare_digits(root / 'inputs', labelled=labelled)
        obj = make_glyph(metadata, root / 'runs', 'image_path', 'resnet18', ['digit_ge_5'] if labelled else [])
        obj.fit()
        if labelled: obj.hypothesis_search()
    obj.visualize()
    # Reload the actual saved dictionary, with no encoder or refitting.
    loaded = Glyph.load_run(obj.run_dir, device='cpu')
    for split, data in obj.embeddings.items():
        for modality in data['embeddings']:
            torch.testing.assert_close(loaded.encode_embeddings(data['embeddings'][modality], modality=modality),
                obj.sae_features[modality][split], rtol=0, atol=0)
    (root / 'result.json').write_text(json.dumps(dict(run=str(obj.run_dir), seconds=time.monotonic()-started,
        workflow=args.workflow, reload_identical=True), indent=2))
    print((root / 'result.json').read_text())


if __name__ == '__main__': main()
