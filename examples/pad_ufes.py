"""New-user DINOv3 skin-lesion demo with an eight-field published clinical baseline."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import requests
from sklearn.model_selection import train_test_split

from glyph import Glyph

MODEL = 'facebook/dinov3-vits16-pretrain-lvd1689m'
REVISION = '114c1379950215c8b35dfcd4e90a5c251dde0d32'
SOURCE = 'https://data.mendeley.com/datasets/zr7vgbcyr2/1'
BASE = 'https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/'
FILES = [
    ('metadata.csv', 'fa850265-57da-48f0-ba3e-998b3e44b1f6', '14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527'),
    ('imgs_part_1.zip', '1cc2f71f-20a2-412d-b746-a9b9bc20c966', '0ab44f60938bf57445e12f518a8878954cc734e6b0aec6d01194e2d26b4b2dca'),
    ('imgs_part_2.zip', '559a60ed-5504-475d-996c-6a8bc253b5e7', 'e2d9a3cbd58e823f5ae33163c48643e7d1b54ae3f9e145f01f8e9f16a363a60b'),
    ('imgs_part_3.zip', '34dcdf8e-e5f1-4b35-aa0b-5135051ff852', 'ecc4ef10143a43e1d01cb736773148607a78b530417eb76f2c38ad24bf5d0d2c'),
]
DIAGNOSES = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
OUTCOMES = ['skin_cancer'] + [f'is_{d.lower()}' for d in DIAGNOSES]
# Display names never replace the source outcome columns.
OUTCOME_NAMES = {
    'skin_cancer': 'Skin cancer (any type)',
    'is_ack': 'Actinic keratosis',
    'is_bcc': 'Basal cell carcinoma',
    'is_mel': 'Melanoma',
    'is_nev': 'Melanocytic nevus (mole)',
    'is_scc': 'Squamous cell carcinoma',
    'is_sek': 'Seborrheic keratosis',
}
OUTCOME_LABELS = {
    'skin_cancer': {'0': 'No skin cancer', '1': 'Skin cancer'},
    'is_ack': {'0': 'Other diagnosis', '1': 'Actinic keratosis'},
    'is_bcc': {'0': 'Other diagnosis', '1': 'Basal cell carcinoma'},
    'is_mel': {'0': 'Other diagnosis', '1': 'Melanoma'},
    'is_nev': {'0': 'Other diagnosis', '1': 'Melanocytic nevus (mole)'},
    'is_scc': {'0': 'Other diagnosis', '1': 'Squamous cell carcinoma'},
    'is_sek': {'0': 'Other diagnosis', '1': 'Seborrheic keratosis'},
}
REGIONS = ['ABDOMEN', 'ARM', 'BACK', 'CHEST', 'EAR', 'FACE', 'FOOT', 'FOREARM', 'HAND', 'LIP', 'NECK', 'NOSE', 'SCALP', 'SHIN', 'THIGH']
SYMPTOMS = ['itch', 'bleed', 'hurt', 'grew', 'changed', 'elevation']
CLINICAL_SOURCE = 'https://arxiv.org/html/1909.12912'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(spec, folder):
    name, file_id, expected = spec
    path = Path(folder) / name
    if path.exists() and sha(path) == expected:
        return path
    for attempt in range(3):
        try:
            with requests.get(BASE + file_id + '/file_downloaded', stream=True, timeout=(20, 120)) as r:
                r.raise_for_status()
                with path.with_suffix(path.suffix + '.partial').open('wb') as f:
                    for block in r.iter_content(4 * 1024 * 1024):
                        f.write(block)
            temporary = path.with_suffix(path.suffix + '.partial')
            if sha(temporary) != expected:
                raise ValueError(f'Published checksum mismatch for {name}')
            temporary.replace(path)
            print('Downloaded and verified', name, flush=True)
            return path
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def select_patients(frame):
    required = {'patient_id', 'lesion_id', 'img_id', 'diagnostic', 'age', 'region', *SYMPTOMS}
    if required - set(frame) or frame.img_id.duplicated().any() or frame.patient_id.isna().any():
        raise ValueError('Invalid source metadata schema or IDs')
    if not set(frame.diagnostic).issubset(DIAGNOSES):
        raise ValueError('Unexpected diagnosis in the pinned release')
    # Outcome-blind, deterministic choice; no repeated patients or photographs.
    result = frame.sort_values(['patient_id', 'img_id']).drop_duplicates('patient_id').copy().reset_index(drop=True)
    train, held = train_test_split(np.arange(len(result)), test_size=.30, stratify=result.diagnostic, random_state=42)
    val, test = train_test_split(held, test_size=.5, stratify=result.diagnostic.iloc[held], random_state=42)
    result['split'] = 'train'; result.loc[val, 'split'] = 'val'; result.loc[test, 'split'] = 'test'
    result['sample_id'] = result.img_id.str.removesuffix('.png')
    result['skin_cancer'] = result.diagnostic.isin(['BCC', 'SCC', 'MEL']).astype(int)
    for d in DIAGNOSES:
        result[f'is_{d.lower()}'] = (result.diagnostic == d).astype(int)
    return result


def clinical_measurements(frame):
    result = frame[['sample_id', 'age']].copy()
    for column in SYMPTOMS:
        result[column] = frame[column].astype(str).str.upper().map({'TRUE': 1., 'FALSE': 0.})
    known = frame.region.isin(REGIONS)
    for region in REGIONS:
        result['site_' + region.lower()] = np.where(known, (frame.region == region).astype(float), np.nan)
    return result


def prepare(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    raw = output / 'source'; raw.mkdir(exist_ok=True)
    original = pd.read_csv(download(FILES[0], raw))
    selected = select_patients(original)
    images = output / 'images'; images.mkdir(exist_ok=True)
    wanted = set(selected.img_id)
    with ThreadPoolExecutor(max_workers=3) as pool:
        archives = list(pool.map(lambda spec: download(spec, raw), FILES[1:]))
    hashes = []
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                name = Path(member.filename).name
                if name not in wanted:
                    continue
                path = images / name
                # Always reconstruct from the verified archive when preparing.
                path.write_bytes(z.read(member))
                with Image.open(path) as image:
                    image.verify()
                hashes.append(dict(sample_id=name.removesuffix('.png'), filename=name, source_archive=archive.name, sha256=sha(path)))
    if set(h['filename'] for h in hashes) != wanted or len(hashes) != len(wanted):
        raise ValueError('Images do not match the selected patients exactly')
    selected['image_path'] = [str(images / name) for name in selected.img_id]
    selected.to_csv(output / 'metadata.csv', index=False)
    clinical_measurements(selected).to_csv(output / 'measurements.csv', index=False)
    pd.DataFrame(hashes).to_csv(output / 'image_provenance.csv', index=False)
    provenance = dict(dataset='PAD-UFES-20 v1', source=SOURCE, doi='10.17632/zr7vgbcyr2.1', license='CC BY 4.0',
        source_images=len(original), source_patients=int(original.patient_id.nunique()), source_lesions=int(original.lesion_id.nunique()),
        selection='Lexicographically first image ID per patient, chosen without using diagnosis; one observation per patient',
        endpoint='Cancer in the imaged lesion (BCC, SCC, MEL) versus ACK, NEV, SEK; six secondary one-vs-rest diagnoses. Diagnostic, not prognostic.',
        label_verification='All cancer diagnoses biopsy-proven; some non-cancer diagnoses use dermatologist consensus. Biopsy status excluded from predictors.',
        split_unit='patient', seed=42, n=len(selected), model=MODEL, encoder_revision=REVISION,
        split_counts={split: dict(n=len(part), **{outcome: int(part[outcome].sum()) for outcome in OUTCOMES}) for split, part in selected.groupby('split')},
        diagnosis_counts=pd.crosstab(selected.split, selected.diagnostic).to_dict(),
        outcome_names=OUTCOME_NAMES,
        source_checksums={name: expected for name, _, expected in FILES},
        pixel_preprocessing='Original PNGs passed directly to the official DINOv3 image processor; no task-specific crop or lesion segmentation.')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(selected.groupby('split')[OUTCOMES].sum().to_string(), flush=True)
    return output / 'metadata.csv'


def make_glyph(metadata, output):
    # Scientific fitting, sparsity search, association and graph settings are defaults.
    return Glyph('config', config_overrides=[
        f'data.metadata={Path(metadata).resolve()}', 'data.input_cols=[image_path]',
        'data.outcome_cols=[' + ','.join(OUTCOMES) + ']',
        f'data.embedding_map={{image_path: "hf:{MODEL}"}}',
        f'data.embedding_model_kwargs={{image_path: {{input_type: image, pooling: cls, revision: "{REVISION}"}}}}',
        f'data.output_dir={Path(output).resolve()}',
        # Runtime-only conveniences; do not change any scientific selection rule.
        'data.num_workers=0', 'data.extraction_batch_size=16', 'autointerp.enabled=false',
        'visualization.outcome_names={'+','.join(
            key+': '+json.dumps(value) for key,value in OUTCOME_NAMES.items())+'}'])


def baseline_manifest():
    definitions = [('age', 'Recorded patient age, in years', 'years')]
    definitions += [(c, f'Patient-reported lesion {c}; unknown or unavailable remains missing', 'binary 0/1') for c in SYMPTOMS]
    definitions += [('site_' + r.lower(), f'Recorded lesion location is {r}; one-hot encoding of the published body-site variable', 'binary 0/1') for r in REGIONS]
    return dict(title='Glyph + DINOv3: smartphone skin-lesion diagnosis', outcome_column='y',
        scope='PAD-UFES-20, one original photograph per patient, patient-held-out internal validation',
        markers=[dict(id=c, column=c, record_id='pmid:31760271', definition=d, units=u, proxy=False,
            citation='Pacheco and Krohling, The impact of patient clinical information on automated skin cancer detection, Computers in Biology and Medicine 2020',
            source_url=CLINICAL_SOURCE, doi='10.1016/j.compbiomed.2019.103545', formula=dict(op='identity', column=c)) for c, d, u in definitions],
        glyph_selection_rule='All training-active concepts; capped variants ranked using training-only two-sided logistic score tests',
        comparison_top_n=10, combined_top_n_strategy='pooled', report_comparison_split='test',
        caveats=['Eight clinical fields encoded as 22 numerical predictors; top-ten arms rank encoded predictors.',
            'Clinical coefficients are fitted on the new training split, not copied from a published classifier.',
            'Contemporaneous diagnosis rather than prognosis; no external validation.',
            'Clinical and histological reference standards differ among non-cancer cases.',
            'No automated semantic interpretation or confounder adjustment.'])




def freeze_baseline(output):
    """Retrieve/review the fixed primary source and freeze definitions before evaluation."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    target = output / 'frozen_manifest_pooled.json'
    if target.exists():
        frozen = json.loads(target.read_text())
        if frozen['markers'] != baseline_manifest()['markers'] or frozen.get('combined_top_n_strategy') != 'pooled':
            raise ValueError('Cached baseline differs; use a new output directory')
        return target
    search = output / 'literature_search'
    protocol = Path(__file__).with_name('pad_ufes_literature_search.json')
    command = [sys.executable, '-B', '-m', 'glyph', 'harness']
    subprocess.run(command + ['literature-search', '--protocol', str(protocol), '--output', str(search)], check=True)
    papers = json.loads((search / 'papers.json').read_text())
    if len(papers) != 1 or papers[0]['pmid'] != '31760271' or papers[0]['doi'] != '10.1016/j.compbiomed.2019.103545':
        raise ValueError('Retrieved record differs from the reviewed clinical-model source')
    ledger = pd.read_csv(search / 'screening.csv', keep_default_na=False)
    ledger['title_decision'] = 'include'; ledger['abstract_decision'] = 'include'; ledger['feasible'] = 'yes'
    ledger['verified_source_url'] = CLINICAL_SOURCE
    ledger['reason'] = 'Pre-reviewed full text section 2.1 specifies age, body site and six symptoms. All eight can be reconstructed directly. New ridge coefficients, not the historical fitted neural classifier.'
    ledger.to_csv(search / 'screening.csv', index=False)
    manifest = baseline_manifest()
    manifest['glyph_features'] = [f'concept_{i}' for i in range(8192)]
    manifest['glyph_selection_rule'] = 'All 8192 dictionary columns fixed before extraction or evaluation; constant predictors do not affect the fit. Capped variants ranked using training-only two-sided logistic score tests.'
    manifest['outcome_labels'] = {'skin_cancer': {'0': 'ACK / NEV / SEK', '1': 'BCC / SCC / MEL'}}
    candidate = output / 'candidate_manifest.json'; candidate.write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run(command + ['literature-freeze', '--search', str(search), '--manifest', str(candidate), '--output', str(target)], check=True)
    return target


def literature_study(obj, inputs, output, frozen_manifest, search, bootstrap=2000):
    """Evaluate all six frozen predictor arms via the installed agent harness."""
    inputs, output = Path(inputs), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(inputs / 'metadata.csv')
    metadata[['sample_id', 'patient_id', 'split', 'skin_cancer']].rename(columns={'skin_cancer': 'y'}).to_csv(output / 'cohort.csv', index=False)
    blocks = []
    for split, data in obj.embeddings.items():
        block = pd.DataFrame(obj.sae_features['image_path'][split].cpu().numpy(), columns=[f'concept_{i}' for i in range(8192)])
        block.insert(0, 'sample_id', data['sample_ids']); blocks.append(block)
    pd.concat(blocks, ignore_index=True).to_csv(output / 'concepts.csv', index=False)
    obj.phewas_results.query("outcome == 'skin_cancer'").to_csv(output / 'glyph_associations.csv', index=False)
    subprocess.run([sys.executable, '-B', '-m', 'glyph', 'harness', 'literature-study',
        '--cohort', str(output / 'cohort.csv'), '--measurements', str(inputs / 'measurements.csv'),
        '--glyph', str(output / 'concepts.csv'), '--glyph-associations', str(output / 'glyph_associations.csv'),
        '--manifest', str(frozen_manifest), '--search', str(search), '--run', str(obj.run_dir),
        '--report', '--image-examples', '--bootstrap', str(bootstrap), '--output', str(output / 'study')], check=True)
    return output / 'study'



def report(run, output, literature):
    output = Path(output)
    metadata = output.parent / 'report_metadata.json'
    metadata.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(scope='PAD-UFES-20: 1,373 patients, one original photograph each, seven diagnostic outcomes',
        outcome_names=OUTCOME_NAMES, outcome_labels=OUTCOME_LABELS,
        interpretation_status='Automated interpretation was not run.',
        notes='Patient-held-out diagnostic benchmark, not prognosis or external validation. '
              'The clinical baseline refits eight published inputs; it is not the historical fitted classifier. '
              'Some non-cancer labels use clinical consensus. Melanoma has only five test examples. '
              'Automated interpretation and confounder adjustment were not run. '
              'Images: Pacheco et al., PAD-UFES-20 v1, doi:10.17632/zr7vgbcyr2.1, CC BY 4.0; resized for display.')
    metadata.write_text(json.dumps(profile, indent=2) + '\n')
    subprocess.run([sys.executable, '-B', '-m', 'glyph', 'harness', 'report', '--run', str(run),
        '--literature', str(Path(literature) / 'comparison'), '--metadata', str(metadata),
        '--image-examples', '--preview-html', '--output', str(output),
        '--title', 'Glyph + DINOv3: smartphone skin-lesion diagnosis'], check=True)
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'fit'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
    else:
        if args.metadata is None:
            parser.error('--metadata is required for fit')
        obj = make_glyph(args.metadata, args.output)
        obj.fit(); obj.hypothesis_search(); obj.visualize(split='train')
        print('RUN', obj.run_dir, flush=True)
