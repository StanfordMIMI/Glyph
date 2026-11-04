"""Public DINOv3 prognosis demo using TCIA's Stony Brook COVID-19 radiographs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import requests
from sklearn.model_selection import train_test_split

from glyph import Glyph
from examples.dinov3_digits import MODEL, REVISION

COLLECTION_URL = 'https://www.cancerimagingarchive.net/collection/covid-19-ny-sbu/'
CLINICAL_URL = ('https://www.cancerimagingarchive.net/wp-content/uploads/'
                'deidentified_overlap_tcia.csv.cleaned.csv_20210806.csv')
IDC_VERSION = '25.0.5'
CLINICAL_SHA256 = '6248f0c37db20d6102d8781b6d86cfeccd5e275635b3ad75d23b42026460264a'


def baseline_cohort(series, clinical):
    frontal = series[(series.BodyPartExamined == 'CHEST') &
                     series.SeriesDescription.isin(['AP', 'PA']) &
                     series.Modality.isin(['CR', 'DX'])].copy()
    if clinical.to_patient_id.duplicated().any():
        raise ValueError('Clinical table must contain one row per patient')
    joined = frontal.merge(clinical, left_on='PatientID', right_on='to_patient_id',
                           validate='many_to_one')
    admission = pd.to_datetime(joined.visit_start_datetime, format='%m/%d/%Y').dt.normalize()
    image = pd.to_datetime(joined.StudyDate, format='%Y-%m-%d').dt.normalize()
    day = (image - admission).dt.days
    # A calendar-day landmark: hospital day two. Every selected image is earlier,
    # and both discharge and death occur on/after that next-day landmark.
    eligible = (day.between(-1, 0) & (joined.length_of_stay >= 2) &
                (joined.visit_concept_name == 'Inpatient Visit') &
                joined['last.status'].isin(['deceased', 'discharged']))
    selected = joined.loc[eligible].copy()
    selected['image_day_relative_to_admission'] = day.loc[eligible]
    selected = selected.sort_values(['PatientID', 'image_day_relative_to_admission',
                                     'SeriesInstanceUID']).drop_duplicates('PatientID')
    selected['in_hospital_death'] = (selected['last.status'] == 'deceased').astype(int)
    if selected.empty:
        raise ValueError('No eligible admission-period frontal radiographs')
    return selected.reset_index(drop=True)


def get(url, **kwargs):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 45), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def to_rgb(dataset, size=512):
    pixels = dataset.pixel_array
    if pixels.ndim != 2:
        raise ValueError('Expected one monochrome radiograph')
    valid = np.isfinite(pixels)
    if hasattr(dataset, 'PixelPaddingValue'):
        start = dataset.PixelPaddingValue
        end = getattr(dataset, 'PixelPaddingRangeLimit', start)
        valid &= ~((pixels >= min(start, end)) & (pixels <= max(start, end)))
    values = pixels.astype(np.float32) * float(getattr(dataset, 'RescaleSlope', 1))
    values += float(getattr(dataset, 'RescaleIntercept', 0))
    if not valid.any():
        raise ValueError('No non-padding image pixels')
    low, high = np.percentile(values[valid], [.5, 99.5])
    if not high > low:
        raise ValueError('Radiograph has no usable intensity range')
    scaled = np.clip((values - low) / (high - low), 0, 1)
    if dataset.PhotometricInterpretation == 'MONOCHROME1':
        scaled = 1 - scaled
    elif dataset.PhotometricInterpretation != 'MONOCHROME2':
        raise ValueError('Unsupported image photometry')
    scaled[~valid] = 0
    image = Image.fromarray(np.rint(scaled * 255).astype(np.uint8)).convert('RGB')
    # Preserve the entire image aspect ratio; the HF processor receives a square.
    return ImageOps.pad(image, (size, size), method=Image.Resampling.LANCZOS, color=0)


def download_patient(row, folder):
    import pydicom
    sample_id = str(row.PatientID)
    image_path, record_path = folder / f'{sample_id}.png', folder / f'{sample_id}.json'
    if image_path.is_file() and record_path.is_file():
        record = json.loads(record_path.read_text())
        if record['SeriesInstanceUID'] != row.SeriesInstanceUID:
            raise ValueError('Cached image belongs to a different selected series')
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != record['png_sha256']:
            raise ValueError('Cached image checksum differs')
        return record
    if int(row.instanceCount) != 1:
        raise ValueError('Expected one radiograph per series')
    endpoint = f'https://{row.aws_bucket}.s3.amazonaws.com/'
    listing = get(endpoint, params={'list-type': '2', 'prefix': row.crdc_series_uuid + '/'}).content
    keys = [item.text for item in ET.fromstring(listing).findall('.//{*}Key')]
    if len(keys) != 1:
        raise ValueError('Expected exactly one public DICOM object')
    url = endpoint + keys[0]
    raw = get(url).content
    dataset = pydicom.dcmread(io.BytesIO(raw))
    if str(dataset.PatientID) != sample_id or str(dataset.SeriesInstanceUID) != row.SeriesInstanceUID:
        raise ValueError('Downloaded DICOM patient or series does not match the index')
    to_rgb(dataset).save(image_path)
    record = dict(sample_id=sample_id, SeriesInstanceUID=row.SeriesInstanceUID,
        StudyInstanceUID=str(dataset.StudyInstanceUID), SOPInstanceUID=str(dataset.SOPInstanceUID),
        source_url=url, dicom_sha256=hashlib.sha256(raw).hexdigest(), downloaded_bytes=len(raw),
        png_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        original_rows=int(dataset.Rows), original_columns=int(dataset.Columns),
        view=str(getattr(dataset, 'ViewPosition', row.SeriesDescription)),
        pixel_preprocessing='Non-padding 0.5/99.5 percentiles, photometry, RGB, aspect-preserving 512-square padding')
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    return record


def prepare(output, workers=8):
    import idc_index_data
    if importlib.metadata.version('idc-index-data') != IDC_VERSION:
        raise ValueError(f'This reproducible demo requires idc-index-data=={IDC_VERSION}')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = output / 'images'; images.mkdir(exist_ok=True)
    clinical_path = output / 'clinical_source.csv'
    if not clinical_path.exists():
        clinical_path.write_bytes(get(CLINICAL_URL).content)
    if hashlib.sha256(clinical_path.read_bytes()).hexdigest() != CLINICAL_SHA256:
        raise ValueError('Clinical release checksum changed; verify the dated source before rerunning')
    clinical = pd.read_csv(clinical_path)
    index_path = Path(idc_index_data.__file__).parent / 'idc_index.parquet'
    # Predicate pushdown avoids loading the entire public IDC index in memory.
    series = pd.read_parquet(index_path, filters=[('collection_id', '=', 'covid_19_ny_sbu'),
                                                ('Modality', 'in', ['CR', 'DX'])])
    cohort = baseline_cohort(series, clinical)
    cohort.to_csv(output / 'selected_series.csv', index=False)
    print(f'Downloading {len(cohort)} patients; deaths={cohort.in_hospital_death.sum()}; '
          f'public DICOM transfer about {cohort.series_size_MB.sum()/1024:.1f} GB', flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_patient, row, images) for row in cohort.itertuples()]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 50 == 0 or len(records) == len(cohort):
                print(f'Converted {len(records)}/{len(cohort)} radiographs', flush=True)
    ids = np.arange(len(cohort)); labels = cohort.in_hospital_death.to_numpy()
    train, held = train_test_split(ids, test_size=.30, stratify=labels, random_state=42)
    val, test = train_test_split(held, test_size=.5, stratify=labels[held], random_state=42)
    split = np.full(len(cohort), 'train', dtype=object); split[val] = 'val'; split[test] = 'test'
    metadata = pd.DataFrame(dict(sample_id=cohort.PatientID, patient_id=cohort.PatientID,
        image_path=[str(images / f'{sample}.png') for sample in cohort.PatientID],
        in_hospital_death=labels, split=split))
    metadata.to_csv(output / 'metadata.csv', index=False)
    pd.DataFrame(records).sort_values('sample_id').to_csv(output / 'image_provenance.csv', index=False)
    provenance = dict(dataset='TCIA COVID-19-NY-SBU', source=COLLECTION_URL,
        doi='10.7937/TCIA.BBAG-2923', license='CC BY 4.0', clinical_url=CLINICAL_URL,
        clinical_sha256=hashlib.sha256(clinical_path.read_bytes()).hexdigest(),
        idc_index_version=IDC_VERSION, idc_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        published_patients=len(clinical), available_xray_patients=int(series.PatientID.nunique()),
        eligible_patients=len(cohort), deaths=int(labels.sum()),
        split_counts=metadata.groupby('split').agg(n=('sample_id', 'size'), deaths=('in_hospital_death', 'sum')).to_dict('index'),
        image_selection='One AP/PA chest series per patient from the day before or day of the selected inpatient admission; earliest available date, then lexical series UID',
        target='In-hospital death after the admission day among patients with stays spanning at least two calendar days; discharged controls',
        timing='Calendar-day hospital-day-two landmark; exact death timestamps are unavailable. ICU/ventilation flags are not treated as future endpoints.',
        split_unit='patient', model=MODEL, encoder_revision=REVISION, seed=42,
        raw_dicom_bytes_transferred=sum(r['downloaded_bytes'] for r in records),
        raw_dicom_stored=False)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2), flush=True)
    return output / 'metadata.csv'


def make_glyph(metadata, output):
    return Glyph('config', config_overrides=[
        f'data.metadata={Path(metadata).resolve()}', 'data.input_cols=[image_path]',
        'data.outcome_cols=[in_hospital_death]',
        f'data.embedding_map={{image_path: "hf:{MODEL}"}}',
        f'data.embedding_model_kwargs={{image_path: {{input_type: image, pooling: cls, revision: "{REVISION}"}}}}',
        f'data.output_dir={Path(output).resolve()}', f'data.cache_dir={Path("cache").resolve()}',
        'data.num_workers=0', 'data.extraction_batch_size=16', 'device=cpu',
        'sae.matryoshka=[8192]', 'sae.top_ks=[8,16,32,64,128,256,512,1024,2048,4096,8192]',
        'sae.max_steps=2000', 'sae.learning_rate=0.001',
        'sae.selection.lookahead_step=128', 'phewas.min_activations=25', 'visualization.correlation_percentile=0.9995', 'autointerp.enabled=false'])


def example_panels(obj, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    frame = pd.read_csv(obj.cfg.data.metadata).set_index('sample_id')
    train_ids = np.asarray(obj.embeddings['train']['sample_ids'])
    rng = np.random.default_rng(42)
    samples = rng.choice(train_ids, size=6, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(9, 7))
    for ax, sample in zip(axes.flat, samples):
        ax.imshow(Image.open(frame.loc[sample, 'image_path']))
        ax.axis('off')
    fig.suptitle('Public admission-period chest radiographs', fontsize=15)
    fig.tight_layout(); fig.savefig(output / 'inputs.png', dpi=140); plt.close(fig)
    train = obj.phewas_results.query("split == 'train'").copy()
    train = train[np.isfinite(train.odds_ratio_p_value)].sort_values('odds_ratio_p_value')
    chosen = train.drop_duplicates('feature_idx').head(3)
    if chosen.empty:
        return
    fig, axes = plt.subplots(len(chosen), 6, figsize=(12, 3 * len(chosen)), squeeze=False)
    for row_index, row in enumerate(chosen.itertuples()):
        column = 0
        for split, count in [('train', 4), ('test', 2)]:
            activations = obj.sae_features['image_path'][split][:, int(row.feature_idx)].cpu().numpy()
            ids = np.asarray(obj.embeddings[split]['sample_ids'])
            for position in np.argsort(-activations, kind='stable')[:count]:
                ax = axes[row_index, column]; record = frame.loc[ids[position]]
                ax.imshow(Image.open(record.image_path)); ax.axis('off')
                outcome = 'later died' if record.in_hospital_death else 'discharged'
                ax.set_title(f'{split} / {outcome}', fontsize=9)
                column += 1
        axes[row_index, 0].text(-.06, .5, f'Concept {int(row.feature_idx)}',
            transform=axes[row_index, 0].transAxes, ha='right', va='center', fontsize=9)
    fig.suptitle('Highest activating images for training-selected concepts\nNo semantic interpretations were generated; test examples are illustrative', fontsize=13)
    fig.tight_layout(rect=(.075, 0, 1, .92))
    fig.savefig(output / 'concept_examples.png', dpi=150); plt.close(fig)


def literature_study(obj, inputs, output, search=None, bootstrap=1000):
    import subprocess
    import sys
    inputs, output = Path(inputs), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    search = Path(search) if search else output.parent / 'literature_search'
    subprocess.run([sys.executable, '-B', 'skills/glyph-literature/scripts/discover.py',
        '--protocol', 'examples/sbu_cxr_literature_search.json', '--output', str(search)], check=True)
    papers = json.loads((search / 'papers.json').read_text())
    if len(papers) != 1 or papers[0]['pmid'] != '32640463' or papers[0]['doi'] != '10.1038/s41586-020-2521-4':
        raise ValueError('Targeted retrieval does not match the pre-reviewed primary source')
    source = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7611074/'
    ledger = pd.read_csv(search / 'screening.csv', keep_default_na=False)
    ledger['title_decision'] = 'include'; ledger['abstract_decision'] = 'include'
    ledger['feasible'] = 'yes'; ledger['verified_source_url'] = source
    ledger['reason'] = ('Fixed pre-reviewed demo source: age and sex supported; use locally available '
                        'age bands as proxies. Not the published fitted model or a systematic review.')
    ledger.to_csv(search / 'screening.csv', index=False)
    metadata = pd.read_csv(obj.cfg.data.metadata)
    clinical = pd.read_csv(inputs / 'selected_series.csv').set_index('PatientID').loc[metadata.sample_id]
    measurements = pd.DataFrame(dict(sample_id=metadata.sample_id,
        age_60_74=(clinical['age.splits'] == '(59,74]').to_numpy(dtype=float),
        age_75_plus=(clinical['age.splits'] == '(74,90]').to_numpy(dtype=float),
        male=clinical.gender_concept_name.map({'MALE': 1., 'FEMALE': 0.}).to_numpy()))
    known_age = clinical['age.splits'].isin(['[18,59]', '(59,74]', '(74,90]']).to_numpy()
    measurements.loc[~known_age, ['age_60_74', 'age_75_plus']] = np.nan
    cohort = metadata[['sample_id', 'patient_id', 'split', 'in_hospital_death']].rename(columns={'in_hospital_death': 'y'})
    cohort.to_csv(output / 'cohort.csv', index=False)
    obj.phewas_results.to_csv(output / 'glyph_associations.csv', index=False)
    measurements.to_csv(output / 'measurements.csv', index=False)
    vectors = []
    for split, data in obj.embeddings.items():
        block = pd.DataFrame(obj.sae_features['image_path'][split].cpu().numpy(),
            columns=[f'concept_{i}' for i in range(obj.sae_features['image_path'][split].shape[1])])
        block.insert(0, 'sample_id', data['sample_ids']); vectors.append(block)
    pd.concat(vectors, ignore_index=True).to_csv(output / 'concepts.csv', index=False)
    selected = np.flatnonzero((obj.sae_features['image_path']['train'].cpu().numpy()>0).any(axis=0)).tolist()
    if not selected:
        raise ValueError('At least one training-active concept is required for comparison')
    definitions = [('age_60_74', 'Recorded age band 60–74 versus all other bands', True),
                   ('age_75_plus', 'Recorded age band 75–90 versus all other bands', True),
                   ('male', 'Recorded male sex versus female; unknown stays missing', False)]
    markers = [dict(id=column, column=column, record_id='pmid:32640463',
        definition=definition, source_url=source,
        citation='Williamson et al., Factors associated with COVID-19-related death using OpenSAFELY, Nature 2020',
        doi='10.1038/s41586-020-2521-4', proxy=proxy, units='binary 0/1',
        formula=dict(op='identity', column=column)) for column, definition, proxy in definitions]
    manifest = dict(title='Glyph + DINOv3: chest X-ray mortality', outcome_column='y',
        outcome_labels={'in_hospital_death': {'0': 'discharged', '1': 'later death'}},
        scope='TCIA COVID-19-NY-SBU (10.7937/TCIA.BBAG-2923); complete training-alive concept space',
        markers=markers, glyph_features=[f'concept_{i}' for i in selected],
        glyph_selection_rule='All training-active dictionary concepts; capped variants ranked by two-sided univariate logistic score-test p-values on training rows only',
        comparison_top_n=10,
        report_comparison_split='test',
        caveats=['One patient per observation; internal validation at one site.',
                 'Admission-period images precede the hospital-day-two mortality landmark.',
                 'Age bands and sex form a simplified literature baseline, not the published model.',
                 'No LLM interpretation or confounder adjustment.'])
    (output / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    subprocess.run([sys.executable, '-B', 'skills/glyph-literature/scripts/workflow.py', 'freeze',
        '--search', str(search), '--manifest', str(output / 'candidate_manifest.json'),
        '--output', str(output / 'frozen_manifest.json')], check=True)
    subprocess.run([sys.executable, '-B', 'skills/glyph-literature/scripts/workflow.py', 'study',
        '--cohort', str(output / 'cohort.csv'), '--measurements', str(output / 'measurements.csv'),
        '--glyph', str(output / 'concepts.csv'), '--glyph-associations', str(output / 'glyph_associations.csv'), '--manifest', str(output / 'frozen_manifest.json'),
        '--search', str(search), '--run', str(obj.run_dir), '--report', '--image-examples',
        '--bootstrap', str(bootstrap), '--output', str(output / 'study')], check=True)
    return output / 'study'
