"""One public-image demo: official DINOv3 weights, new SAE and digit associations."""
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch

from glyph import Glyph

MODEL = 'facebook/dinov3-vits16-pretrain-lvd1689m'
REVISION = '114c1379950215c8b35dfcd4e90a5c251dde0d32'


def prepare(output, n=1000):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    images = output / 'images'
    images.mkdir()
    digits = load_digits()
    selected, _ = train_test_split(np.arange(len(digits.target)), train_size=n,
                                  stratify=digits.target, random_state=42)
    train, held = train_test_split(selected, test_size=.30,
                                  stratify=digits.target[selected], random_state=42)
    val, test = train_test_split(held, test_size=.5,
                               stratify=digits.target[held], random_state=42)
    splits = {int(i): split for split, indices in [('train', train), ('val', val), ('test', test)]
              for i in indices}
    rows = []
    for i in selected:
        sample_id = f'digit-{i:04d}'
        path = images / f'{sample_id}.png'
        Image.fromarray((digits.images[i] * 255 / 16).astype(np.uint8)).convert('RGB').save(path)
        row = dict(sample_id=sample_id, image_path=str(path), split=splits[int(i)])
        row.update({f'is_digit_{label}': int(digits.target[i] == label) for label in range(10)})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'metadata.csv', index=False)
    provenance = dict(dataset='UCI Optical Recognition of Handwritten Digits / sklearn.load_digits',
        source='https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html',
        seed=42, rows=n, selected_indices=selected.tolist(), model=MODEL, revision=REVISION,
        pixel_sha256=hashlib.sha256(digits.images[selected].tobytes()).hexdigest(),
        split_counts=frame.split.value_counts().to_dict(),
        note='Public 8x8 grayscale digits converted to RGB; model processor resizes them. '
             'No author research cohort. Digit identity labels are used only for probes and association tests.')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for label in range(10):
        for j, i in enumerate([i for i in selected if digits.target[i] == label][:2]):
            axes[j, label].imshow(digits.images[i], cmap='gray', vmin=0, vmax=16)
            axes[j, label].axis('off')
            if j == 0:
                axes[j, label].set_title(str(label))
    fig.suptitle('Public inputs: two examples of each digit')
    fig.tight_layout()
    fig.savefig(output / 'inputs.png', dpi=150)
    plt.close(fig)
    return output / 'metadata.csv'


def make_glyph(metadata, output):
    return Glyph('config', config_overrides=[
        f'data.metadata={Path(metadata).resolve()}',
        'data.input_cols=[image_path]',
        'data.outcome_cols=[' + ','.join(f'is_digit_{i}' for i in range(10)) + ']',
        f'data.embedding_map={{image_path: "hf:{MODEL}"}}',
        f'data.embedding_model_kwargs={{image_path: {{input_type: image, pooling: cls, revision: "{REVISION}"}}}}',
        f'data.output_dir={Path(output).resolve()}',
        f'data.cache_dir={Path("cache").resolve()}',
        'data.num_workers=0', 'data.extraction_batch_size=16',
        'device=cpu', 'sae.top_ks=[8,16,32,64,128,256,512,1024,2048,4096,8192]', 'sae.matryoshka=[8192]',
        'sae.max_steps=2000', 'sae.learning_rate=0.001',
        'sae.selection.lookahead_step=128', 'phewas.min_activations=5',
        'visualization.correlation_percentile=0.9995', 'autointerp.enabled=false',
    ])


def summarize(obj, output):
    output = Path(output)
    associations = obj.phewas_results.copy()
    associations.to_csv(output / 'associations.csv', index=False)
    discovery = associations[associations.split == 'train'].copy()
    pcol = 'odds_ratio_p_value_bonferroni'
    hits = discovery[(discovery[pcol] < .05) & (discovery.odds_ratio > 1)].sort_values(pcol)
    chosen = hits.drop_duplicates('outcome').drop_duplicates('feature_idx').head(4)
    metadata = pd.read_csv(obj.cfg.data.metadata).set_index('sample_id')
    if len(chosen):
        fig, axes = plt.subplots(len(chosen), 8, figsize=(12, 2 * len(chosen)), squeeze=False)
        for j, row in enumerate(chosen.itertuples()):
            values = obj.sae_features['image_path']['train'][:, int(row.feature_idx)].cpu().numpy()
            ids = np.asarray(obj.embeddings['train']['sample_ids'])
            for col, pos in enumerate(np.argsort(-values, kind='stable')[:8]):
                sample = metadata.loc[ids[pos]]
                label = next(i for i in range(10) if sample[f'is_digit_{i}'] == 1)
                axes[j, col].imshow(Image.open(sample.image_path))
                axes[j, col].axis('off')
                axes[j, col].set_title(f'digit {label}', fontsize=9)
            axes[j, 0].text(-.1, .5, f'Concept {int(row.feature_idx)}\n{row.outcome}',
                            transform=axes[j, 0].transAxes, ha='right', va='center', fontsize=9)
        fig.suptitle('Strongest activating training images\nConcepts selected by corrected training associations; no semantic interpretation was run')
        fig.tight_layout(rect=(.12, 0, 1, .9))
        fig.savefig(output / 'concept_examples.png', dpi=160)
        plt.close(fig)
    loaded = Glyph.load_run(obj.run_dir, device='cpu')
    for split, data in obj.embeddings.items():
        torch.testing.assert_close(loaded.encode_embeddings(data['embeddings']['image_path'], modality='image_path'),
                                   obj.sae_features['image_path'][split], rtol=0, atol=0)
    summary = dict(run=str(obj.run_dir), selected_k=obj.sae_models['image_path'].selected_top_k,
                   selection=obj.sae_models['image_path'].selection_report,
                   alive_concepts=int(np.asarray(next(iter(obj.sae_alive_features.values()))).sum()),
                   discovery_positive_hits=len(hits), reload_identical=True)
    (output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def digit4_focus(obj):
    """Choose complementary digit-4 examples using training data and map only."""
    train=obj.phewas_results.query("split == 'train'")
    dominant=train.sort_values(['odds_ratio_p_value_bonferroni','feature_idx','outcome']).drop_duplicates('feature_idx')
    candidates=dominant[(dominant.outcome=='is_digit_4')&(dominant.odds_ratio>1)&(dominant.odds_ratio_p_value_bonferroni<.05)]
    values=obj.sae_features['image_path']['train'].cpu().numpy();cases=np.asarray(obj.outcomes['train']['is_digit_4'])==1
    k=obj.sae_models['image_path'].selected_top_k;html=obj.run_dir/f'concept_map_image_path_k{k}_train.html'
    if not html.exists():obj.visualize(split='train')
    match=re.search(r'const DATA=(.*?),NS=',html.read_text());data=json.loads(match.group(1));positions={n['feature_idx']:(n['x'],n['y']) for n in data['nodes']}
    eligible=[]
    for index in candidates.feature_idx.astype(int):
        active=values[:,index]>1e-8;case_count=int((active&cases).sum())
        if 6<=case_count<=.85*cases.sum() and (active&cases).sum()/active.sum()>=.6:eligible.append(index)
    pairs=[]
    for i,a in enumerate(eligible):
        for b in eligible[i+1:]:
            distance=float(np.linalg.norm(np.array(positions[a])-positions[b]))
            correlation=float(np.corrcoef(values[cases,a],values[cases,b])[0,1])
            if distance>=.18 and np.isfinite(correlation):pairs.append((correlation,-distance,a,b))
    if not pairs:return {}
    correlation,negative_distance,a,b=min(pairs)
    result=dict(focus_concepts=[f'image_path:{k}:{a}',f'image_path:{k}:{b}'],focus_outcome='is_digit_4',
        focus_selection_note='Two positive, corrected digit-4 associations chosen using training examples only, with low within-digit activation correlation and separation on the saved map; descriptive style illustrations, not validated subtype labels.',
        digit4_selection=dict(training_case_count=int(cases.sum()),eligible_concepts=len(eligible),within_digit4_correlation=correlation,map_distance=-negative_distance))
    profile=Path(__file__).resolve().parents[1]/'demos/dinov3_digits/report_profile.json'
    if profile.exists():
        known=json.loads(profile.read_text())
        if known.get('source_run')==obj.run_dir.name:
            chosen=known.get('focus_concepts',[])
            indices=[int(code.rsplit(':',1)[-1]) for code in chosen]
            if len(indices)==2 and len(set(indices))==2 and all(i in candidates.feature_idx.values for i in indices):
                a,b=indices
                distance=float(np.linalg.norm(np.array(positions[a])-positions[b]))
                if distance>=.18:
                    result['focus_concepts']=chosen
                    result['concept_notes']=known.get('concept_notes',{})
                    result['focus_selection_note']=known.get('selection_note',result['focus_selection_note'])
                    result['digit4_selection'].update(within_digit4_correlation=float(np.corrcoef(values[cases,a],values[cases,b])[0,1]),map_distance=distance,selection='Training-image review of nearby, higher-prevalence concepts',training_active_counts=[int((values[:,i]>1e-8).sum()) for i in indices])
    return result


def report(run, output):
    output = Path(output)
    metadata = output.parent / 'report_metadata.json'
    profile=dict(
        scope='All training-alive concepts learned from public DINOv3 digit embeddings',
        interpretation_status='Style descriptions are unvalidated training-image observations.',
        outcome_labels={f'is_digit_{i}': {'0': 'other digits', '1': f'digit {i}'} for i in range(10)},
        notes='Public UCI optical digits via sklearn.load_digits. Probe AUC shown is digit 0 versus rest; '
              'all ten outcome probes are saved. Associations are unadjusted. '
              'No automated interpretation pipeline or literature comparison was run.',
    )
    profile.update(digit4_focus(Glyph.load_run(run,device='cpu')))
    metadata.write_text(json.dumps(profile,indent=2)+'\n')
    subprocess.run([sys.executable, 'skills/glyph-report/scripts/build_report.py', '--run', str(run),
        '--output', str(output), '--metadata', str(metadata),
        '--title', 'Glyph + DINOv3: handwritten digits', '--image-examples', '--preview-html',
        '--split', 'train', '--comparison-split', 'test'], check=True)
