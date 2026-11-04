"""Build a small synthetic viewer to inspect the HTML presentation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from glyph.visualization import build_interactive_concept_atlas


def build_example(output_path, view='simple', n_outcomes=4):
    rng = np.random.default_rng(12)
    modalities = ['Image', 'Report', 'Laboratory']
    outcomes = [f'Outcome {chr(65+i)}' if i<26 else f'Outcome {i+1}' for i in range(n_outcomes)]
    centers = [(0.28, 0.32), (0.64, 0.35), (0.49, 0.69)]
    concepts, associations = [], []
    for i in range(42):
        group = i % 3
        x, y = np.clip(rng.normal(centers[group], 0.105), 0.1, 0.9)
        concept_id = f'example-{i + 1}'
        concepts.append(dict(
            concept_id=concept_id, name=f'{modalities[group]} concept {i + 1}',
            modality=modalities[group], x=x, y=y, active=100 + i * 7,
            rate=10 + i % 25, interpretation=(
                'An illustrative concept from a synthetic dataset. '
                'Select outcomes or change the display options to explore its associations.'),
        ))
        for j, outcome in enumerate(outcomes):
            odds_ratio = float(np.exp(rng.normal(0.1 + 0.15 * (group == j % 3), 0.42)))
            corrected_p = 0.01 if (i + j) % 4 else 0.4
            associations.append(dict(
                concept_id=concept_id, outcome=outcome, odds_ratio=odds_ratio,
                odds_ratio_p_value=corrected_p / (42*n_outcomes),
                odds_ratio_ci_lower=odds_ratio*.8,
                odds_ratio_ci_upper=odds_ratio*1.2,
                c_f=100+i*7,
                odds_ratio_p_value_bonferroni=corrected_p,
                odds_ratio_p_value_bonferroni_modality=corrected_p / 3,
                bonferroni_significant=corrected_p < 0.05,
            ))
    edges = []
    positions = np.array([[node['x'], node['y']] for node in concepts])
    for i in range(len(concepts)):
        nearest = np.argsort(np.linalg.norm(positions - positions[i], axis=1))[1:3]
        edges.extend((concepts[i]['concept_id'], concepts[j]['concept_id'], 0.65)
                     for j in nearest if i < j)
    return build_interactive_concept_atlas(
        pd.DataFrame(concepts), pd.DataFrame(associations), output_path,
        edges=edges, title='Glyph · Concept explorer', initial_outcome='Outcome A',
        selection_note='Synthetic example · illustrative associations only', view=view,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path('concept_explorer.html'))
    parser.add_argument('--view', choices=['simple', 'full'], default='simple')
    parser.add_argument('--outcomes', type=int, default=4)
    args = parser.parse_args()
    if args.outcomes < 1: parser.error('--outcomes must be positive')
    build_example(args.output, args.view, args.outcomes)
