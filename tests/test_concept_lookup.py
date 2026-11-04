from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from glyph import Glyph


def example(modality='ct_path', bundle='ct-merlin@1'):
    obj = Glyph.__new__(Glyph)
    obj.sae_features = {modality: {'test': torch.arange(300, dtype=torch.float32).reshape(3, 100)}}
    obj.sae_models = {modality: SimpleNamespace(selection_report={'pretrained': bundle} if bundle else {})}
    obj.embeddings = {'test': {'sample_ids': ['scan_c', 'scan_a', 'scan_b']}}
    return obj


@pytest.mark.parametrize('code,column,bundle,index', [
    ('I87', 'ct_path', 'ct-merlin@1', 87),
    ('R64', 'reports', 'findings-qwen3@1', 64),
    ('L92', 'lab_panels', 'labs-loinc@1', 92),
])
def test_codes_preserve_original_indices_and_sample_order(code, column, bundle, index):
    obj = example(column, bundle)
    result = obj.measure(code)
    pd.testing.assert_series_equal(result, pd.Series(
        np.arange(3, dtype=np.float32) * 100 + index,
        index=pd.Index(['scan_c', 'scan_a', 'scan_b'], name='sample_id'), name=code))
    result.iloc[0] = -100
    assert obj.sae_features[column]['test'][0, index] == index


def test_codes_do_not_assign_paper_identity_to_new_dictionaries():
    obj = example(bundle=None)
    with pytest.raises(KeyError, match='published dictionary'):
        obj.measure('I87')
    result = obj.measure(87, modality='ct_path')
    assert result.name == 'Concept_87' and result.iloc[0] == 87


def test_codes_resolve_by_dictionary_not_column_name():
    obj = example('arbitrary_column')
    assert obj.measure('i87').iloc[0] == 87
    with pytest.raises(KeyError, match='published dictionary'):
        obj.measure('R87')


def test_ambiguous_dictionaries_need_explicit_modality():
    obj = example()
    obj.sae_models['other_ct'] = obj.sae_models['ct_path']
    obj.sae_features['other_ct'] = {'test': obj.sae_features['ct_path']['test'] + 1}
    with pytest.raises(ValueError, match='ambiguous'):
        obj.measure('I87')
    assert obj.measure('I87', modality='other_ct').iloc[0] == 88
    with pytest.raises(ValueError, match='ambiguous'):
        obj.measure(87)


@pytest.mark.parametrize('code', ['unknown', 'I-1', 'I1.5', True, -1, 2.5])
def test_invalid_codes_fail_clearly(code):
    with pytest.raises(ValueError):
        example().measure(code)


def test_unavailable_split_and_index_fail_clearly():
    with pytest.raises(KeyError, match='Unknown split'):
        example().measure('I87', split='external')
    with pytest.raises(KeyError, match='indices'):
        example().measure('I100')
    with pytest.raises(KeyError, match='Unknown modality'):
        example().measure(87, modality='absent')
    with pytest.raises(RuntimeError, match='fit'):
        Glyph.__new__(Glyph).measure('I87')


def test_zero_activations_are_not_filtered_out():
    obj = example()
    obj.sae_features['ct_path']['test'][:, 87] = 0
    assert obj.measure('I87').eq(0).all()


def test_filtered_csv_omissions_are_not_returned_as_real_zero_activations():
    obj = example()
    obj.sae_models = {}
    mask = torch.zeros(100, dtype=torch.bool)
    mask[87] = True
    obj.sae_alive_features = {('ct_path', 40): mask}
    assert obj.measure(87).iloc[0] == 87
    with pytest.raises(KeyError, match='not retained'):
        obj.measure(86)
