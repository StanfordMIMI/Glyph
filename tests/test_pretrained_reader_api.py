from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from glyph import Glyph
from glyph.zoo.pretrained import PretrainedSAE
from glyph.zoo.registry import resolve


@pytest.fixture
def reader(tmp_path, monkeypatch):
    entry = resolve('ct-merlin')
    weights = torch.zeros(entry.input_dim, entry.n_concepts)
    weights[0, :16] = torch.arange(1, 17)
    weights[0, 87] = 20
    bundle = PretrainedSAE(entry, weights, torch.zeros(entry.n_concepts),
        {'mean': [0.] * entry.input_dim, 'std': 1., 'dim_scale': 1.},
        pd.DataFrame({'feature_name': ['Concept_87'], 'extracted_interpretation': ['Synthetic reference interpretation']}))
    loads = []
    monkeypatch.setattr('glyph.zoo.load_pretrained', lambda *args, **kwargs: loads.append(args[0]) or bundle)
    calls = []
    class Encoder:
        def embed_batch(self, values):
            calls.extend(values)
            return torch.tensor([[float(value)] * entry.input_dim for value in values])
    monkeypatch.setattr('glyph.models.get_model', lambda *args, **kwargs: Encoder())
    monkeypatch.setattr('glyph.mechinterp.sae.SAE.fit', lambda *args, **kwargs: pytest.fail('Pretrained measurement trained an SAE'))
    obj = Glyph.from_pretrained('ct-merlin', cache_dir=tmp_path/'cache', output_dir=tmp_path/'runs')
    frame = pd.DataFrame({'sample_id': ['c', 'a', 'b'], 'image_path': ['1.0', '1.1', '0.9']})
    return obj, frame, calls, loads


def test_single_study_then_whole_space_reuses_embeddings_and_keeps_all_codes(reader):
    obj, frame, calls, loads = reader
    assert calls == []
    one = obj.measure('I87', metadata=frame, sample_id='a')
    assert one.index.tolist() == ['a'] and calls == ['1.1']
    full = obj.measure(metadata=frame)
    assert full.shape == (3,8192)
    assert full.index.tolist() == ['c','a','b']
    assert full.columns.tolist() == [f'I{i}' for i in range(8192)]
    assert calls == ['1.1','1.0','0.9']
    pd.testing.assert_series_equal(obj.measure('I87'), full.I87)
    pd.testing.assert_series_equal(full.loc[['a'],'I87'], one)
    assert len(loads) == 1
    assert obj.cfg.data.outcome_cols == []
    assert set(obj.embeddings) == {'cohort'}


def test_csv_ids_and_no_train_test_split_are_preserved(reader, tmp_path):
    obj, frame, _, _ = reader
    frame['sample_id'] = ['001','003','002']
    path = tmp_path/'metadata.csv';frame.to_csv(path,index=False)
    result = obj.measure('I87', metadata=path)
    assert result.index.tolist() == ['001','003','002']


def test_saved_measurement_run_reloads_without_encoder_or_dictionary_download(reader, monkeypatch):
    obj, frame, _, _ = reader
    expected = obj.measure(metadata=frame)
    monkeypatch.setattr('glyph.models.get_model', lambda *args, **kwargs: pytest.fail('Reload loaded encoder'))
    monkeypatch.setattr('glyph.zoo.load_pretrained', lambda *args, **kwargs: pytest.fail('Reload downloaded dictionary'))
    loaded = Glyph.load_run(obj.run_dir, device='cpu')
    pd.testing.assert_frame_equal(loaded.measure(), expected)
    pd.testing.assert_series_equal(loaded.measure('I87', sample_id='b'), expected.loc[['b'],'I87'])


def test_explicit_embeddings_skip_raw_inputs_and_survive_reload(reader, tmp_path, monkeypatch):
    obj, frame, calls, _ = reader
    file=tmp_path/'raw.pt'
    raw=torch.tensor([[float(value)]*2048 for value in frame.image_path])
    torch.save(dict(sample_ids=frame.sample_id.tolist(),embeddings=raw),file)
    result=obj.measure(metadata=frame[['sample_id']],embeddings_path=file)
    assert calls==[]
    loaded=Glyph.load_run(obj.run_dir)
    pd.testing.assert_frame_equal(loaded.measure(),result)


def test_metadata_filters_errors_and_recompute(reader):
    obj, frame, calls, _=reader
    with pytest.raises(ValueError,match='first measure'):
        obj.measure('I87')
    with pytest.raises(KeyError,match='No studies'):
        obj.measure('I87',metadata=frame,sample_id='absent')
    assert not calls
    labelled=frame.assign(split=['external','external','discovery'])
    selected=obj.measure('I87',metadata=labelled,split='external')
    assert selected.index.tolist()==['c','a']
    first=obj.run_dir
    obj.measure('I87',metadata=labelled,split='external',recompute=True)
    assert first!=obj.run_dir and first.is_dir()
    assert len(calls)==4
    with pytest.raises(ValueError,match='unique'):
        obj.measure(metadata=pd.concat([frame,frame.iloc[:1]]))


def test_whole_space_can_be_visualized_without_labels(reader, monkeypatch, tmp_path):
    obj, frame, _, _=reader
    obj.measure(metadata=frame)
    monkeypatch.setattr('glyph.visualization.interactive_concept_network._layout',
        lambda correlation,*args,**kwargs: np.column_stack([np.linspace(0,1,len(correlation))]*2))
    paths=obj.visualize()
    assert len(paths)==1 and Path(paths[0]).is_file()
    text=Path(paths[0]).read_text()
    assert 'Synthetic reference interpretation' in text
    assert 'No association testing' in text or 'no association testing' in text
    assert 'NaN' not in text


def test_invalid_concept_rejected_before_encoder_loading(reader):
    obj, frame, calls, _=reader
    for code in ['R64','I99999']:
        with pytest.raises(KeyError):obj.measure(code,metadata=frame)
    assert not calls


def test_distribution_check_warns_by_default_and_strict_mode_rejects(reader):
    obj, frame, _, _=reader
    shifted=frame.assign(image_path='3.0')
    with pytest.warns(UserWarning,match='embedding magnitude'):
        result=obj.measure(metadata=shifted)
    assert result.shape==(3,8192)
    obj._measurement_strict=True
    with pytest.raises(ValueError,match='off-distribution'):
        obj.measure(metadata=shifted,recompute=True)


def test_reload_preserves_nondefault_dataframe_index_and_cache(reader,monkeypatch):
    obj, frame, _, _=reader
    frame.index=[42,9,17]
    expected=obj.measure(metadata=frame)
    loaded=Glyph.load_run(obj.run_dir)
    monkeypatch.setattr('glyph.models.get_model',lambda *args,**kwargs:pytest.fail('Cache was not retained'))
    pd.testing.assert_frame_equal(loaded.measure(),expected)


def test_measurement_can_reuse_exported_embeddings_regardless_of_partition_names(reader,tmp_path):
    obj, frame, calls, _=reader
    expected=obj.measure(metadata=frame)
    file=obj.run_dir/'image_embeddings.pt'
    other=Glyph.from_pretrained('ct-merlin',output_dir=tmp_path/'other',cache_dir=tmp_path/'cache')
    calls.clear()
    actual=other.measure(metadata=frame[['sample_id']],embeddings_path=file)
    pd.testing.assert_frame_equal(actual,expected)
    assert calls==[]
