"""Opt-in real public weights on synthetic new inputs, never research cohorts."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from glyph import Glyph

if os.environ.get('GLYPH_EXPECT_INSTALLED_PREFIX'):
    import glyph
    assert Path(glyph.__file__).is_relative_to(Path(os.environ['GLYPH_EXPECT_INSTALLED_PREFIX']))


pytestmark = pytest.mark.skipif(os.environ.get('GLYPH_READER_HF_INTEGRATION') != '1',
    reason='Set GLYPH_READER_HF_INTEGRATION=1 for actual published model inference')


def test_released_labs_whole_space_single_code_and_html(tmp_path):
    from glyph.extraction.labs import LabsEncoder
    cache = Path(os.environ.get('HF_HOME', 'cache/huggingface'))
    encoder = LabsEncoder(device='cpu', cache_dir=cache, token=False)
    stats = encoder.metadata[encoder.metadata.retained & encoder.metadata.is_numeric].head(20)
    metadata=[]
    for index in range(4):
        tokens=[dict(lab_id=int(row.lab_id),Value=float(row['mean'])+float(row['std'])*(.1*index-.15),years_prior=.1)
                for _,row in stats.iterrows()]
        path=tmp_path/f'labs_{index}.json';path.write_text(json.dumps(tokens))
        metadata.append(dict(sample_id=f'synthetic_{index}',labs_path=str(path)))
    frame=pd.DataFrame(metadata);csv=tmp_path/'labs_metadata.csv';frame.to_csv(csv,index=False)
    obj=Glyph.from_pretrained('labs-loinc',device='cpu',cache_dir=os.environ.get('GLYPH_READER_CACHE_DIR','cache'),output_dir=tmp_path/'runs',
        encoder_kwargs={'cache_dir':str(cache),'token':False})
    full=obj.measure(metadata=csv)
    assert full.shape==(4,8192) and full.columns.tolist()==[f'L{i}' for i in range(8192)]
    assert np.isfinite(full.to_numpy()).all()
    pd.testing.assert_series_equal(obj.measure('L127'),full.L127)
    original=obj._measurement_encoder.embed_batch
    def no_recompute(*args,**kwargs):pytest.fail('Repeated lookup recomputed embeddings')
    obj._measurement_encoder.embed_batch=no_recompute
    pd.testing.assert_series_equal(obj.measure('L127',sample_id='synthetic_2'),full.loc[['synthetic_2'],'L127'])
    obj._measurement_encoder.embed_batch=original
    direct=obj.pretrained_bundles['labs_path'].encode(torch.cat([encoder.embed_batch([value]) for value in frame.labs_path]))
    np.testing.assert_array_equal(full.to_numpy(),direct.numpy())
    paths=obj.visualize()
    assert Path(paths[0]).is_file()
    restored=Glyph.load_run(obj.run_dir,device='cpu')
    pd.testing.assert_frame_equal(restored.measure(),full)
    (tmp_path/'real_labs_verification.json').write_text(json.dumps(dict(real_weights=True,inputs='synthetic independent cohort',
        samples=4,concepts=8192,single_code=True,html=str(paths[0]),reload_exact=True),indent=2))


def test_released_qwen8b_findings_dictionary_on_new_reports(tmp_path):
    reports=[
        'FINDINGS: The liver is enlarged with diffuse low attenuation. No ascites. IMPRESSION: Hepatic steatosis.',
        'FINDINGS: Kidneys are normal in size. No hydronephrosis. IMPRESSION: No acute abdominal finding.',
        'FINDINGS: Aortic atherosclerotic calcifications are present. IMPRESSION: Atherosclerosis.'
    ]
    rows=[]
    for i,text in enumerate(reports):
        path=tmp_path/f'report_{i}.txt';path.write_text(text)
        rows.append(dict(sample_id=f'synthetic_report_{i}',report_path=str(path)))
    frame=pd.DataFrame(rows);csv=tmp_path/'reports.csv';frame.to_csv(csv,index=False)
    obj=Glyph.from_pretrained('findings-qwen3',device='cpu',cache_dir=os.environ.get('GLYPH_READER_CACHE_DIR','cache'),output_dir=tmp_path/'runs')
    full=obj.measure(metadata=csv)
    assert full.shape==(3,8192) and np.isfinite(full.to_numpy()).all()
    pd.testing.assert_series_equal(obj.measure('R64'),full.R64)
    bundle=obj.pretrained_bundles['report_path']
    expected=bundle.encode(torch.cat([obj._measurement_encoder.embed_batch([value]) for value in frame.report_path]), strict=False)
    np.testing.assert_allclose(full.to_numpy(),expected.numpy(),rtol=2e-4,atol=2e-4)
    paths=obj.visualize();assert Path(paths[0]).is_file()
    restored=Glyph.load_run(obj.run_dir,device='cpu')
    pd.testing.assert_frame_equal(restored.measure(),full)
    (tmp_path/'real_findings_verification.json').write_text(json.dumps(dict(real_weights=True,inputs='synthetic new reports',
        samples=3,concepts=8192,encoder='Qwen/Qwen3-Embedding-8B',single_code=True,html=str(paths[0]),reload_exact=True),indent=2))


def test_released_ct_dictionary_on_independent_public_volume(tmp_path):
    case_path=os.environ.get('GLYPH_PUBLIC_CT_CASE')
    if not case_path:
        pytest.skip('Supply GLYPH_PUBLIC_CT_CASE describing the official public CT download')
    case=json.loads(Path(case_path).read_text())
    assert case['repo']=='MONAI/testing_data'
    frame=pd.DataFrame([dict(sample_id='public_spleen',image_path=case['path'])])
    csv=tmp_path/'ct_metadata.csv';frame.to_csv(csv,index=False)
    obj=Glyph.from_pretrained('ct-merlin',device='cpu',cache_dir=os.environ.get('GLYPH_READER_CACHE_DIR','cache'),output_dir=tmp_path/'runs')
    single=obj.measure('I87',metadata=csv,sample_id='public_spleen')
    full=obj.measure(metadata=csv)
    assert full.shape==(1,8192) and np.isfinite(full.to_numpy()).all()
    pd.testing.assert_series_equal(single,full.I87)
    expected=obj.pretrained_bundles['image_path'].encode(obj.embeddings['cohort']['embeddings']['image_path'],strict=False)
    np.testing.assert_array_equal(full.to_numpy(),expected.numpy())
    paths=obj.visualize()
    assert Path(paths[0]).is_file()
    html=Path(paths[0]).read_text()
    assert 'I87' in html or full.I87.iloc[0]==0
    assert 'correlations not estimated' in html
    restored=Glyph.load_run(obj.run_dir,device='cpu')
    pd.testing.assert_frame_equal(restored.measure(),full)
    (tmp_path/'real_ct_verification.json').write_text(json.dumps(dict(real_weights=True,
        dataset=case['repo'],dataset_revision=case['revision'],file=case['file'],
        samples=1,concepts=8192,single_code=True,html=str(paths[0]),reload_exact=True),indent=2))
