import base64
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('demo_site',Path(__file__).parents[1]/'scripts/build_demo_site.py');site=importlib.util.module_from_spec(spec);spec.loader.exec_module(site)


def test_pages_site_only_copies_public_allowlist_and_externalizes_images(tmp_path):
    repo=tmp_path/'repo';demo=repo/'demos/public_demo';demo.mkdir(parents=True)
    (repo/'demos/manifest.json').write_text(json.dumps(dict(public_or_synthetic_only=True,current_demos=['public_demo'])))
    raw=b'a thumbnail';data={'image_bank':{'abc123':{'src':'data:image/png;base64,'+base64.b64encode(raw).decode()}},'concept_images':{'0':[['abc123',2.]]},'nodes':[{'i':0,'x':.2,'y':.4}]}
    original='const DATA='+json.dumps(data)+",NS='svg';";(demo/'explorer.html').write_text(original)
    (demo/'private_notes.csv').write_text('do not publish');(demo/'report.pdf').write_bytes(b'%PDF');(demo/'report_files').mkdir();(demo/'report_files/report.html').write_text('report')
    result=site.build_site(repo,tmp_path/'site');published=tmp_path/'site/public_demo'
    assert (published/'images/abc123.png').read_bytes()==raw
    assert 'data:image/png' not in (published/'explorer.html').read_text()
    assert 'images/abc123.png' in (published/'explorer.html').read_text()
    assert not (published/'private_notes.csv').exists()
    assert (demo/'explorer.html').read_text()==original
    assert result[0]['thumbnail_bytes']==len(raw)
    assert (tmp_path/'site/.nojekyll').exists()


def test_unconfirmed_private_demo_inputs_are_not_published(tmp_path):
    (tmp_path/'demos').mkdir();(tmp_path/'demos/manifest.json').write_text(json.dumps(dict(public_or_synthetic_only=False,current_demos=[])))
    with pytest.raises(ValueError,match='public/synthetic'):site.build_site(tmp_path,tmp_path/'site')
    assert not (tmp_path/'site').exists()
