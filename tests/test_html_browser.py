"""Opt-in standalone browser checks: GLYPH_BROWSER_TESTS=1 pytest tests/test_html_browser.py."""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glyph.visualization import build_interactive_concept_atlas


@pytest.fixture(scope='module')
def browser():
    if os.environ.get('GLYPH_BROWSER_TESTS') != '1':
        pytest.skip('Set GLYPH_BROWSER_TESTS=1 and install Playwright Chromium')
    playwright = pytest.importorskip('playwright.sync_api')
    with playwright.sync_playwright() as p:
        instance = p.chromium.launch(headless=True, args=['--no-sandbox'])
        yield instance
        instance.close()


def viewer(tmp_path, count=4, **kwargs):
    concepts = pd.DataFrame([
        dict(concept_id='a', feature_idx=7, name='Imaging pattern', interpretation='An illustrative pattern.',
             x=.3, y=.3, modality='Image', active=32, rate=40),
        dict(concept_id='b', feature_idx=98, name='Laboratory pattern', interpretation='A second illustrative pattern.',
             x=.7, y=.7, modality='Laboratory', active=24, rate=30),
    ])
    associations = pd.DataFrame([
        dict(concept_id=node, outcome=f'Outcome {i:03d}', odds_ratio=float(np.exp((i % 9 - 4) * .15)),
             odds_ratio_ci_lower=float(np.exp((i % 9 - 4) * .15 - .2)),
             odds_ratio_ci_upper=float(np.exp((i % 9 - 4) * .15 + .2)),
             odds_ratio_p_value=1e-30 if i == 0 else .1,
             odds_ratio_p_value_bonferroni=1e-27 if i == 0 else .2,
             bonferroni_significant=i == 0, c_f=32 if node == 'a' else 24)
        for node in ['a', 'b'] for i in range(count)
    ])
    if not count: associations=pd.DataFrame(columns=['concept_id','outcome','odds_ratio'])
    return build_interactive_concept_atlas(concepts, associations, tmp_path/f'viewer-{count}.html',
            edges=[('a', 'b', .8)], title='Glyph browser test', **kwargs)


def open_page(browser, path, width=1440, height=960):
    page = browser.new_page(viewport=dict(width=width, height=height))
    errors, remote = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('request', lambda request: remote.append(request.url) if request.url.startswith(('http:', 'https:')) else None)
    page.goto(path.resolve().as_uri())
    return page, errors, remote


@pytest.mark.parametrize('count,labels', [(1,1), (4,4), (20,20), (21,0), (150,0)])
def test_wheel_preserves_all_outcomes_and_stats(browser, tmp_path, count, labels):
    page, errors, remote = open_page(browser, viewer(tmp_path, count))
    try:
        page.locator('.hit[data-node="0"]').click()
        assert page.locator('.wheel-spoke').count() == count
        assert page.locator('.wheel-label').count() == labels
        # Every spoke, including a dense 150-outcome wheel, remains keyboard accessible.
        for index in range(count):
            spoke = page.locator('.wheel-spoke').nth(index)
            spoke.focus()
            text = page.locator('#tip').inner_text()
            assert f'Outcome {index:03d}' in text
            assert 'Odds ratio' in text and '95% confidence interval' in text
            assert 'Raw p-value' in text and 'Bonferroni p-value' in text
            assert 'Reported support' in text
        page.locator('.wheel-spoke').first.focus()
        assert '1.00e-30' in page.locator('#tip').inner_text()
        assert '1.00e-27' in page.locator('#tip').inner_text()
        assert not errors and not remote
        page.screenshot(path=str(tmp_path/f'wheel-{count}.png'))
    finally:
        page.close()


def test_spoke_hover_export_and_outcome_search(browser, tmp_path):
    page, errors, remote = open_page(browser, viewer(tmp_path))
    try:
        page.locator('.hit[data-node="0"]').click()
        # Move to an outer hit region, independent of the visible effect radius.
        location = page.locator('.wheel').evaluate('''svg=>{const p=svg.createSVGPoint();p.x=200;p.y=40;const q=p.matrixTransform(svg.getScreenCTM());return {x:q.x,y:q.y}}''')
        page.mouse.move(location['x'], location['y'])
        assert 'Outcome 000' in page.locator('#tip').inner_text()
        assert page.locator('#tip').evaluate('e=>getComputedStyle(e).opacity') == '1'
        with page.expect_download() as event:
            page.locator('#exportConcept').click()
        destination = tmp_path/'associations.csv'
        event.value.save_as(destination)
        with destination.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4 and float(rows[0]['p_value']) == 1e-30
        assert all(row['concept'] == 'a' for row in rows)
        assert all(row['ci_lower'] for row in rows)
        page.locator('#outcomePicker>summary').click()
        page.locator('#outcomeSearch').fill('003')
        assert page.locator('#outcomeChoices label:visible').count() == 1
        page.keyboard.press('Escape')
        assert not page.locator('#outcomePicker').evaluate('e=>e.open')
        assert not errors and not remote
    finally:
        page.close()


def test_search_selection_options_and_empty_state(browser, tmp_path):
    page, errors, _ = open_page(browser, viewer(tmp_path))
    try:
        assert page.locator('.edge').count() == 1  # connections are visible by default
        assert page.locator('#showEdges').is_checked()
        page.locator('.hit[data-node="0"]').focus()
        page.keyboard.press('Enter')
        assert page.locator('#detail h2').first.inner_text() == 'Imaging pattern'
        page.locator('#search').fill('does not exist')
        assert page.locator('#emptyState').is_visible()
        assert '0 of 2' in page.locator('#mapStatus').inner_text()
        assert page.locator('#intro').is_visible()
        assert page.locator('.hit[tabindex="0"]').count() == 0
        page.locator('#displaySettings>summary').click()
        page.locator('#filterReset').click()
        assert not page.locator('#emptyState').is_visible()
        page.locator('#showEdges').check()
        assert page.locator('.edge').count() == 1
        page.locator('input[name="associationDirection"][value="risk"]').check()
        page.locator('#onlySignificant').check()
        assert page.locator('#emptyState').is_visible()  # selected outcome is protective
        page.locator('#filterReset').click()
        page.locator('#displaySettings>summary').click()
        page.locator('.hit[data-node="0"]').click()
        page.locator('#clearSelection').click()
        assert page.locator('#intro').is_visible()
        assert not errors
    finally:
        page.close()


def test_zoom_anchor_pan_and_mobile(browser, tmp_path):
    path = viewer(tmp_path,150)
    page, errors, remote = open_page(browser,path)
    try:
        coordinates = '''()=>{const p=svg.createSVGPoint();p.x=300;p.y=300;const q=p.matrixTransform(camera.getScreenCTM());return {x:q.x,y:q.y}}'''
        before = page.evaluate(coordinates)
        page.mouse.move(before['x'],before['y'])
        page.mouse.wheel(0,-100)
        page.wait_for_function('scale>1')
        after = page.evaluate(coordinates)
        assert abs(before['x']-after['x']) < 1 and abs(before['y']-after['y']) < 1
        page.mouse.move(500,450)
        page.mouse.down()
        page.mouse.move(580,490,steps=5)
        page.mouse.up()
        assert page.evaluate('moved') > 4
        page.locator('#reset').click()
        assert page.evaluate('scale') == 1
        assert page.locator('#intro').is_visible()
        page.set_viewport_size(dict(width=390,height=844))
        page.locator('.hit[data-node="0"]').click()
        assert page.locator('.wheel-spoke').count() == 150
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.locator('#displaySettings>summary').click()
        bounds=page.locator('#displaySettings>.outcomes').bounding_box()
        assert bounds['x'] >= 0 and bounds['x']+bounds['width'] <= 390
        page.locator('#displaySettings>summary').click()
        page.screenshot(path=str(tmp_path/'mobile-150.png'))
        assert not errors and not remote
    finally:
        page.close()


def test_full_view_retains_rankings(browser,tmp_path):
    page,errors,_=open_page(browser,viewer(tmp_path,view='full'))
    try:
        if page.locator('#tourCard').is_visible():page.locator('#tourSkip').click()
        page.locator('#rankingTab').click()
        assert page.locator('#rankTable tbody tr').count()==2
        page.locator('#rankTable tbody tr').first.click()
        assert page.locator('.wheel-spoke').count()==4
        assert not errors
    finally:
        page.close()


def test_wheel_effects_extend_inward_and_outward_from_one(browser,tmp_path):
    page,errors,_=open_page(browser,viewer(tmp_path,9))
    try:
        page.locator('.hit[data-node="0"]').click()
        references=page.locator('.wheel-reference').evaluate_all("rows=>rows.map(e=>({or:Number(e.dataset.or),r:Number(e.getAttribute('r'))}))")
        baseline=next(r['r'] for r in references if r['or']==1)
        assert baseline==(min(r['r'] for r in references)+max(r['r'] for r in references))/2
        assert page.locator('.wheel-reference[data-or="1"]').count()==1
        effects=page.locator('.wheel-effect').evaluate_all("rows=>rows.map(e=>({or:Number(e.dataset.or),start:Number(e.dataset.startRadius),end:Number(e.dataset.endRadius),fill:e.getAttribute('fill')}))")
        assert any(e['or']<1 and e['end']<e['start'] for e in effects)
        assert any(e['or']>1 and e['end']>e['start'] for e in effects)
        assert all(not e['fill'].startswith('url(') for e in effects)
        assert all(e['or']!=1 for e in effects)
        assert len(effects)==8
        assert not errors
    finally:
        page.close()


def test_hosted_url_opens_directly_on_viewer(browser,tmp_path):
    from glyph.visualization.hosting import serve_html
    with serve_html(viewer(tmp_path),open_browser=False) as server:
        page=browser.new_page()
        try:
            page.goto(server.url)
            assert page.locator('#graph').is_visible()
            assert page.locator('.hit').count()==2
            page.locator('.hit[data-node="0"]').click()
            assert page.locator('.wheel-spoke').count()==4
        finally:
            page.close()


def test_no_label_exploration_is_usable(browser,tmp_path):
    path=viewer(tmp_path,count=0)
    page,errors,_=open_page(browser,path)
    try:
        assert page.locator('#outcomePicker').is_hidden()
        page.locator('.hit[data-node="0"]').click()
        assert 'No association testing has been performed' in page.locator('#detail').inner_text()
        assert page.locator('.wheel-spoke').count()==0
        assert page.locator('.hit').count()==2
        assert not errors
    finally:page.close()


def test_interactive_node_area_is_proportional_to_prevalence(browser,tmp_path):
    page=browser.new_page();page.goto(viewer(tmp_path).as_uri())
    measured=page.evaluate("""() => {
        const low=radius({rate:10}),high=radius({rate:40});
        return {ratio:high**2/low**2,zero:radius({rate:0}),settings:NODE_SIZING};
    }""")
    assert measured['ratio']==pytest.approx(4.)
    assert measured['zero']==pytest.approx(.35)
    assert measured['settings']['full_prevalence_area']==pytest.approx(np.pi*14**2)
    page.close()


def test_full_network_renderer_has_sizing_metadata_and_visible_connections(browser,tmp_path):
    import torch
    from glyph.visualization import build_interactive_concept_network
    features={'train':torch.tensor([[1.,1.],[1.,0.],[0.,1.],[0.,0.]])}
    rows=pd.DataFrame([dict(feature_idx=i,feature_name=f'feature_{i}',outcome='y',split='train',modality='image',top_k=1,odds_ratio=1.5,odds_ratio_p_value=.2) for i in range(2)])
    output=build_interactive_concept_network(features,torch.ones(2,dtype=torch.bool),rows,tmp_path/'full-network.html','image',1,view='full')
    page,errors,_=open_page(browser,output)
    assert page.locator('.node-group').count()==2
    assert page.evaluate('Number.isFinite(radius(DATA.nodes[0]))')
    assert not errors
    page.close()


def test_correlation_lines_are_thin_and_strength_scaled(browser,tmp_path):
    page=browser.new_page();page.goto(viewer(tmp_path).as_uri())
    line=page.locator('.edge').first
    assert float(line.get_attribute('stroke-width'))==pytest.approx(.30*.8)
    assert float(line.get_attribute('stroke-opacity'))==pytest.approx(.14*.8)
    page.close()


def test_sidebar_shows_real_embedded_examples_and_keeps_unlabelled_exploration(browser,tmp_path):
    from glyph.visualization.image_examples import collect_image_examples
    from PIL import Image
    path=tmp_path/'input.png';Image.new('RGB',(16,16),'teal').save(path)
    metadata=pd.DataFrame(dict(sample_id=['first','second'],image_path=[str(path),str(path)]))
    gallery=collect_image_examples(np.array([[2.],[1.]]),metadata.sample_id,metadata,'image_path')
    gallery['concept_images']['a']=gallery['concept_images'].pop('0')
    page,errors,remote=open_page(browser,viewer(tmp_path,count=0,image_examples=gallery))
    try:
        page.locator('.hit[data-node="0"]').click()
        assert page.locator('.concept-images img').count()==2
        assert page.locator('.concept-images img').evaluate_all('(xs)=>xs.every(x=>x.complete && x.naturalWidth===16)')
        assert 'first' in page.locator('.concept-images').inner_text()
        assert 'activation 2.00' in page.locator('.concept-images').inner_text()
        assert page.locator('.wheel-spoke').count()==0
        assert not errors and not remote
    finally:page.close()
