import pandas as pd
import torch
from omegaconf import OmegaConf

from glyph import Glyph
from glyph.visualization import build_interactive_concept_atlas, build_interactive_concept_network


def test_build_interactive_concept_network(tmp_path):
    features = {
        "train": torch.tensor([
            [1.0, 0.0, 0.3],
            [0.7, 0.0, 0.2],
            [0.0, 1.0, 0.5],
            [0.0, 0.8, 0.4],
        ])
    }
    rows = []
    for feature_idx in range(3):
        for outcome_idx, outcome in enumerate(("recovery", "toxicity")):
            rows.append({
                "feature_idx": feature_idx,
                "feature_name": f"feature_{feature_idx}",
                "outcome": outcome,
                "split": "train",
                "modality": "assay",
                "top_k": 2,
                "c_f": 2,
                "odds_ratio": 1.5 + feature_idx if outcome_idx == 0 else 0.7,
                "odds_ratio_p_value": 0.001 if feature_idx == 0 else 0.5,
            })
    interpretations = pd.DataFrame([{
        "feature_idx": 0,
        "feature_name": "feature_0",
        "modality": "assay",
        "top_k": 2,
        "interpretation": "A </script><distinctive> pattern",
        "test_accuracy": 0.9,
    }])
    output = tmp_path / "map.html"

    result = build_interactive_concept_network(
        sae_features=features,
        alive_mask=torch.tensor([True, True, True]),
        phewas_results=pd.DataFrame(rows),
        output_path=output,
        modality="assay",
        top_k=2,
        initial_outcome="toxicity",
        correlation_percentile=0.5,
        autointerp_results=interpretations,
    )

    html = output.read_text()
    assert result == output
    assert "__GLYPH_DATA__" not in html
    assert "feature_0" in html
    assert "recovery" in html
    assert "toxicity" in html
    assert "A </script><distinctive> pattern" not in html
    assert r"A <\/script><distinctive> pattern" in html
    assert "Scroll to zoom and drag to pan" in html
    assert "Outcome wheel" in html
    assert "outcomeChoices" in html
    assert "selectedHits" in html
    assert "function wedge" in html
    assert "outcomeColor" in html
    assert html.count('"significant":true') == 2


def test_single_concept_does_not_require_umap(tmp_path):
    hypotheses = pd.DataFrame([{
        "feature_idx": 0,
        "outcome": "result",
        "split": "train",
        "modality": "text",
        "top_k": 1,
        "c_f": 2,
        "odds_ratio": 2.0,
        "odds_ratio_p_value": 0.01,
    }])

    output = build_interactive_concept_network(
        sae_features={"train": torch.tensor([[1.0], [0.0]])},
        alive_mask=torch.tensor([True]),
        phewas_results=hypotheses,
        output_path=tmp_path / "one.html",
        modality="text",
        top_k=1,
    )

    assert output.exists()
    assert "feature_0" in output.read_text()


def test_build_multimodal_concept_atlas(tmp_path):
    concepts = pd.DataFrame([
        {"concept_id": "gator:ehr:0", "feature_idx": 0, "name": "ehr_0", "x": .1, "y": .2,
         "modality": "ehr", "embedding_family": "gator", "active": 3, "rate": 75},
        {"concept_id": "indomain:image:1", "feature_idx": 1, "name": "image_1", "x": .8, "y": .7,
         "modality": "image", "embedding_family": "indomain", "active": 2, "rate": 50},
    ])
    associations = pd.DataFrame([
        {"concept_id": "gator:ehr:0", "outcome": "CKM stage 4", "odds_ratio": 2.1,
         "odds_ratio_p_value": .001, "odds_ratio_p_value_bonferroni": .02,
         "odds_ratio_p_value_bonferroni_modality": .0005,
         "bonferroni_significant": True},
        {"concept_id": "indomain:image:1", "outcome": "CKM stage 4", "odds_ratio": .8,
         "odds_ratio_p_value": .2, "odds_ratio_p_value_bonferroni": 1.0,
         "odds_ratio_p_value_bonferroni_modality": .4,
         "bonferroni_significant": False},
    ])
    output = tmp_path / "atlas.html"

    result = build_interactive_concept_atlas(
        concepts, associations, output,
        edges=[("gator:ehr:0", "indomain:image:1", .7)],
        selection_note="2 of 10 concepts shown",
        technical_details={
            "method_note": "Pipeline metrics",
            "rows": [{
                "modality": "ehr", "input_dim": 384, "dict_size": 4096,
                "top_k": 40, "n_alive": 1200,
                "sample_counts": {"train": 10, "val": 3, "test": 4},
                "reconstruction_r2": {"train": .8, "val": .7, "test": .6},
                "aucs": [{"outcome": "CKM stage 4", "embedding": {"test": .7},
                          "sae": {"test": .72}}],
            }],
        },
        outcome_groups={"CKM stages": ["CKM stage 4"], "Transitions": ["2 → 4"]},
        outcome_colors={"2 → 4": "#DC2626"},
    )

    html = result.read_text()
    assert '"modalities":["ehr","image"]' in html
    assert '"embedding_families":["gator","indomain"]' in html
    assert "Modalities / families" in html
    assert "Color: outcomes" in html
    assert "Color: modality" in html
    assert "stroke-opacity:.38" in html
    assert "extremeOnly=!isSimple" in html
    assert "Protective (OR &lt; 1)" in html
    assert "protectiveFill" in html
    assert "▲ predictive" in html
    assert "▼ protective" in html
    assert "Math.max(5,Math.ceil(risk.length*.1))" in html
    assert "Math.max(5,Math.ceil(protective.length*.1))" in html
    assert "updateEdgeVisibility" in html
    assert 'id="edges"' not in html
    assert '<g id="edgeLayer"></g>' in html
    assert "2 of 10 concepts shown" in html
    assert "Significant associations" in html
    assert "renderRankings" in html
    assert "modality_corrected_p_value" in html
    assert "Technical details" in html
    assert "Pipeline metrics" in html
    assert "Reconstruction R²" in html
    assert "wheel_label_limit" in html
    assert "outcomeOrder=new Map(DATA.outcomes.map" in html
    assert "Hover or focus a spoke for statistics" in html
    assert "document.title=DATA.title" in html
    assert "Δ SAE−embedding" in html
    assert '"outcome_groups":{"CKM stages":["CKM stage 4"]}' in html
    assert '"2 → 4":"#DC2626"' in html
    assert '"CKM stage 4":"#0d9488"' in html
    assert "STAGE_PALETTE" not in html


def test_glyph_visualize_writes_public_html(tmp_path):
    glyph = Glyph.__new__(Glyph)
    glyph.cfg = OmegaConf.create({
        "seed": 42,
        "data": {
            "output_dir": str(tmp_path),
            "outcome_cols": ["result"],
            "embedding_map": {"text": "precomputed"},
        },
        "visualization": {
            "interactive": True,
            "static_jpg": False,
            "correlation_percentile": 0.975,
            "bonferroni_alpha": 0.05,
            "figsize": [14, 14],
            "dpi": 150,
            "layout_k": None,
        },
    })
    glyph.sae_features = {"text": {"train": torch.tensor([[1.0], [0.0]])}}
    glyph.sae_alive_features = {("text", 1): torch.tensor([True])}
    glyph.phewas_results = pd.DataFrame([{
        "feature_idx": 0,
        "outcome": "result",
        "split": "train",
        "modality": "text",
        "top_k": 1,
        "c_f": 1,
        "odds_ratio": 2.0,
        "odds_ratio_p_value": 0.01,
    }])
    glyph.autointerp_results = None

    generated = glyph.visualize(serve=True, open_browser=False)
    try:
        from urllib.request import urlopen
        with urlopen(glyph.viewer.url) as response:
            assert b'id="graph"' in response.read()
    finally:
        glyph.viewer.close()

    assert generated == [tmp_path / "concept_map_text_k1_train.html"]
    assert generated[0].exists()


def test_exact_p_values_confidence_intervals_and_optional_support(tmp_path):
    import json
    import re
    concepts=pd.DataFrame([dict(concept_id='x',x=.5,y=.5,modality='image')])
    associations=pd.DataFrame([dict(concept_id='x',outcome='outcome',odds_ratio=1.2,
        odds_ratio_p_value=1e-30,odds_ratio_p_value_bonferroni=1e-27,
        odds_ratio_ci_lower=1.1,odds_ratio_ci_upper=1.3)])
    output=build_interactive_concept_atlas(concepts,associations,tmp_path/'precise.html',wheel_label_limit=12)
    data=json.loads(re.search(r'const DATA=(.*?),NS=',output.read_text()).group(1))
    association=data['associations']['0'][0]
    assert association['p_value']==1e-30 and association['corrected_p_value']==1e-27
    assert association['ci_lower']==1.1 and association['ci_upper']==1.3
    assert association['count'] is None
    assert data['wheel_label_limit']==12


def test_invalid_atlas_coordinates_and_edges_are_rejected(tmp_path):
    import pytest
    concepts=pd.DataFrame([dict(concept_id='x',x=float('nan'),y=.5,modality='image')])
    associations=pd.DataFrame([dict(concept_id='x',outcome='outcome',odds_ratio=1.2)])
    with pytest.raises(ValueError,match='finite'):
        build_interactive_concept_atlas(concepts,associations,tmp_path/'bad.html')
    concepts['x']=.5
    with pytest.raises(ValueError,match='unknown concept'):
        build_interactive_concept_atlas(concepts,associations,tmp_path/'bad-edge.html',edges=[('x','missing',.5)])


def test_exploration_without_labels_preserves_every_alive_concept(tmp_path):
    import json
    import re
    output = build_interactive_concept_network(
        sae_features={'train': torch.tensor([[1.,0.,1.],[0.,1.,.5],[1.,1.,0.],[.3,.4,.2]])},
        alive_mask=torch.tensor([True,True,True]), phewas_results=None,
        output_path=tmp_path/'unlabelled.html', modality='input',top_k=2)
    payload=json.loads(re.search(r'const DATA=(.*?),NS=',output.read_text()).group(1))
    assert len(payload['nodes'])==3
    assert payload['outcomes']==[] and payload['initial_outcome'] is None
    assert all(not rows for rows in payload['associations'].values())


def test_association_map_retains_alive_concepts_without_test_rows(tmp_path):
    import json,re
    features={'train':torch.tensor([[1.,0.],[0.,1.],[.2,.3],[.3,.2]])}
    rows=pd.DataFrame([dict(feature_idx=0,outcome='y',split='train',modality='x',top_k=1,
        odds_ratio=2.,odds_ratio_p_value_bonferroni=.01)])
    path=build_interactive_concept_network(sae_features=features,alive_mask=torch.tensor([True,True]),
        phewas_results=rows,output_path=tmp_path/'map.html',modality='x',top_k=1)
    data=json.loads(re.search(r'const DATA=(.*?),NS=',path.read_text()).group(1))
    assert len(data['nodes'])==2
    assert data['associations']['1']==[]


def test_correlation_edges_keep_threshold_and_isolated_node_connections():
    import numpy as np
    from glyph.visualization.interactive_concept_network import _graph_edges
    correlation = np.array([[1., .9, 0., .2], [.9, 1., .5, 0.],
                            [0., .5, 1., .3], [.2, 0., .3, 1.]])
    assert _graph_edges(correlation, .9) == [[0, 1, .9], [1, 2, .5], [2, 3, .3]]
    assert _graph_edges(np.eye(4), .9) == []


def test_correlation_edges_enforce_eight_links_at_both_endpoints():
    import numpy as np
    from glyph.visualization.interactive_concept_network import _graph_edges
    from glyph.visualization.correlation_edges import limit_correlation_edges
    matrix=np.ones((20,20));np.fill_diagonal(matrix,1)
    edges=_graph_edges(matrix,0.)
    degree=np.bincount(np.array(edges)[:,:2].astype(int).ravel(),minlength=20)
    assert degree.max()<=8
    star=limit_correlation_edges([[0,j,1-j/100] for j in range(1,20)],20)
    assert [edge[1] for edge in star]==list(range(1,9))
