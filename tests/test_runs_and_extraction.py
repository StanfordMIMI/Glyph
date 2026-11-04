import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from glyph import Glyph
from glyph.extraction import extract_embeddings, load_embeddings
from glyph.zoo.pretrained import PretrainedSAE
from glyph.zoo.registry import resolve


@pytest.fixture
def cohort(tmp_path):
    rng = np.random.default_rng(28)
    frame = pd.DataFrame({"sample_id": [f"s{i:03}" for i in range(80)],
                          "scan": [str(x) for x in rng.normal(size=80)],
                          "outcome": np.arange(80) % 2,
                          "split": ["train"] * 40 + ["val"] * 20 + ["test"] * 20})
    path = tmp_path / "metadata.csv"
    frame.to_csv(path, index=False)
    return frame, path


@pytest.fixture
def pipeline(cohort, tmp_path, monkeypatch):
    frame, path = cohort
    calls = []

    class FakeEncoder:
        def embed_batch(self, values):
            calls.extend(values)
            x = torch.tensor([float(v) for v in values])
            return torch.stack([x, x * x, x.sin(), x.cos()], dim=1)

    monkeypatch.setattr("glyph.extraction.api.get_model", lambda *args, **kw: FakeEncoder())
    monkeypatch.setattr("glyph.core.glyph.fit_linear_probe_from_embeddings",
                        lambda **kw: {"all_metrics": {"train": .5, "val": .5, "test": .5}})
    obj = Glyph("config", [f"data.metadata={path}", f"data.output_dir={tmp_path / 'runs'}",
                          "data.input_cols=[scan]", "data.outcome_cols=[outcome]",
                          "data.embedding_map={scan:resnet18}", "data.num_workers=0", "device=cpu",
                          "sae.top_ks=[2]", "sae.matryoshka=[8]", "sae.max_steps=3",
                          "sae.use_ghost_grads=false", "phewas.min_activations=1"])
    return obj, calls


def test_raw_extraction_recomputes_each_fit_and_retains_runs(pipeline):
    obj, calls = pipeline
    obj.fit()
    first = obj.run_dir
    original = (first / "fitted_saes.pt").read_bytes()
    assert len(calls) == 80
    obj.fit()
    assert len(calls) == 160
    assert first != obj.run_dir and first.parent == obj.run_dir.parent
    assert (first / "fitted_saes.pt").read_bytes() == original
    assert json.loads((first / "run.json").read_text())["status"] == "complete"
    assert json.loads((obj.run_dir / "run.json").read_text())["embedding_source"] == "raw_inputs"


def test_trained_run_reload_has_identical_concepts_and_ids(pipeline):
    obj, _ = pipeline
    obj.fit()
    loaded = Glyph.load_run(obj.run_dir, device="cpu")
    for split, data in obj.embeddings.items():
        actual = loaded.encode_embeddings(data["embeddings"]["scan"], modality="scan")
        torch.testing.assert_close(actual, obj.sae_features["scan"][split], rtol=0, atol=0)
        assert loaded.embeddings[split]["sample_ids"] == data["sample_ids"]
    assert torch.equal(loaded.sae_alive_features[("scan", 2)], obj.sae_alive_features[("scan", 2)])
    assert loaded.outcome_types == obj.outcome_types
    # Reloaded runs can be refitted into a new sibling run.
    loaded.fit(embeddings_path=obj.run_dir)
    assert loaded.run_dir != obj.run_dir


def test_explicit_embeddings_skip_encoder_and_align_by_id(pipeline, cohort, tmp_path, monkeypatch):
    obj, calls = pipeline
    frame, _ = cohort
    x = torch.randn(80, 4)
    file = tmp_path / "precomputed.pt"
    torch.save({"sample_ids": frame.sample_id.tolist()[::-1], "embeddings": x.flip(0)}, file)
    monkeypatch.setattr("glyph.extraction.api.get_model", lambda *a, **kw: pytest.fail("encoder loaded"))
    obj.fit(embeddings_path=file)
    assert calls == []
    assert torch.equal(obj.embeddings["train"]["embeddings"]["scan"], x[:40])
    assert json.loads((obj.run_dir / "run.json").read_text())["embedding_source"] == "explicit_files"


def test_no_implicit_cache_reuse(pipeline):
    obj, calls = pipeline
    with pytest.raises(ValueError, match="explicit embeddings_path"):
        obj.fit(recompute_embeddings=False)
    assert calls == []


def test_failure_marks_run_and_keeps_prior_run(pipeline, monkeypatch):
    obj, _ = pipeline
    obj.fit()
    first = obj.run_dir
    def fail(*a, **kw):
        raise ValueError("encoder error")
    monkeypatch.setattr("glyph.extraction.api.get_model", fail)
    with pytest.raises(ValueError, match="encoder error"):
        obj.fit()
    assert json.loads((obj.run_dir / "run.json").read_text())["status"] == "failed"
    assert json.loads((first / "run.json").read_text())["status"] == "complete"


def test_pretrained_fit_and_reload_preserve_published_activations(pipeline, cohort, tmp_path, monkeypatch):
    obj, _ = pipeline
    frame, _ = cohort
    entry = replace(resolve("ct-merlin"), input_dim=4, n_concepts=8, top_k=2)
    torch.manual_seed(11)
    bundle = PretrainedSAE(entry, torch.randn(4, 8), torch.randn(8),
                          {"mean": [0.] * 4, "std": 1., "dim_scale": 2048 ** .5 / 2},
                          pd.DataFrame({"feature_name": ["Concept_3"],
                                        "extracted_interpretation": ["Example concept"],
                                        "test_interpretation_discrimination_accuracy": [.8]}))
    monkeypatch.setattr("glyph.zoo.load_pretrained", lambda *a, **kw: bundle)
    monkeypatch.setattr("glyph.core.runs.SAE.fit", lambda *a, **kw: pytest.fail("pretrained SAE retrained"))
    x = torch.randn(80, 4)
    file = tmp_path / "embeddings.pt"
    torch.save({"sample_ids": frame.sample_id.tolist(), "embeddings": x}, file)
    obj.fit(pretrained={"scan": "ct-merlin"}, embeddings_path=file)
    assert obj.cfg.data.embedding_map.scan == "merlin"
    assert obj.autointerp_results.feature_idx.tolist() == [3]
    assert obj.autointerp_results.interpretation.tolist() == ["Example concept"]
    assert obj.sae_models["scan"].selection_report["revision"] == entry.revision
    obj.hypothesis_search()
    assert obj.phewas_results is not None and not obj.phewas_results.empty
    assert not list(obj.run_dir.glob("*.html"))  # HTML is optional by default.
    obj.visualize()
    assert any("Example concept" in file.read_text() for file in obj.run_dir.glob("*.html"))
    loaded = Glyph.load_run(obj.run_dir, device="cpu")
    pd.testing.assert_frame_equal(loaded.phewas_results, obj.phewas_results)
    for split, data in obj.embeddings.items():
        expected = bundle.encode(data["embeddings"]["scan"])
        torch.testing.assert_close(obj.sae_features["scan"][split], expected, atol=0, rtol=0)
        torch.testing.assert_close(loaded.encode_embeddings(data["embeddings"]["scan"], modality="scan"), expected, atol=0, rtol=0)
    assert loaded.autointerp_results.feature_idx.tolist() == [3]
    expected = obj.sae_features['scan']['test'][:, 3].numpy()
    np.testing.assert_array_equal(obj.measure('I3', split='test').to_numpy(), expected)
    pd.testing.assert_series_equal(loaded.measure('I3', split='test'), obj.measure('I3', split='test'))
    assert loaded.measure('I3', split='test').index.tolist() == obj.embeddings['test']['sample_ids']


def test_embedding_import_rejects_duplicates_missing_and_split_mismatch(cohort, tmp_path):
    frame, _ = cohort
    path = tmp_path / "input.pt"
    ids = frame.sample_id.tolist()
    x = torch.randn(80, 4)
    for payload, message in [
        ({"sample_ids": ids[:-1] + [ids[0]], "embeddings": x}, "duplicate"),
        ({"sample_ids": ids[:-1], "embeddings": x[:-1]}, "lack embeddings"),
        ({"sample_ids": ids, "embeddings": x, "split": ["test"] * 80}, "splits differ"),
        ({"sample_ids": ids, "embeddings": x * float("nan")}, "finite"),
    ]:
        torch.save(payload, path)
        with pytest.raises(ValueError, match=message):
            load_embeddings(path, frame, ["scan"])


def test_labs_without_any_release_pin_requires_explicit_bundle(monkeypatch):
    from glyph.extraction.labs import LabsEncoder
    monkeypatch.setattr("glyph.extraction.labs.LABS_ENCODER_REVISION", None)
    with pytest.raises(ValueError, match="not yet been publicly released"):
        LabsEncoder()


def test_labs_preparation_preserves_reference_window_and_patient_id_types(tmp_path):
    from glyph.extraction.labs import prepare_labs
    pd.DataFrame({"lab_id": [1], "is_numeric": [True], "retained": [True]}).to_json(
        tmp_path / "lab_metadata.json", orient="records")
    pd.DataFrame({"loinc_code": ["123-4"], "lab_id": [1]}).to_csv(tmp_path / "lab_loinc_map.csv", index=False)
    metadata = pd.DataFrame({"sample_id": ["a"], "patient_id": [123], "date": ["2024-07-01"]})
    labs = pd.DataFrame({"patient_id": ["123"] * 5, "date": ["2023-12-31", "2024-02-01", "2024-03-01", "2024-08-30", "2024-08-31"],
                         "loinc_code": ["LOINC/123-4"] * 5, "value": [999, 1, 2, 3, 999]})
    samples = prepare_labs(metadata, labs, tmp_path, profile="hypothesisextractor-v1")
    assert [v["Value"] for v in samples["a"]] == [2., 3.]
    # The published preprocessing timestamps each aggregated value at its quarter start.
    assert samples["a"][-1]["years_prior"] == 0


def test_labs_bundle_roundtrip_and_checksum_validation(tmp_path):
    from glyph.extraction.bundle import export_labs_bundle
    from glyph.extraction.labs import LabsEmbedder, LabsEncoder
    stats = pd.DataFrame({"lab_id": [1, 2], "is_numeric": [True, False],
                          "retained": [True, True], "mean": [2., 0.], "std": [1., 1.],
                          "categories": [[], ["positive", "negative"]]})
    torch.manual_seed(78)
    original = LabsEmbedder(stats).eval()
    checkpoint = tmp_path / "original.ckpt"
    torch.save({"state_dict": original.state_dict()}, checkpoint)
    stats.to_pickle(tmp_path / "stats.pkl")
    pd.DataFrame({"loinc_code": ["123-4", "567-8"], "lab_id": [1, 2]}).to_csv(tmp_path / "map.csv", index=False)
    (tmp_path / "lab_loinc_source_audit.csv").write_text("lab_id,agreement\n1,matches proposal\n")
    output = export_labs_bundle(checkpoint, tmp_path / "stats.pkl", tmp_path / "map.csv", tmp_path / "bundle")
    encoder = LabsEncoder(bundle_dir=output)
    samples = [[{"lab_id": 1, "Value": 2.5, "years_prior": .1},
                {"lab_id": 2, "Value": "positive", "years_prior": .2}],
               [{"lab_id": 1, "Value": -2., "years_prior": -.1}]]
    with torch.inference_mode():
        expected = original(samples)
    torch.testing.assert_close(encoder.embed_batch(samples), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="no eligible labs"):
        encoder.embed_batch([[]])
    audit = output / "lab_loinc_source_audit.csv"
    before = audit.read_text()
    audit.write_text("altered")
    with pytest.raises(ValueError, match="checksum"):
        LabsEncoder(bundle_dir=output)
    audit.write_text(before)
    (output / "lab_loinc_map.csv").write_text("altered")
    with pytest.raises(ValueError, match="checksum"):
        LabsEncoder(bundle_dir=output)


def test_qwen_adapter_uses_findings_eos_last_token_and_l2_pooling(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    from glyph.extraction.encoders import Qwen3Encoder
    observed = {}

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token = "<eos>"
        def __call__(self, texts, **kwargs):
            observed.update(texts=texts, **kwargs)
            return Batch(attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]))

    class Model:
        def eval(self): return self
        def to(self, device): return self
        def __call__(self, **kwargs):
            hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
            return SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
        AutoModel=SimpleNamespace(from_pretrained=lambda *a, **kw: Model())))
    paths = [tmp_path / "report1.txt", tmp_path / "report2.txt"]
    paths[0].write_text("HISTORY: example\nFINDINGS: sample finding\nIMPRESSION: summary")
    paths[1].write_text("No section headers")
    encoder = Qwen3Encoder()
    output = encoder.embed_batch(paths)
    expected = torch.nn.functional.normalize(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)[:, -1], dim=1)
    assert observed["texts"] == ["sample finding<eos>", "No section headers<eos>"]
    assert observed["max_length"] == 8192
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_precomputed_cohort_does_not_need_raw_columns(pipeline, cohort, tmp_path):
    obj, _ = pipeline
    frame, metadata_path = cohort
    frame.drop(columns="scan").to_csv(metadata_path, index=False)
    imported = Glyph("config", [f"data.metadata={metadata_path}", f"data.output_dir={tmp_path / 'other_runs'}",
                     "data.input_cols=[scan]", "data.outcome_cols=[outcome]", "data.embedding_map={scan:merlin}",
                     "data.num_workers=0", "device=cpu", "sae.top_ks=[2]", "sae.matryoshka=[8]",
                     "sae.max_steps=3", "sae.use_ghost_grads=false", "phewas.min_activations=1"])
    path = tmp_path / "raw_embeddings.pt"
    torch.save({"sample_ids": frame.sample_id.tolist(), "embeddings": torch.randn(80, 4)}, path)
    imported.fit(embeddings_path=path)
    with pytest.raises(ValueError, match="lacks input columns"):
        imported.fit()


def test_run_snapshot_redacts_credentials_without_erasing_token_limits(pipeline):
    from omegaconf import OmegaConf
    obj, _ = pipeline
    OmegaConf.update(obj.cfg, "autointerp.llm.api_key", "test-credential", force_add=True)
    OmegaConf.update(obj.cfg, "autointerp.llm.max_tokens", 123, force_add=True)
    obj.fit()
    config = OmegaConf.load(obj.run_dir / "config.yaml")
    assert config.autointerp.llm.api_key is None
    assert config.autointerp.llm.max_tokens == 123


def test_precomputed_rejects_a_declared_wrong_encoder(cohort, tmp_path):
    frame, _ = cohort
    path = tmp_path / "wrong_encoder.pt"
    torch.save({"sample_ids": frame.sample_id.tolist(), "embeddings": torch.randn(80, 4), "encoder": "resnet18"}, path)
    with pytest.raises(ValueError, match="saved encoder"):
        load_embeddings(path, frame, {"scan": "merlin"})


def test_merlin_batch_axis_is_preserved(monkeypatch):
    from glyph.extraction.encoders import MerlinEncoder
    encoder = MerlinEncoder.__new__(MerlinEncoder)
    encoder.device = "cpu"
    encoder.transform = lambda item: {"image": torch.ones(1, 2, 2, 2)}
    for shape in [(1, 2, 2048), (2, 1, 2048)]:
        encoder.model = lambda x: torch.ones(*shape)
        assert encoder.embed_batch(["a", "b"]).shape == (2, 2048)


def test_unlabelled_fit_is_reproducible_without_probes(pipeline,monkeypatch):
    obj,_=pipeline
    obj.cfg.data.outcome_cols=[]
    obj.outcome_types={}  # Match initialization of an actual no-label configuration.
    monkeypatch.setattr('glyph.core.glyph.fit_linear_probe_from_embeddings',
        lambda **kw: pytest.fail('A no-label run must not call an outcome probe'))
    obj.fit()
    expected=obj.sae_features['scan']['test'].clone()
    obj.fit()
    torch.testing.assert_close(obj.sae_features['scan']['test'],expected,rtol=0,atol=0)
    assert obj.outcomes=={'train':{},'val':{},'test':{}}
    loaded=Glyph.load_run(obj.run_dir,device='cpu')
    assert loaded.outcome_types=={}
    with pytest.raises(ValueError,match='requires outcome labels'):
        loaded.hypothesis_search()


def test_local_activation_release_preserves_concepts_and_uses_only_mapped_public_ids(pipeline,tmp_path):
    import importlib.util
    from scipy.sparse import load_npz
    script=Path(__file__).parents[1]/'scripts/prepare_activation_release.py'
    spec=importlib.util.spec_from_file_location('activation_release',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    obj,_=pipeline;obj.fit()
    ids=[str(sample) for data in obj.embeddings.values() for sample in data['sample_ids']]
    mapping=pd.DataFrame({'sample_id':ids,'public_sample_id':[f'public-{i}' for i in range(len(ids))]})
    path=tmp_path/'mapping.csv';mapping.to_csv(path,index=False)
    manifest=module.prepare(obj.run_dir,path,tmp_path/'release')
    actual=load_npz(tmp_path/'release/activations_scan.npz').toarray()
    expected=np.concatenate([obj.sae_features['scan'][split].numpy() for split in obj.embeddings])
    np.testing.assert_array_equal(actual,expected)
    samples=pd.read_csv(tmp_path/'release/samples.csv')
    assert samples.columns.tolist()==['public_sample_id','split']
    assert samples.public_sample_id.tolist()==mapping.public_sample_id.tolist()
    assert manifest['status']=='local_staging_only_not_published'
    assert manifest['models'][0]['has_decoder']
    for path in (tmp_path/'release').glob('*.json'):
        assert not any(f'"{sample}"' in path.read_text() for sample in ids)


def test_activation_release_refuses_incomplete_mapping(pipeline,tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('activation_release',Path(__file__).parents[1]/'scripts/prepare_activation_release.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    obj,_=pipeline;obj.fit()
    path=tmp_path/'mapping.csv';pd.DataFrame({'sample_id':['missing'],'public_sample_id':['public']}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='cover every run sample'):module.prepare(obj.run_dir,path,tmp_path/'release')
    assert not (tmp_path/'release').exists()
