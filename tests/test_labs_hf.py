"""Public HF download -> lab preparation -> embeddings -> released SAE.

Run with GLYPH_HF_INTEGRATION=1. Uses synthetic inputs and an empty test cache.
No HF login or private clinical data is needed.
"""
import json
import os
from pathlib import Path
import pandas as pd
import pytest
import torch


def test_anonymous_labs_download_and_concepts(tmp_path, monkeypatch):
    if os.environ.get("GLYPH_HF_INTEGRATION") != "1":
        pytest.skip("Set GLYPH_HF_INTEGRATION=1 to download the public labs models")
    from huggingface_hub import HfApi
    from glyph.extraction.labs import LabsEncoder, prepare_lab_inputs, LABS_ENCODER_REPO, LABS_ENCODER_REVISION
    from glyph.extraction import extract_embeddings
    from glyph.zoo import load_pretrained
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    assert LABS_ENCODER_REVISION and len(LABS_ENCODER_REVISION) == 40
    info = HfApi(token=False).model_info(LABS_ENCODER_REPO, revision=LABS_ENCODER_REVISION)
    assert not info.private and not info.gated
    cache = tmp_path/"fresh-hf-cache"
    assert not cache.exists()
    encoder = LabsEncoder(cache_dir=cache, token=False)
    folder = encoder.bundle_dir
    assert LABS_ENCODER_REVISION in str(folder)
    assert (folder/"lab_loinc_source_audit.csv").is_file()
    mapping = pd.read_csv(folder/"lab_loinc_map.csv")
    assert len(mapping) == 350 and mapping.loinc_code.nunique() == 193
    assert mapping.later_source_agreement.eq("conflicts with proposal").sum() == 8
    row = mapping[mapping.loinc_code.notna()].iloc[0]
    stats = encoder.metadata.set_index("lab_id").loc[int(row.lab_id)]
    metadata = pd.DataFrame([dict(sample_id="synthetic", patient_id="synthetic", date="2024-07-01")])
    measurements = pd.DataFrame([dict(patient_id="synthetic", date="2024-05-01", loinc_code=row.loinc_code,
                                     lab_name=stats["name"], unit=stats.training_unit, value=str(float(stats["mean"])))])
    prepared = prepare_lab_inputs(metadata, measurements, folder, tmp_path/"prepared")
    frame = pd.read_csv(prepared)
    tokens = json.loads(Path(frame.labs_path.iloc[0]).read_text())
    direct = encoder.embed_batch([tokens])
    extracted = extract_embeddings(frame, {"labs_path":"labs"}, device="cpu",
                    model_kwargs={"labs_path":{"cache_dir":str(cache), "token":False}})
    actual = extracted["train"]["embeddings"]["labs_path"]
    assert actual.shape == (1,768) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, direct, rtol=0, atol=0)
    reference = os.environ.get("GLYPH_LABS_LOCAL_REFERENCE")
    if reference:
        expected = LabsEncoder(bundle_dir=reference).embed_batch([tokens])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sae = load_pretrained("labs-loinc", cache_dir=tmp_path/"fresh-sae-cache")
    # Synthetic values test the mechanics, not clinical distribution validity.
    concepts = sae.encode(actual, strict=False)
    assert concepts.shape == (1,8192) and torch.isfinite(concepts).all()
    assert 0 < (concepts > 0).sum() <= 40
    report = dict(repo_id=LABS_ENCODER_REPO, revision=LABS_ENCODER_REVISION,
                  anonymous_download=True, embedding_shape=list(actual.shape),
                  concept_shape=list(concepts.shape), active_concepts=int((concepts>0).sum()),
                  local_reference_checked=bool(reference), sae_revision=sae.entry.revision)
    (tmp_path/"labs_hf_report.json").write_text(json.dumps(report,indent=2))
