"""Opt-in raw CT -> concepts comparison against a previously cached study.

Set GLYPH_CT_PARITY_CASE to a local JSON file containing ct_path,
embedding_cache, cache_index, and optionally activation_parquet and study_id.
No identifiers or image data are emitted to the test report.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


def test_raw_ct_matches_cached_embedding_and_concepts():
    case_path = os.environ.get("GLYPH_CT_PARITY_CASE")
    if not case_path:
        pytest.skip("Set GLYPH_CT_PARITY_CASE to run the real CT integration test")
    from glyph.extraction import extract_embeddings
    from glyph.zoo import load_pretrained
    case = json.loads(Path(case_path).read_text())
    extracted = extract_embeddings(pd.DataFrame({"sample_id": ["parity-sample"], "ct": [case["ct_path"]]}),
                                   {"ct": "merlin"}, batch_size=1, device=case.get("device", "cpu"))
    actual = extracted["train"]["embeddings"]["ct"]
    # The private research cache contains MONAI MetaTensor objects. Only use trusted local caches.
    cached = torch.load(case["embedding_cache"], map_location="cpu", weights_only=False)
    vectors = cached["vectors"] if "vectors" in cached else cached["embeddings"]
    expected = vectors[case["cache_index"]:case["cache_index"] + 1].float()
    if hasattr(expected, "as_tensor"):
        expected = expected.as_tensor()
    sae = load_pretrained("ct-merlin", cache_dir=case.get("sae_cache_dir", "cache/zoo"))
    concepts, expected_concepts = sae.encode(actual), sae.encode(expected)
    report = {"embedding_max_abs_error": float((actual - expected).abs().max()),
              "embedding_cosine": float(torch.nn.functional.cosine_similarity(actual, expected).item()),
              "concept_max_abs_error": float((concepts - expected_concepts).abs().max()),
              "active_concepts_match": bool(torch.equal(concepts > 0, expected_concepts > 0)),
              "embedding_relative_l2_error": float((actual - expected).norm() / expected.norm()),
              "concept_relative_l2_error": float((concepts - expected_concepts).norm() / expected_concepts.norm()),
              "strict_numeric_match": bool(torch.allclose(actual, expected, rtol=2e-4, atol=2e-4)),
              "sae_revision": sae.entry.revision}
    if "activation_parquet" in case:
        row = pd.read_parquet(case["activation_parquet"], filters=[("study_id", "==", case["study_id"])])
        assert len(row) == 1
        columns = [c for c in row if c.startswith("Concept_")]
        indices = [int(c.split("_")[1]) for c in columns]
        released = torch.tensor(row[columns].to_numpy(dtype=np.float32))
        report["released_concept_max_abs_error"] = float((concepts[:, indices] - released).abs().max())
        report["released_active_concepts_match"] = bool(torch.equal(concepts[:, indices] > 0, released > 0))
    Path(case.get("report_path", "test_outputs/ct_parity_report.json")).write_text(json.dumps(report, indent=2))
    # Cross-runtime acceptance: exact active IDs, cosine >= .99999, <= .5% L2 drift.
    # The report separately retains strict numerical parity; these are not bitwise claims.
    assert report["embedding_cosine"] >= .99999
    assert report["embedding_relative_l2_error"] <= .005
    assert report["concept_relative_l2_error"] <= .005
    assert report["active_concepts_match"]
    if "activation_parquet" in case:
        assert float((concepts[:, indices] - released).norm() / released.norm()) <= .005
        assert report["released_active_concepts_match"]
