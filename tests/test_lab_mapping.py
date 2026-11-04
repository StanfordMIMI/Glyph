from glyph.extraction.lab_mapping import resolve_lab_ids
import pandas as pd
import pytest


def mapping_and_stats():
    mapping = pd.DataFrame({"lab_id": [1, 2, 3, 4], "loinc_code": ["123-4", "123-4", None, None],
                            "selected_code": ["123-4", "123-4", "STANFORD_LAB//local", ""],
                            "selected_code_system": ["LOINC", "LOINC", "STANFORD_LAB", "unresolved"]})
    stats = pd.DataFrame({"lab_id": [1, 2, 3, 4], "retained": [True]*4,
                          "name": ["Glucose (mg/dL)", "Glucose (mmol/L)", "Local (no units)", "pH (no units)"]})
    return mapping, stats


def test_ambiguous_code_never_silently_overwrites():
    mapping, stats = mapping_and_stats()
    labs = pd.DataFrame({"loinc_code": ["LOINC/123-4"]})
    with pytest.raises(ValueError, match="multiple model labs"):
        resolve_lab_ids(labs, mapping, stats)
    labs["unit"] = "mmol/L"
    assert resolve_lab_ids(labs, mapping, stats) == [2]
    labs["unit"] = "wrong"
    with pytest.raises(ValueError, match="no unit conversion"):
        resolve_lab_ids(labs, mapping, stats)


def test_local_codes_are_separate_from_unresolved_entries():
    mapping, stats = mapping_and_stats()
    labs = pd.DataFrame({"loinc_code": ["STANFORD_LAB//local", "", "999-9"]})
    assert resolve_lab_ids(labs, mapping, stats) == [3, None, None]


def test_explicit_id_and_lab_name_disambiguation():
    mapping, stats = mapping_and_stats()
    labs = pd.DataFrame({"loinc_code": ["123-4"], "lab_id": [1]})
    assert resolve_lab_ids(labs, mapping, stats) == [1]
    labs = pd.DataFrame({"loinc_code": ["123-4"], "lab_name": ["Glucose (mmol/L)"]})
    assert resolve_lab_ids(labs, mapping, stats) == [2]


def test_bundle_export_preserves_review_and_audit(tmp_path, monkeypatch):
    import json
    import hashlib
    import torch
    from glyph.extraction import bundle
    stats=pd.DataFrame([dict(lab_id=1,name="Test (mg/dL)",is_numeric=True,retained=True,mean=0.,std=1.)])
    stats.to_pickle(tmp_path/"stats.pkl")
    mapping, _ = mapping_and_stats()
    mapping["status"] = "provisional"
    mapping["later_source_agreement"] = "conflicts with proposal"
    mapping.to_csv(tmp_path/"lab_loinc_map.csv",index=False)
    (tmp_path/"lab_loinc_source_audit.csv").write_text("lab_id,agreement\n1,conflicts with proposal\n")
    (tmp_path/"lab_loinc_mapping_provenance.json").write_text(json.dumps({"version":"source-first-2026-09-28"}))
    monkeypatch.setattr(bundle,"LabsEmbedder",lambda *a:torch.nn.Linear(1,1))
    out=bundle.export_labs_bundle(tmp_path/"unused.ckpt",tmp_path/"stats.pkl",tmp_path/"lab_loinc_map.csv",tmp_path/"bundle")
    saved=pd.read_csv(out/"lab_loinc_map.csv")
    assert saved.status.tolist()==["provisional"]
    assert saved.later_source_agreement.tolist()==["conflicts with proposal"]
    manifest=json.loads((out/"bundle.json").read_text())
    assert manifest["mapping_version"]=="source-first-2026-09-28"
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected
