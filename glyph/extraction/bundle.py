"""Prepare a labs encoder release from trusted research artifacts.

This is a release tool, not an import-time dependency or an automatic upload.
Run with python -m glyph.extraction.bundle --help.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import torch
from glyph.extraction.labs import LabsEmbedder


def split_lab_name(name):
    text = str(name).strip()
    depth = 0
    if text.endswith(")"):
        for index in range(len(text) - 1, -1, -1):
            depth += (text[index] == ")") - (text[index] == "(")
            if depth == 0:
                return text[:index].strip(), text[index + 1:-1]
    return text, None


def export_labs_bundle(checkpoint, metadata_pickle, loinc_map, output_dir):
    # Only use trusted research metadata: pickle is not a public interchange format.
    metadata = pd.read_pickle(metadata_pickle)
    metadata = metadata[metadata.retained].copy()
    if "name" in metadata:
        parsed = metadata["name"].map(split_lab_name)
        metadata["measurement_name"] = parsed.map(lambda x: x[0])
        metadata["training_unit"] = parsed.map(lambda x: x[1])
    model = LabsEmbedder(metadata, checkpoint)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": model.state_dict()}, output / "labs_encoder.ckpt")
    metadata.to_json(output / "lab_metadata.json", orient="records", double_precision=15)
    mapping = pd.read_csv(loinc_map, dtype={"loinc_code": str})
    mapping = mapping[mapping.lab_id.isin(metadata.lab_id)].copy()
    columns = [c for c in ["lab_id", "name", "loinc_code", "confidence", "match_type", "selected_code", "selected_code_system", "status", "mapping_basis",
                          "legacy_loinc", "legacy_confidence", "legacy_match_type", "later_source_loinc", "later_source_agreement"] if c in mapping]
    mapping[columns].to_csv(output / "lab_loinc_map.csv", index=False)
    numeric_units = metadata.loc[metadata.is_numeric, "training_unit"] if "training_unit" in metadata else pd.Series(dtype=str)
    manifest = {"schema_version": 1, "embedding_dim": 768,
                "architecture": "feature-tokenizer-transformer-3x768-4heads",
                "n_model_labs": len(metadata),
                "n_local_codes": int(mapping.get("selected_code_system", pd.Series(dtype=str)).eq("STANFORD_LAB").sum()),
                "n_unresolved_labs": int(mapping.get("selected_code_system", pd.Series(dtype=str)).eq("unresolved").sum()),
                "n_loinc_codes": int(mapping.loinc_code.nunique()),
                "n_ambiguous_loinc_codes": int((mapping.groupby("loinc_code").lab_id.nunique() > 1).sum()),
                "n_numeric_labs_with_documented_units": int(numeric_units.notna().sum()),
                "preprocessing_profile": "published-v1", "pad_to_tokens": 128,
                "value_parser": "research-v1; preserve raw value strings",
                "preprocessing": "global-quarterly-last; quarter-start timestamps; 0-to-0.5-years-prior; newest-first; first-128-tokens",
                "attention_mask": "boolean-padding; matches published fused-inference semantics",
                "source_sha256": {"checkpoint": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                                  if Path(checkpoint).is_file() else None,
                                  "metadata": hashlib.sha256(Path(metadata_pickle).read_bytes()).hexdigest(),
                                  "mapping": hashlib.sha256(Path(loinc_map).read_bytes()).hexdigest()},
                "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ("labs_encoder.ckpt", "lab_metadata.json", "lab_loinc_map.csv")}}
    for filename in ("lab_loinc_source_audit.csv", "lab_loinc_mapping_provenance.json"):
        source = Path(loinc_map).parent / filename
        if source.exists():
            (output / filename).write_bytes(source.read_bytes())
            manifest["files"][filename] = hashlib.sha256((output / filename).read_bytes()).hexdigest()
    provenance = output / "lab_loinc_mapping_provenance.json"
    if provenance.exists():
        manifest["mapping_version"] = json.loads(provenance.read_text())["version"]
    if "name" in metadata:
        catalog_columns = [c for c in ["lab_id", "name", "measurement_name", "training_unit", "is_numeric", "mean", "std"] if c in metadata]
        metadata[catalog_columns].to_csv(output / "lab_catalog.csv", index=False)
    (output / "README.md").write_text("""---
tags: [glyph, clinical-research, feature-extraction]
---

# Glyph labs encoder

This is the 768-dimensional laboratory foundation encoder used by the published
Glyph labs-loinc SAE. Model weights are downloaded separately from the pip package.
The three-layer, four-head transformer uses a feature tokenizer and mean pooling.

The bundle contains the full tensor-only checkpoint, supported model-lab training
statistics and units, LOINC mappings, lab_catalog.csv, and a checksummed manifest.
Use glyph.extraction.labs.LabsEncoder with bundle_dir, or repo_id plus a pinned
revision. See https://github.com/RobbieHolland/Glyph/blob/main/docs/extraction.md.

Published preprocessing is versioned in bundle.json. It aggregates the last
measurement per lab and calendar quarter, uses the quarter-start timestamp for
the 0-to-0.5-year lookback, orders most recent first, and retains the first 128
tokens. The historical attention-mask behavior is preserved. A quarter's selected
measurement can occur after its quarter timestamp; this is a reproduction profile
of the published inputs, not a guarantee of strictly prescan measurement times.
No implicit unit conversion or ambiguous LOINC-to-lab mapping is permitted.

The model is a research feature extractor, not a clinically validated diagnostic.
Training/checkpoint provenance and the intended weight license must be finalized
by the model owner before public release. No patient-level records are included.
""")
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description="Prepare labs assets for a versioned Hugging Face release; no upload")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata-pickle", required=True)
    parser.add_argument("--loinc-map", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export_labs_bundle(args.checkpoint, args.metadata_pickle, args.loinc_map, args.output_dir)


if __name__ == "__main__":
    main()
