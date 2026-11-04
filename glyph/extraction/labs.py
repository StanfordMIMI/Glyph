"""Published lab architecture and explicit per-study input preparation.

Weights, training statistics, and LOINC mappings belong in a versioned HF bundle,
not the wheel. Source: HypothesisExtractor/models/labs_model.py.
"""
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from glyph.extraction.encoders import BatchEncoder
from glyph.extraction.lab_mapping import resolve_lab_ids

class FeatureTokenizer(nn.Module):
    def __init__(self, labs_metadata, embedding_dim):
        super().__init__()
        self.labs_metadata = labs_metadata.set_index("lab_id")
        self.embedding_dim = embedding_dim

        self.pad_embedding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, 1, embedding_dim)))
        self.mask_embedding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, embedding_dim)))

        self.time_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, embedding_dim),
        )

        self.pos_embeddings = nn.ParameterDict({
            str(row["lab_id"]): nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, embedding_dim)))
            for _, row in labs_metadata.iterrows() if row["retained"]
        })

        self.num_embeddings = nn.ParameterDict({
            str(row["lab_id"]): nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, embedding_dim)))
            for _, row in labs_metadata.iterrows() if row["is_numeric"] and row["retained"]
        })

        self.cat_embeddings = nn.ModuleDict()
        for _, row in labs_metadata.iterrows():
            if not row["is_numeric"] and row["retained"]:
                lab_id = str(row["lab_id"])
                self.cat_embeddings[lab_id] = nn.ParameterDict()
                for category in row["categories"]:
                    self.cat_embeddings[lab_id][str(category).replace(".", "_")] = nn.Parameter(
                        nn.init.kaiming_normal_(torch.empty(1, embedding_dim))
                    )

    def forward(self, lab_ids, values, years_prior, truncate=128, pad_to=None):
        max_length = min(max(len(v) for v in values), truncate) if pad_to is None else pad_to
        if not 1 <= max_length <= truncate:
            raise ValueError("Lab padding length must be between 1 and 128")
        embeddings = self.pad_embedding.repeat(len(values), max_length, 1)
        attention_mask = torch.ones(len(values), max_length, dtype=torch.float32, device=embeddings.device)

        for i, (batch_lab_ids, batch_values, batch_years_prior) in enumerate(zip(lab_ids, values, years_prior)):
            num_mask = [self.labs_metadata.loc[lid, "is_numeric"] for lid in batch_lab_ids]

            for j, (is_numeric, lab_id, value, yp) in enumerate(
                zip(num_mask, batch_lab_ids, batch_values, batch_years_prior)
            ):
                if j >= max_length:
                    break

                if is_numeric and isinstance(value, (int, float)):
                    std_val = self.labs_metadata.loc[lab_id, "std"]
                    adj = (value - self.labs_metadata.loc[lab_id, "mean"]) / std_val if std_val > 1e-6 else value
                    adj = torch.clamp(torch.tensor(adj, dtype=torch.float32, device=embeddings.device), -3, 3)
                    embeddings[i, j] = adj * self.num_embeddings[str(lab_id)]
                    attention_mask[i, j] = 0
                elif not is_numeric:
                    # Match FTTransformer's categorical numeric aliases as well as strings.
                    possible = [value]
                    if isinstance(value, (int, float)) and float(value).is_integer():
                        integer = int(value)
                        possible.extend([str(integer), f"{integer}.0", f"{integer}.00"])
                    for candidate in possible:
                        key = str(candidate).replace(".", "_")
                        if key in self.cat_embeddings[str(lab_id)]:
                            embeddings[i, j] = self.cat_embeddings[str(lab_id)][key]
                            attention_mask[i, j] = 0

                time_emb = self.time_embedding(
                    torch.tensor([float(yp)], dtype=torch.float32).to(embeddings.device)
                )
                embeddings[i, j] = embeddings[i, j] + time_emb + self.pos_embeddings[str(lab_id)]

        if attention_mask.bool().all(dim=1).any():
            raise ValueError("A study has no usable lab tokens under the published value parser")
        return embeddings, attention_mask


class LabsEmbedder(nn.Module):
    def __init__(self, labs_metadata, checkpoint_path=None, pad_to=None):
        super().__init__()
        self.pad_to = pad_to
        # Retain the trained checkpoint's CLS parameter; mean pooling does not use it.
        self.cls = nn.Parameter(torch.zeros(1, 1, 768))
        self.feature_tokenizer = FeatureTokenizer(labs_metadata, embedding_dim=768)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=4, dim_feedforward=3072, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path)
        if path.is_dir():
            paths = sorted(path.glob("*.ckpt"))
            if len(paths) != 1:
                raise ValueError("Labs checkpoint directory must contain exactly one .ckpt")
            path = paths[0]
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        prefix = "encoder.inference_map.labs_with_mask->ft_transformer."
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.load_state_dict(state, strict=True)

    def forward(self, batch):
        # batch: list of lists of dicts, each dict has lab_id, Value, years_prior
        lab_ids = [[v["lab_id"] for v in sample] for sample in batch]
        values = [
            [
                float(v["Value"])
                if isinstance(v["Value"], (int, float))
                or (isinstance(v["Value"], str) and v["Value"].replace(".", "", 1).isdigit())
                else v["Value"]
                for v in sample
            ]
            for sample in batch
        ]
        years_prior = [[v["years_prior"] for v in sample] for sample in batch]

        embeddings, mask = self.feature_tokenizer(lab_ids, values, years_prior, pad_to=self.pad_to)
        mask = mask.to(device=embeddings.device)
        # Freeze the padding semantics used by published fused inference.
        # Float 0/1 masks otherwise become additive biases when fastpath is disabled.
        output = self.encoder(embeddings, src_key_padding_mask=mask.bool())

        # Mean pool over non-padded tokens
        mask_exp = (1 - mask.float().unsqueeze(-1)).expand_as(output)
        return (output * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-6)



# The revision is filled from the approved HF release commit, never from mutable main.
LABS_ENCODER_REPO = "RobbieHolland/Glyph-labs-encoder"
LABS_ENCODER_REVISION = "972acbaea6d7ea2249a84d562a5f6dbabb994827"
LABS_BUNDLE_ASSETS = {"labs_encoder.ckpt", "lab_metadata.json", "lab_loinc_map.csv",
                     "lab_loinc_source_audit.csv", "lab_loinc_mapping_provenance.json"}


class LabsEncoder(BatchEncoder):
    embedding_dim = 768

    def __init__(self, device="cpu", bundle_dir=None, repo_id=None, revision=None, cache_dir=None, token=None):
        if bundle_dir is None:
            repo_id = repo_id or LABS_ENCODER_REPO
            revision = revision or LABS_ENCODER_REVISION
            if not repo_id or not revision:
                raise ValueError("Labs extraction requires bundle_dir, or repo_id plus pinned revision. "
                                 "The labs encoder bundle has not yet been publicly released.")
            from huggingface_hub import snapshot_download
            bundle_dir = snapshot_download(repo_id, revision=revision, cache_dir=cache_dir,
                                          allow_patterns=sorted(LABS_BUNDLE_ASSETS | {"bundle.json"}), token=token)
        folder = Path(bundle_dir)
        self.bundle_dir = folder
        for name in ("labs_encoder.ckpt", "lab_metadata.json", "lab_loinc_map.csv", "bundle.json"):
            if not (folder / name).is_file():
                raise FileNotFoundError(f"Labs bundle missing {name}")
        manifest = json.loads((folder / "bundle.json").read_text())
        if manifest.get("schema_version") != 1 or manifest.get("embedding_dim") != 768:
            raise ValueError("Unsupported labs bundle schema or embedding dimension")
        for name, expected in manifest.get("files", {}).items():
            if name not in LABS_BUNDLE_ASSETS:
                raise ValueError("Unexpected labs bundle file")
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Labs bundle checksum mismatch: {name}")
        self.metadata = pd.read_json(folder / "lab_metadata.json", orient="records")
        self.device = device
        self.profile = manifest.get("preprocessing_profile", "published-v1")
        self.pad_to = manifest.get("pad_to_tokens", 128)
        self.model = LabsEmbedder(self.metadata, folder / "labs_encoder.ckpt", pad_to=self.pad_to).eval().to(device)

    def embed_batch(self, data_list):
        samples = [json.loads(Path(p).read_text()) if isinstance(p, (str, Path)) else p for p in data_list]
        if any(not sample for sample in samples):
            raise ValueError("A study with no eligible labs cannot be encoded as a normal lab observation")
        retained = set(self.metadata.loc[self.metadata.retained, "lab_id"])
        if any(v["lab_id"] not in retained for sample in samples for v in sample):
            raise ValueError("Lab input contains unrecognized or nonretained lab_id")
        with torch.inference_mode():
            output = self.model(samples).detach().cpu()
        if not torch.isfinite(output).all():
            raise ValueError("Lab input has no usable tokens or produced nonfinite embeddings")
        return output


def _unit(value):
    return "" if pd.isna(value) else "".join(str(value).lower().split()).replace("µ", "u").replace("μ", "u")


def prepare_labs(metadata, labs, bundle_dir, *, profile=None, require_units=True, return_report=False):
    """Prepare published lab tokens from measurements without guessing units/mappings.

    published-v1 reproduces the research cache: quarterly-last aggregation,
    followed by a [0, .5]-year window on quarter timestamps, newest first.
    hypothesisextractor-v1 retains the older helper's calendar window.
    """
    folder = Path(bundle_dir)
    manifest = json.loads((folder / "bundle.json").read_text()) if (folder / "bundle.json").exists() else {}
    profile = profile or manifest.get("preprocessing_profile", "published-v1")
    if profile not in {"published-v1", "hypothesisextractor-v1"}:
        raise ValueError("Unknown lab preprocessing profile")
    stats = pd.read_json(folder / "lab_metadata.json", orient="records")
    stats = stats[stats.retained].copy().set_index("lab_id", drop=False)
    metadata = metadata.copy()
    if metadata.sample_id.isna().any() or metadata.sample_id.astype(str).duplicated().any():
        raise ValueError("Lab preparation needs unique, nonmissing study IDs")
    if metadata.patient_id.isna().any() or pd.to_datetime(metadata.date, errors="coerce").isna().any():
        raise ValueError("Lab preparation needs nonmissing patient IDs and valid study dates")
    metadata["patient_id"] = metadata.patient_id.astype(str)
    aliases = {"start": "date", "Taken Date": "date", "code": "loinc_code", "Value": "value", "Units": "unit", "units": "unit"}
    labs = labs.rename(columns={key: value for key, value in aliases.items() if value not in labs}).copy()
    labs["patient_id"] = labs.patient_id.astype(str)
    labs = labs[labs.patient_id.isin(metadata.patient_id)].copy()
    n_input = len(labs)
    labs["date"] = pd.to_datetime(labs.date, errors="coerce")
    if "lab_id" not in labs or labs.lab_id.isna().any():
        if "loinc_code" not in labs:
            raise ValueError("Measurements need loinc_code or explicit model lab_id")
        mapping = pd.read_csv(folder / "lab_loinc_map.csv", dtype={"loinc_code": str})
        labs["lab_id"] = resolve_lab_ids(labs, mapping, stats.reset_index(drop=True))
    labs = labs[labs.lab_id.isin(stats.index)].dropna(subset=["value", "date"]).copy()
    labs["lab_id"] = labs.lab_id.astype(int)
    if "training_unit" in stats:
        for lab_id, group in labs.groupby("lab_id"):
            expected = _unit(stats.loc[lab_id, "training_unit"])
            actual = group["unit"].map(_unit) if "unit" in group else pd.Series("", index=group.index)
            if expected not in {"", "nounits"} and require_units and (actual == "").any():
                raise ValueError(f"lab_id {lab_id} needs measurement units; set require_units=False only for values already in training units")
            if (actual.ne("") & actual.ne(expected)).any():
                raise ValueError(f"lab_id {lab_id} has units differing from training; convert explicitly before extraction")
    numeric_ids = set(stats.loc[stats.is_numeric, "lab_id"])
    labs["value"] = labs.value.astype(object)
    numeric = labs.lab_id.isin(numeric_ids)
    numeric_values = pd.to_numeric(labs.value, errors="coerce")
    if profile == "hypothesisextractor-v1":
        labs.loc[numeric, "value"] = numeric_values[numeric]
    legacy_unparsed = numeric & labs.value.map(lambda v: isinstance(v, str) and not v.replace('.', '', 1).isdigit())
    n_legacy_unparsed = int((legacy_unparsed & np.isfinite(numeric_values)).sum())
    valid = (~numeric | np.isfinite(numeric_values)) & labs.value.notna()
    labs = labs[valid].copy()
    if profile == "published-v1":
        # Aggregate before the study window exactly as in Data/preprocess/labs.py.
        labs = labs.sort_values(["patient_id", "lab_id", "date"], kind="stable").groupby(
            ["patient_id", "lab_id", pd.Grouper(key="date", freq="QS")]).last().reset_index()
    result = {}
    for row in metadata.itertuples():
        date = pd.Timestamp(row.date)
        subset = labs[labs.patient_id == str(row.patient_id)].copy()
        if profile == "published-v1":
            prior = (date - subset.date).dt.total_seconds() / (365.25 * 24 * 3600)
            subset = subset[(prior >= 0) & (prior <= .5)].sort_values(["date", "lab_id"], ascending=[False, True], kind="stable")
        else:
            subset = subset[(subset.date >= date - pd.DateOffset(months=6)) &
                            (subset.date <= date + pd.Timedelta(days=60))].sort_values("date")
            subset = subset.groupby(["lab_id", pd.Grouper(key="date", freq="QS")]).last().reset_index()
        result[str(row.sample_id)] = [
            {"lab_id": int(v.lab_id), "Value": v.value.item() if isinstance(v.value, np.generic) else v.value,
             "years_prior": (date - v.date).total_seconds() / (365.25 * 24 * 3600)}
            for v in subset.itertuples()]
    report = {"profile": profile, "input_rows": n_input, "retained_rows": len(labs),
              "legacy_unparsed_numeric_strings": n_legacy_unparsed,
              "studies": len(result), "studies_without_eligible_labs": sum(not x for x in result.values())}
    return (result, report) if return_report else result


def prepare_lab_inputs(metadata, labs, bundle_dir, output_dir, *, input_column="labs_path",
                       drop_empty=False, require_units=True):
    """Write Glyph-ready lab inputs to a new directory after validating all measurements."""
    metadata = pd.read_csv(metadata, dtype={"sample_id": str, "patient_id": str}) if isinstance(metadata, (str, Path)) else metadata.copy()
    labs = pd.read_csv(labs, dtype={"patient_id": str, "loinc_code": str, "code": str, "value": str, "Value": str}) if isinstance(labs, (str, Path)) else labs
    tokens, report = prepare_labs(metadata, labs, bundle_dir, require_units=require_units, return_report=True)
    empty = [key for key, values in tokens.items() if not values]
    if empty and not drop_empty:
        raise ValueError(f"{len(empty)} studies have no eligible labs; use drop_empty=True to exclude them explicitly")
    metadata = metadata[~metadata.sample_id.astype(str).isin(empty)].copy()
    if metadata.empty:
        raise ValueError("No studies have eligible lab measurements")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for key, values in tokens.items():
        if not values:
            continue
        path = output / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        path.write_text(json.dumps(values, allow_nan=False))
        paths[key] = str(path)
    metadata[input_column] = metadata.sample_id.astype(str).map(paths)
    metadata.to_csv(output / "metadata.csv", index=False)
    report["written_studies"] = len(metadata)
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2))
    return output / "metadata.csv"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare a raw lab table for Glyph using a versioned encoder bundle")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--labs", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-column", default="labs_path")
    parser.add_argument("--drop-empty", action="store_true")
    parser.add_argument("--assume-training-units", action="store_true")
    args = parser.parse_args()
    path = prepare_lab_inputs(args.metadata, args.labs, args.bundle_dir, args.output_dir,
                              input_column=args.input_column, drop_empty=args.drop_empty,
                              require_units=not args.assume_training_units)
    print(f"Prepared metadata: {path}")


if __name__ == "__main__":
    main()
