import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glyph.visualization.image_examples import image_payload
from glyph.visualization.node_sizes import node_area_parameters
from glyph.visualization.colors import outcome_palette
from glyph.visualization.correlation_edges import limit_correlation_edges


def _as_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _json_for_script(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")



def _optional_number(value):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _association(row, significant):
    return {
        "outcome": str(row.outcome),
        "odds_ratio": float(row.odds_ratio),
        "p_value": _optional_number(getattr(row, "odds_ratio_p_value", None)),
        "corrected_p_value": _optional_number(getattr(row, "odds_ratio_p_value_bonferroni", None)),
        "modality_corrected_p_value": _optional_number(getattr(row, "odds_ratio_p_value_bonferroni_modality", None)),
        "ci_lower": _optional_number(getattr(row, "odds_ratio_ci_lower", None)),
        "ci_upper": _optional_number(getattr(row, "odds_ratio_ci_upper", None)),
        "significant": bool(significant),
        "count": _optional_number(getattr(row, "c_f", None)),
    }

def _layout(
    correlation,
    seed,
    n_neighbors=20,
    min_dist=0.1,
    spread=1.0,
    repulsion_strength=1.0,
):
    n_nodes = len(correlation)
    if n_nodes == 1:
        return np.array([[0.5, 0.5]])
    if n_nodes <= 3:
        angles = np.linspace(-math.pi / 2, 3 * math.pi / 2, n_nodes, endpoint=False)
        return np.column_stack((0.5 + 0.28 * np.cos(angles), 0.5 + 0.28 * np.sin(angles)))

    import umap

    distance = np.clip(1 - np.abs(correlation), 0, 1)
    np.fill_diagonal(distance, 0)
    embedding = umap.UMAP(
        n_components=2,
        metric="precomputed",
        n_neighbors=min(n_neighbors, n_nodes - 1),
        min_dist=min_dist,
        spread=spread,
        repulsion_strength=repulsion_strength,
        random_state=seed,
    ).fit_transform(distance)
    span = np.ptp(embedding, axis=0)
    normalized = (embedding - embedding.min(axis=0)) / np.where(span > 0, span, 1)
    return 0.06 + 0.88 * normalized


def _graph_edges(correlation, percentile):
    n_nodes = len(correlation)
    if n_nodes < 2:
        return []

    absolute = np.abs(correlation)
    upper = absolute[np.triu_indices(n_nodes, k=1)]
    finite = upper[np.isfinite(upper)]
    threshold = np.quantile(finite, percentile) if len(finite) else math.inf
    del upper, finite
    rows, columns = np.nonzero(np.triu(
        np.isfinite(absolute) & (absolute > 0) & (absolute >= threshold), k=1))
    edges = {(int(i), int(j)): float(absolute[i, j]) for i, j in zip(rows, columns)}

    degree = np.zeros(n_nodes, dtype=int)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    for i in np.where(degree == 0)[0]:
        candidates = absolute[i].copy()
        candidates[i] = -1
        candidates[~np.isfinite(candidates)] = -1
        j = int(np.argmax(candidates))
        if candidates[j] > 0:
            key = tuple(sorted((int(i), j)))
            edges[key] = float(candidates[j])
            degree[i] += 1
            degree[j] += 1
    return limit_correlation_edges([[i, j, round(weight, 5)] for (i, j), weight in sorted(edges.items())],n_nodes)


def _interpretations(autointerp_results, modality, top_k):
    if autointerp_results is None or autointerp_results.empty:
        return {}

    frame = autointerp_results.copy()
    if "modality" in frame:
        frame = frame[frame["modality"] == modality]
    if "top_k" in frame:
        frame = frame[frame["top_k"] == top_k]
    if frame.empty:
        return {}

    if "test_accuracy" in frame:
        frame = frame.sort_values("test_accuracy", ascending=False)
    frame = frame.drop_duplicates("feature_idx")
    return {
        int(row.feature_idx): {
            "text": "" if pd.isna(getattr(row, "interpretation", "")) else str(getattr(row, "interpretation", "")),
            "accuracy": None if pd.isna(getattr(row, "test_accuracy", np.nan))
            else round(float(row.test_accuracy) * 100, 1),
        }
        for row in frame.itertuples()
    }


def build_interactive_concept_atlas(
    concepts,
    associations,
    output_path,
    edges=None,
    title="Glyph concept atlas",
    initial_outcome=None,
    bonferroni_alpha=0.05,
    selection_note=None,
    technical_details=None,
    outcome_groups=None,
    outcome_colors=None,
    agentic_analysis=None,
    analysis_report_url=None,
    context_min_activations=20,
    view="simple",
    wheel_label_limit=20,
    image_examples=None,
):
    """Write a self-contained atlas from an already-computed shared layout.

    ``concepts`` must contain ``concept_id``, ``x``, ``y``, and ``modality``.
    Optional columns include ``embedding_family``, ``feature_idx``, ``name``,
    ``active``, ``rate``, ``interpretation``, and ``accuracy``. ``associations``
    uses ``concept_id`` to join outcome statistics to concepts. This entry point
    is intended for adapters that pool independently fitted SAE dictionaries.
    """
    if view not in ("simple", "full"):
        raise ValueError("view must be 'simple' or 'full'")
    concepts = pd.DataFrame(concepts).copy()
    associations = pd.DataFrame(associations).copy()
    required = {"concept_id", "x", "y", "modality"}
    missing = required.difference(concepts.columns)
    if missing:
        raise ValueError(f"Concept table is missing columns: {sorted(missing)}")
    if concepts.empty:
        raise ValueError("At least one concept is required")
    if concepts["concept_id"].isna().any():
        raise ValueError("concept_id values must be nonmissing")
    concepts["concept_id"] = concepts["concept_id"].astype(str)
    if not np.isfinite(concepts[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError("Concept coordinates must be finite")
    if concepts["concept_id"].duplicated().any():
        raise ValueError("concept_id values must be unique")

    assoc_required = {"concept_id", "outcome", "odds_ratio"}
    missing = assoc_required.difference(associations.columns)
    if missing:
        raise ValueError(f"Association table is missing columns: {sorted(missing)}")

    if associations.assign(concept_id=associations.concept_id.astype(str)).duplicated(["concept_id", "outcome"]).any():
        raise ValueError("Duplicate concept/outcome associations")
    concepts = concepts.reset_index(drop=True)
    concept_lookup = {str(value): i for i, value in enumerate(concepts["concept_id"])}
    nodes = []
    for i, row in concepts.iterrows():
        raw_contexts = row.get("contexts", {})
        if not isinstance(raw_contexts, dict):
            raw_contexts = {}
        contexts = {
            str(name): {
                "active": int(values.get("active", 0)),
                "rate": round(float(values.get("rate", 0)), 3),
                "n": int(values.get("n", 0)),
            }
            for name, values in raw_contexts.items()
        }
        nodes.append({
            "i": int(i),
            "concept_id": str(row["concept_id"]),
            "feature_idx": int(row.get("feature_idx", i)),
            "name": str(row.get("name", row["concept_id"])),
            "x": round(float(row["x"]), 6),
            "y": round(float(row["y"]), 6),
            "modality": str(row["modality"]),
            "embedding_family": str(row.get("embedding_family", "default")),
            "active": int(row.get("active", 0) or 0),
            "rate": round(float(row.get("rate", 0) or 0), 3),
            "interpretation": str(row.get("interpretation", "") or ""),
            "accuracy": None if pd.isna(row.get("accuracy", np.nan)) else round(float(row["accuracy"]), 1),
            "contexts": contexts,
        })

    packed_associations = {str(i): [] for i in range(len(nodes))}
    for row in associations.itertuples(index=False):
        local_idx = concept_lookup.get(str(row.concept_id))
        if local_idx is None:
            continue
        odds_ratio = float(row.odds_ratio)
        if not np.isfinite(odds_ratio) or odds_ratio <= 0:
            continue
        corrected_p = _optional_number(getattr(row, "odds_ratio_p_value_bonferroni", None))
        significant = getattr(row, "bonferroni_significant", None)
        if significant is None or pd.isna(significant):
            significant = corrected_p is not None and corrected_p < bonferroni_alpha
        packed_associations[str(local_idx)].append(_association(row, significant))

    packed_edges = []
    for edge in edges or []:
        if isinstance(edge, dict):
            a, b, weight = edge["source"], edge["target"], edge.get("weight", 1.0)
        else:
            a, b, weight = edge
        a = concept_lookup.get(str(a), a if isinstance(a, int) else None)
        b = concept_lookup.get(str(b), b if isinstance(b, int) else None)
        if a is None or b is None or not (0 <= int(a) < len(nodes) and 0 <= int(b) < len(nodes)):
            raise ValueError("Edge references an unknown concept")
        if not np.isfinite(weight):
            raise ValueError("Edge weights must be finite")
        if a != b:
            packed_edges.append([int(a), int(b), round(float(weight), 5)])

    packed_edges = limit_correlation_edges(packed_edges,len(nodes))

    available_outcomes = {str(value) for value in associations["outcome"].dropna().unique()}
    packed_groups = {}
    ordered_outcomes = []
    for group, values in (outcome_groups or {}).items():
        group_values = [str(value) for value in values if str(value) in available_outcomes]
        if group_values:
            packed_groups[str(group)] = group_values
            ordered_outcomes.extend(value for value in group_values if value not in ordered_outcomes)
    ordered_outcomes.extend(sorted(available_outcomes.difference(ordered_outcomes)))
    outcomes = ordered_outcomes
    initial_outcome = str(initial_outcome) if initial_outcome is not None else (outcomes[0] if outcomes else None)
    if initial_outcome is not None and initial_outcome not in outcomes:
        raise ValueError(f"Unknown outcome {initial_outcome!r}; available outcomes: {outcomes}")

    payload = {
        "title": str(title),
        "view": view,
        "wheel_label_limit": max(0, int(wheel_label_limit)),
        "modality": "multiple modalities",
        "top_k": "independent SAEs",
        "split": None,
        "initial_outcome": initial_outcome,
        "outcomes": outcomes,
        "modalities": sorted({node["modality"] for node in nodes}),
        "embedding_families": sorted({node["embedding_family"] for node in nodes}),
        "bonferroni_threshold": bonferroni_alpha,
        "bonferroni_alpha": bonferroni_alpha,
        "selection_note": selection_note,
        "nodes": nodes,
        "edges": packed_edges,
        "associations": packed_associations,
        "technical_details": technical_details,
        "agentic_analysis": None if agentic_analysis is None else str(agentic_analysis),
        "analysis_report_url": None if analysis_report_url is None else str(analysis_report_url),
        "contexts": list(dict.fromkeys(
            name for node in nodes for name in node["contexts"]
        )),
        "context_min_activations": int(context_min_activations),
        "node_sizing": node_area_parameters(len(nodes), full_prevalence_area=math.pi*14**2, minimum_area=math.pi*.35**2),
        "outcome_groups": packed_groups,
        "outcome_colors": {**outcome_palette(outcomes),**{str(key): str(value) for key, value in (outcome_colors or {}).items()}},
    }
    payload.update(image_payload(image_examples))
    html = _HTML.replace("__GLYPH_DATA__", _json_for_script(payload))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  ✓ Interactive concept atlas saved to: {output_path}")
    return output_path


def build_interactive_concept_network(
    sae_features,
    alive_mask,
    phewas_results,
    output_path,
    modality,
    top_k,
    split="train",
    initial_outcome=None,
    correlation_percentile=0.975,
    bonferroni_alpha=0.05,
    seed=42,
    autointerp_results=None,
    umap_n_neighbors=20,
    umap_min_dist=0.1,
    umap_spread=1.0,
    umap_repulsion_strength=1.0,
    view="simple",
    wheel_label_limit=20,
    concept_prefix=None,
    image_examples=None,
):
    """Write a self-contained interactive HTML map of a Glyph concept space."""
    if view not in ("simple", "full"):
        raise ValueError("view must be 'simple' or 'full'")
    if split not in sae_features:
        raise ValueError(f"Unknown split {split!r}; available splits: {sorted(sae_features)}")
    if phewas_results is None:
        features = _as_numpy(sae_features[split])
        alive = _as_numpy(alive_mask).astype(bool)
        if features.ndim != 2 or len(alive) != features.shape[1]:
            raise ValueError("SAE features and alive mask have incompatible shapes")
        indices = np.where(alive)[0]
        if not len(indices):
            raise ValueError("No alive concepts are available for exploration")
        values = features[:, indices]
        correlation = np.eye(len(indices))
        variable = np.where(values.std(axis=0) > 1e-12)[0]
        enough_samples = len(values) >= 3
        if enough_samples and len(variable) > 1:
            correlation[np.ix_(variable, variable)] = np.nan_to_num(np.corrcoef(values[:, variable].T))
        if enough_samples:
            coordinates = _layout(correlation, seed, n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist, spread=umap_spread, repulsion_strength=umap_repulsion_strength)
        elif len(indices) == 1:
            coordinates = np.array([[.5, .5]])
        else:
            angles = np.linspace(0, 2 * np.pi, len(indices), endpoint=False)
            coordinates = np.column_stack((.5 + .35 * np.cos(angles), .5 + .35 * np.sin(angles)))
        interps = _interpretations(autointerp_results, modality, top_k)
        concepts = pd.DataFrame([dict(concept_id=str(i), feature_idx=int(i), name=f"{concept_prefix}{i}" if concept_prefix else f"feature_{i}",
            modality=modality, x=coordinates[j, 0], y=coordinates[j, 1],
            active=int((values[:, j] > 1e-8).sum()), rate=float((values[:, j] > 1e-8).mean() * 100),
            interpretation=interps.get(int(i), {}).get("text", ""),
            accuracy=interps.get(int(i), {}).get("accuracy")) for j, i in enumerate(indices)])
        edges = [(str(indices[a]), str(indices[b]), weight) for a, b, weight in (_graph_edges(correlation, correlation_percentile) if enough_samples else [])]
        return build_interactive_concept_atlas(concepts=concepts,
            associations=pd.DataFrame(columns=["concept_id", "outcome", "odds_ratio"]),
            edges=edges, output_path=output_path, title=f"{modality} concept exploration",
            selection_note=f"{split} · no association testing performed" + (" · fewer than 3 studies; correlations not estimated" if not enough_samples else ""), view=view,
            wheel_label_limit=wheel_label_limit, image_examples=image_examples)
    if phewas_results.empty:
        raise ValueError("No saved association rows; pass None for exploration without association results")

    required = {"feature_idx", "outcome", "split", "modality", "top_k", "odds_ratio"}
    missing = required.difference(phewas_results.columns)
    if missing:
        raise ValueError(f"PheWAS results are missing columns: {sorted(missing)}")

    if 'odds_ratio_p_value_bonferroni' not in phewas_results:
        from glyph.phewas.phewas import PheWAS
        phewas_results = PheWAS._add_multiple_testing_corrections(phewas_results)
    hypotheses = phewas_results[
        (phewas_results["modality"] == modality)
        & (phewas_results["top_k"] == top_k)
        & (phewas_results["split"] == split)
    ].copy()
    if hypotheses.empty:
        raise ValueError(f"No PheWAS rows for modality={modality!r}, top_k={top_k}, split={split!r}")

    if hypotheses.duplicated(['outcome', 'feature_idx']).any():
        raise ValueError("Ambiguous duplicate hypotheses; select one dictionary/run")

    features = _as_numpy(sae_features[split])
    alive = _as_numpy(alive_mask).astype(bool)
    if features.ndim != 2 or len(alive) != features.shape[1]:
        raise ValueError("SAE features and alive mask have incompatible shapes")

    # Keep untested concepts visible; their absence from PheWAS is not an absence from the dictionary.
    indices = np.where(alive)[0]
    if not len(indices):
        raise ValueError("No alive concepts have PheWAS results for this visualization")

    values = features[:, indices]
    if len(indices) == 1:
        correlation = np.ones((1, 1))
    else:
        correlation = np.corrcoef(values.T)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(correlation, 1)

    coordinates = _layout(
        correlation,
        seed,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        spread=umap_spread,
        repulsion_strength=umap_repulsion_strength,
    )
    edges = _graph_edges(correlation, correlation_percentile)
    interps = _interpretations(autointerp_results, modality, top_k)
    outcomes = sorted(str(value) for value in hypotheses["outcome"].dropna().unique())
    if not outcomes:
        raise ValueError("No outcomes are available for the interactive concept map")
    initial_outcome = str(initial_outcome) if initial_outcome is not None else outcomes[0]
    if initial_outcome not in outcomes:
        raise ValueError(f"Unknown outcome {initial_outcome!r}; available outcomes: {outcomes}")

    n_tests = max(1, len(indices))
    p_threshold = bonferroni_alpha / n_tests
    nodes = []
    for local_idx, feature_idx in enumerate(indices):
        active = int((values[:, local_idx] > 1e-8).sum())
        interp = interps.get(int(feature_idx), {})
        nodes.append({
            "i": local_idx,
            "feature_idx": int(feature_idx),
            "name": f"feature_{feature_idx}",
            "modality": str(modality),
            "embedding_family": "default",
            "x": round(float(coordinates[local_idx, 0]), 6),
            "y": round(float(coordinates[local_idx, 1]), 6),
            "active": active,
            "rate": round(active / len(values) * 100, 2),
            "interpretation": interp.get("text", ""),
            "accuracy": interp.get("accuracy"),
        })

    index_lookup = {feature_idx: local_idx for local_idx, feature_idx in enumerate(indices)}
    associations = {str(local_idx): [] for local_idx in range(len(indices))}
    for row in hypotheses.itertuples():
        feature_idx = int(row.feature_idx)
        if feature_idx not in index_lookup:
            continue
        odds_ratio = float(row.odds_ratio)
        if not np.isfinite(odds_ratio) or odds_ratio <= 0:
            continue
        corrected_p = _optional_number(getattr(row, "odds_ratio_p_value_bonferroni", None))
        has_stored_correction = corrected_p is not None
        associations[str(index_lookup[feature_idx])].append(
            _association(row, has_stored_correction and corrected_p < bonferroni_alpha)
        )

    payload = {
        "title": f"{modality} concept map",
        "view": view,
        "wheel_label_limit": max(0, int(wheel_label_limit)),
        "modality": str(modality),
        "top_k": int(top_k),
        "split": str(split),
        "initial_outcome": initial_outcome,
        "outcomes": outcomes,
        "modalities": [str(modality)],
        "embedding_families": ["default"],
        "bonferroni_threshold": p_threshold,
        "bonferroni_alpha": bonferroni_alpha,
        "nodes": nodes,
        "edges": edges,
        "outcome_colors": outcome_palette(outcomes),
        "node_sizing": node_area_parameters(len(nodes),full_prevalence_area=math.pi*14**2,minimum_area=math.pi*.35**2),
        "associations": associations,
    }
    payload.update(image_payload(image_examples))
    html = _HTML.replace("__GLYPH_DATA__", _json_for_script(payload))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  ✓ Interactive concept map saved to: {output_path}")
    return output_path


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Glyph concept explorer</title>
<style>
:root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--paper:#fff;--ground:#f8fafc;
 --risk:#dc2626;--protective:#2563eb;--neutral:#e2e8f0}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--ground);color:var(--ink);
 font:14px/1.45 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}
#app{display:grid;grid-template-columns:minmax(0,1fr) 420px;height:100vh}
#stage{position:relative;background:var(--paper);overflow:hidden}#graph{display:block;width:100%;height:100%;cursor:grab}
#graph:active{cursor:grabbing}.edge{stroke:#94a3b8;vector-effect:non-scaling-stroke}
.node{stroke:#64748b;stroke-opacity:.38;stroke-width:.35;vector-effect:non-scaling-stroke;pointer-events:none}.node-group{pointer-events:none;transform-box:fill-box;transform-origin:center;transition:transform .22s cubic-bezier(.2,.8,.2,1),opacity .18s ease;will-change:transform,opacity}.hit{cursor:pointer}
#toolbar{position:absolute;z-index:5;display:flex;gap:7px;flex-wrap:wrap;top:12px;left:12px;right:12px}
button,input,summary{font:inherit;font-size:12.5px;background:#fff;color:#40454d;border:1px solid var(--line);
 border-radius:8px;padding:6px 9px}button{cursor:pointer}button[aria-pressed=true]{background:#0f172a;color:#fff}
input{width:190px}details{position:relative}summary{cursor:pointer;list-style:none;min-width:150px}summary::-webkit-details-marker{display:none}
.outcomes{position:absolute;top:36px;left:0;width:310px;max-height:380px;overflow:auto;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px;box-shadow:0 12px 32px #0f172a1f}
.picker-actions{display:flex;gap:6px;padding:2px 3px 7px}.picker-actions button{flex:1}
.outcomes label{display:flex;gap:7px;align-items:flex-start;padding:5px;border-radius:5px;cursor:pointer}.outcomes label:hover{background:#f4f4f1}.outcomes input{width:auto;margin-top:3px}
.legend{position:absolute;bottom:12px;left:12px;max-width:calc(100% - 24px);background:#fffffff0;border:1px solid var(--line);
 border-radius:8px;padding:8px 11px;font-size:12px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 5px 0 10px}.direction-key{display:inline-block;width:13px;height:13px;border-radius:50%;vertical-align:-2px;margin:0 4px 0 10px}.direction-key.risk{background:#475569}.direction-key.protective{background:repeating-linear-gradient(135deg,#fff 0 3px,#64748b 3px 5px);border:1px solid #94a3b8}
#side{background:#fff;border-left:1px solid var(--line);overflow:auto;padding:20px}h1{font-size:17px;margin:0 0 3px}h2{font-size:15px;margin:18px 0 5px}
.side-tabs{display:flex;gap:5px;margin:14px 0}.side-tabs button{flex:1}.side-tabs button.active{background:#0f172a;color:#fff}
.rank-wrap{overflow:auto;max-height:calc(100vh - 155px);border:1px solid var(--line);border-radius:9px}.rank-table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;font-size:11px}
.rank-table th,.rank-table td{padding:6px 8px;border-right:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;white-space:nowrap;text-align:right}.rank-table th{position:sticky;top:0;z-index:2;cursor:pointer;color:#fff}.rank-table th.base{background:#334155;text-align:left}.rank-table td:first-child,.rank-table td:nth-child(2){text-align:left;position:sticky;background:#fff;z-index:1}.rank-table td:first-child{left:0;width:205px;max-width:205px;overflow:hidden;text-overflow:ellipsis}.rank-table td:nth-child(2){left:205px}.rank-table tr:hover td{background:#f1f5f9}.rank-cell{font-variant-numeric:tabular-nums}.stars{font-weight:800;letter-spacing:-1px}.rank-note{font-size:11px;color:var(--muted);margin:0 0 8px}
.tech-controls{display:flex;align-items:center;gap:8px;margin:10px 0}.tech-controls select{font:inherit;border:1px solid var(--line);border-radius:7px;padding:5px 8px;background:#fff}.tech-table{border-collapse:collapse;width:100%;font-size:11.5px;margin:8px 0 18px}.tech-table th,.tech-table td{padding:6px 7px;border-bottom:1px solid var(--line);text-align:right}.tech-table th:first-child,.tech-table td:first-child,.tech-table th:nth-child(2),.tech-table td:nth-child(2){text-align:left}.tech-table th{background:#f8fafc;position:sticky;top:0}.metric-good{color:#047857;font-weight:700}.metric-bad{color:#b91c1c}.tech-scroll{max-height:calc(100vh - 260px);overflow:auto;border:1px solid var(--line);border-radius:9px}.tech-note{font-size:11px;color:var(--muted)}
.ranking-open #app{grid-template-columns:minmax(320px,1fr) min(72vw,1100px)}
.meta{font-size:12px;color:var(--muted)}p{color:#484d55}dl{display:grid;grid-template-columns:auto auto;gap:4px 12px;font-size:12.5px}
.analysis-box{margin:12px 0;border:1px solid #dbeafe;border-radius:10px;background:#f8fbff;overflow:hidden}.analysis-box>summary{width:100%;min-width:0;border:0;border-radius:0;padding:9px 11px;background:#eff6ff;color:#1e3a8a;font-weight:750}.analysis-copy{padding:2px 11px 8px;font-size:12px}.analysis-copy p{margin:8px 0;color:#334155}
dt{color:var(--muted)}dd{margin:0;text-align:right}.wheel{width:100%;height:auto;overflow:visible}.assoc{width:100%;border-collapse:collapse;font-size:12px}
.assoc th,.assoc td{padding:5px 4px;border-bottom:1px solid #eee;text-align:right}.assoc th:first-child,.assoc td:first-child{text-align:left}
#tip{position:fixed;z-index:20;pointer-events:none;opacity:0;background:#0f172a;color:#fff;max-width:330px;
 border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 7px 24px #0004}
.tour-shade{position:fixed;inset:0;background:#0f172a99;backdrop-filter:blur(1px);z-index:30}.tour-card{position:fixed;z-index:32;width:min(360px,calc(100vw - 32px));background:#fff;border:1px solid #dbe3ee;border-radius:14px;padding:18px;box-shadow:0 20px 55px #0f172a40;transition:left .2s ease,top .2s ease}.tour-card h2{margin:2px 0 7px;font-size:17px}.tour-card p{margin:0 0 15px}.tour-kicker{color:#2563eb;font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.tour-actions{display:flex;align-items:center;gap:7px}.tour-actions .tour-skip{margin-right:auto;border:0;color:var(--muted)}.tour-target{z-index:31!important;box-shadow:0 0 0 4px #fff,0 0 0 8px #60a5fa!important;border-radius:10px}.tour-dots{color:#94a3b8;font-size:11px}
.simple-view #app{grid-template-columns:minmax(0,1fr) 420px}
.simple-view #side{padding:28px 24px}.simple-view h1{font-size:20px;letter-spacing:-.03em}
.simple-view #toolbar{gap:8px;top:20px;left:20px;right:20px}
.simple-view #toolbar>input{flex:1;max-width:250px;min-width:130px}
.simple-view #toolbar>button,.simple-view #toolbar>details>summary{padding:9px 12px;border-radius:10px;box-shadow:0 2px 6px #0f172a06}
.simple-view #toolbar>input{padding:9px 12px;border-radius:10px;box-shadow:0 2px 6px #0f172a06}
.simple-view .side-tabs,.simple-view #agenticAnalysis{display:none!important}
.simple-view #intro{margin-top:30px;color:#64748b;line-height:1.8}
.simple-view #detail h2{margin-top:24px}.simple-view .assoc{font-size:11.5px}
.simple-view .legend{left:20px;bottom:20px;padding:10px 14px;border-radius:10px;line-height:1.8}
.simple-view .outcomes{top:43px;padding:12px;border-radius:12px;box-shadow:0 12px 36px #0f172a12}
.simple-view #displaySettings .outcomes .outcomes{position:static;width:auto;max-height:none;border:0;padding:8px 0;box-shadow:none}
.simple-view #displaySettings .outcomes b{display:block;margin-top:8px;color:#334155}
.simple-view #displaySettings .outcomes p{display:none}
.hit:focus-visible{outline:none;stroke:#0f172a;stroke-width:2}
@media(max-width:850px){#app,.simple-view #app{grid-template-columns:1fr;grid-template-rows:minmax(300px,55vh) minmax(0,1fr);height:100dvh}#side{display:block;border-left:0;border-top:1px solid var(--line)}.simple-view #side{padding:20px}.simple-view #toolbar{top:12px;left:12px;right:12px;gap:6px}.simple-view #toolbar>input{max-width:180px}.simple-view .legend{left:12px;bottom:12px;font-size:11px}.ranking-open #app{grid-template-columns:1fr}}

.node-group.is-selected .node{stroke:#0f172a;stroke-opacity:1;stroke-width:2}
#outcomeSearch{width:100%;margin:0 0 8px}.outcomes input[type=search]{width:100%}
#mapStatus{position:absolute;top:76px;left:22px;font-size:12px;color:var(--muted);pointer-events:none}
#emptyState{position:absolute;left:50%;top:45%;transform:translate(-50%,-50%);max-width:310px;padding:20px;background:#fffffff0;border:1px solid var(--line);border-radius:12px;text-align:center;pointer-events:none}
#emptyState b{display:block;margin-bottom:5px}#emptyState p{margin:0;color:var(--muted);font-size:12px}
.detail-actions{display:flex;gap:8px;justify-content:space-between;margin-top:22px}.detail-actions button{font-size:11.5px}
.wheel-spoke{cursor:help;outline:none}.wheel-spoke:hover,.wheel-spoke:focus-visible{fill:#0f172a;fill-opacity:.09;stroke:#475569;stroke-width:.8}
.wheel-label{pointer-events:none}.assoc td{vertical-align:top}.assoc .ci{display:block;font-size:10px;color:var(--muted);white-space:nowrap}
.assoc .outcome-link{padding:0;border:0;background:none;text-align:left;font-size:11.5px;color:inherit}.assoc .outcome-link:hover{text-decoration:underline}
.p-value{font-variant-numeric:tabular-nums;white-space:nowrap}.assoc-note{font-size:11px;color:var(--muted);margin:8px 0}
#tip dl{margin:7px 0 0;gap:2px 14px;font-size:11.5px}#tip dt{color:#cbd5e1}#tip dd{color:#fff}
#filterReset{margin-top:9px;width:100%}#zoomControls{position:absolute;right:20px;bottom:20px;display:flex;gap:4px}#zoomControls button{font-size:18px;width:32px;height:32px;padding:0}
#search:focus-visible,button:focus-visible,summary:focus-visible{outline:2px solid #64748b;outline-offset:2px}
@media(max-width:850px){#toolbar details>.outcomes{position:fixed;left:12px;right:12px;top:110px;width:auto;max-height:calc(52dvh - 130px)}#mapStatus{top:110px;left:14px;font-size:11px}#side{min-height:0}.simple-view #side{padding:20px}.outcomes{max-width:calc(100vw - 28px)}.simple-view .legend{max-width:calc(100% - 70px);font-size:10px}.simple-view #toolbar>details>summary{min-width:105px}.simple-view #toolbar>input{max-width:none;flex-basis:135px}.simple-view #toolbar>button{padding:8px}#zoomControls{right:12px;bottom:12px;flex-direction:column}#mapStatus{max-width:calc(100% - 28px)}.simple-view #app{grid-template-rows:minmax(320px,52dvh) minmax(0,1fr)}}
.concept-images{margin:20px 0}.example-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.example-grid figure{margin:0;min-width:0}.example-grid img{display:block;width:100%;height:auto;aspect-ratio:1;object-fit:contain;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:10px}.example-grid img.pixelated{image-rendering:pixelated}.example-grid figcaption{margin-top:6px;font-size:10px;overflow-wrap:anywhere;color:#475569}.example-grid figcaption span{display:block;margin-top:3px;color:#64748b}
</style></head><body><div id="app"><main id="stage">
<div id="toolbar"><details id="outcomePicker"><summary id="outcomeSummary">Outcomes</summary><div class="outcomes"><div class="picker-actions"><button id="selectAllOutcomes" type="button">Select all</button><button id="clearOutcomes" type="button">Clear</button></div><input id="outcomeSearch" type="search" aria-label="Find an outcome" placeholder="Find an outcome"><div id="outcomeChoices"></div></div></details><details id="contextPicker"><summary id="contextSummary">Context: all</summary><div class="outcomes" id="contextChoices"></div></details><details id="facetPicker"><summary>Modalities / families</summary><div class="outcomes" id="facetChoices"></div></details><details id="colorSettings"><summary id="colorSummary">Color: outcomes</summary><div class="outcomes"><b>Node color</b><label><input type="radio" name="colorMode" value="outcome" checked><span>Outcome associations</span></label><label><input type="radio" name="colorMode" value="modality"><span>Modality</span></label><p class="meta">Outcome mode uses the selected significant associations. Modality mode shows where each data source lives.</p></div></details><details id="displaySettings"><summary id="settingsSummary">Both directions · extreme 10% · min 5</summary><div class="outcomes"><b>Association direction</b><label><input type="radio" name="associationDirection" value="risk"><span>Risk (OR &gt; 1)</span></label><label><input type="radio" name="associationDirection" value="protective"><span>Protective (OR &lt; 1)</span></label><label><input type="radio" name="associationDirection" value="both" checked><span>Both directions</span></label><label><input id="extremeOnly" type="checkbox" checked><span>Most extreme 10% per direction (minimum 5)</span></label><p class="meta">Risk uses the highest odds ratios; protective uses the lowest. Ranking is recalculated within each outcome and the currently visible context.</p></div></details><input id="search" type="search" aria-label="Search concepts" placeholder="Search concepts">
<button id="reset">Reset view</button><button id="tourButton">Tour</button></div>
<svg id="graph" role="group" aria-label="Interactive concept network"><g id="camera"><g id="edgeLayer"></g><g id="nodeLayer"></g></g></svg>
<div id="mapStatus" role="status" aria-live="polite"></div><div id="emptyState" hidden><b>No matching concepts</b><p>Try another search or change the display options.</p></div><div class="legend" id="legend"></div><div id="zoomControls"><button id="zoomIn" aria-label="Zoom in">+</button><button id="zoomOut" aria-label="Zoom out">−</button></div>
<div id="tip" role="tooltip"></div></main><aside id="side"><h1 id="mapTitle">Glyph</h1><div class="meta" id="mapMeta"></div><details class="analysis-box" id="agenticAnalysis" hidden><summary>Agentic analysis</summary><div class="analysis-copy" id="agenticAnalysisContent"></div><div class="analysis-copy"><a id="analysisReportLink" target="_blank" rel="noopener" hidden>Open evidence report ↗</a></div></details><div class="side-tabs"><button id="detailTab" class="active">Concept detail</button><button id="rankingTab">Significant associations</button><button id="technicalTab">Technical details</button></div>
<section id="detailPanel"><p id="intro">Select a concept to inspect its interpretation and association profile. Scroll to zoom and drag to pan.</p><div id="detail"></div></section><section id="rankingPanel" hidden><p class="rank-note" id="rankNote">Globally significant associations. Stars use within-modality Bonferroni p: * &lt;.05, ** &lt;.01, *** &lt;.001. Click an outcome header to sort by odds ratio.</p><div class="rank-wrap"><table class="rank-table" id="rankTable"></table></div></section><section id="technicalPanel" hidden><p class="tech-note" id="technicalNote"></p><div class="tech-controls"><label for="technicalSplit"><b>Evaluation split</b></label><select id="technicalSplit"><option value="train">Train</option><option value="val">Validation</option><option value="test" selected>Test</option></select></div><div id="technicalContent"></div></section></aside></div>
<div id="tourShade" class="tour-shade" hidden></div><div id="tourCard" class="tour-card" hidden><div class="tour-kicker" id="tourKicker"></div><h2 id="tourTitle"></h2><p id="tourText"></p><div class="tour-actions"><button class="tour-skip" id="tourSkip">Skip</button><span class="tour-dots" id="tourDots"></span><button id="tourBack">Back</button><button id="tourNext">Next</button></div></div>
<script>
const DATA=__GLYPH_DATA__,NS='http://www.w3.org/2000/svg',S=1000,BASE_NODE_OPACITY=.7;
const isSimple=DATA.view!=='full';document.body.classList.toggle('simple-view',isSimple);
document.title=DATA.title;document.getElementById('mapTitle').textContent=DATA.title;
const svg=document.getElementById('graph'),camera=document.getElementById('camera'),edgeLayer=document.getElementById('edgeLayer'),nodeLayer=document.getElementById('nodeLayer'),tip=document.getElementById('tip');
const patternDefs=document.createElementNS(NS,'defs');svg.prepend(patternDefs);const protectivePatterns=new Map();
svg.setAttribute('viewBox',`0 0 ${S} ${S}`);document.getElementById('mapMeta').textContent=isSimple?`${DATA.nodes.length} concepts · ${DATA.outcomes.length} outcomes${DATA.split?' · '+DATA.split+' split · unadjusted associations':''}${DATA.selection_note?' · '+DATA.selection_note:''}`:`${DATA.modality} · ${DATA.top_k} · ${DATA.split} · ${DATA.nodes.length} displayed concepts · Bonferroni-adjusted p < ${(DATA.bonferroni_alpha??.05).toPrecision(3)}${DATA.selection_note?' · '+DATA.selection_note:''}`;
const escapeHTML=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
if(DATA.agentic_analysis||DATA.analysis_report_url){const box=document.getElementById('agenticAnalysis'),content=document.getElementById('agenticAnalysisContent'),link=document.getElementById('analysisReportLink');box.hidden=false;if(DATA.agentic_analysis)content.innerHTML=String(DATA.agentic_analysis).split(/\n\s*\n/).filter(Boolean).map(p=>`<p>${escapeHTML(p).replace(/\n/g,'<br>')}</p>`).join('');if(DATA.analysis_report_url){link.href=DATA.analysis_report_url;link.hidden=false}}
const A=DATA.associations;
let selectedNode=null,onlySignificant=false;
const formatP=p=>p==null?'—':p===0?'0 (reported)':p<.001?p.toExponential(2):p.toPrecision(3);
const formatOR=v=>v==null?'—':v>=1000||v<.01?v.toExponential(2):v!==1&&Math.abs(v-1)<.005?v.toPrecision(6):v.toFixed(2);
function moveTip(e){const r=e.target?.getBoundingClientRect?.();let x=e.clientX??(r?.right??20),y=e.clientY??(r?.top??20);if(e.type==='focusin'||e.type==='focus'){x=r?.right??20;y=r?.top??20}tip.style.opacity=1;tip.style.left=Math.max(8,Math.min(innerWidth-tip.offsetWidth-8,x+14))+'px';tip.style.top=Math.max(8,Math.min(innerHeight-tip.offsetHeight-8,y+14))+'px'}
function associationTip(e,a){tip.innerHTML=`<b>${escapeHTML(a.outcome)}</b><div>${a.odds_ratio<1?'Lower':'Higher'} odds · ${a.significant?'Bonferroni-significant':'Not Bonferroni-significant'}</div><dl><dt>Odds ratio</dt><dd>${formatOR(a.odds_ratio)}</dd><dt>95% confidence interval</dt><dd>${a.ci_lower!=null&&a.ci_upper!=null?formatOR(a.ci_lower)+'–'+formatOR(a.ci_upper):'Not available'}</dd><dt>Raw p-value</dt><dd>${formatP(a.p_value)}</dd><dt>Bonferroni p-value</dt><dd>${formatP(a.corrected_p_value)}</dd>${a.count==null?'':`<dt>Reported support</dt><dd>${a.count.toLocaleString()}</dd>`}</dl>`;moveTip(e)}
function renderEdges(){if(edgeLayer.childElementCount)return;const fragment=document.createDocumentFragment();DATA.edges.forEach(([a,b,w])=>{const x=document.createElementNS(NS,'line');x.classList.add('edge');x.dataset.a=a;x.dataset.b=b;x.setAttribute('x1',DATA.nodes[a].x*S);x.setAttribute('y1',DATA.nodes[a].y*S);x.setAttribute('x2',DATA.nodes[b].x*S);x.setAttribute('y2',DATA.nodes[b].y*S);x.setAttribute('stroke-width',.30*Math.abs(w));x.setAttribute('stroke-opacity',.14*Math.abs(w));fragment.appendChild(x)});edgeLayer.appendChild(fragment)}renderEdges();
let activeContext=null;
const contextValues=n=>activeContext?n.contexts?.[activeContext]:null;
const nodeRate=n=>contextValues(n)?.rate??n.rate,nodeActive=n=>contextValues(n)?.active??n.active;
const contextPasses=n=>!activeContext||nodeActive(n)>=(DATA.context_min_activations??20);
const NODE_SIZING=DATA.node_sizing;
const radius=n=>Math.sqrt(Math.max(NODE_SIZING.minimum_area,NODE_SIZING.full_prevalence_area*Math.max(0,Math.min(100,nodeRate(n)))/100)/Math.PI);
DATA.nodes.forEach(n=>{const g=document.createElementNS(NS,'g');g.classList.add('node-group');g.style.opacity=BASE_NODE_OPACITY;n.el=g;nodeLayer.appendChild(g);
 const h=document.createElementNS(NS,'circle');n.hit=h;h.classList.add('hit');h.setAttribute('cx',n.x*S);h.setAttribute('cy',n.y*S);h.setAttribute('r',Math.max(9,radius(n)+3));h.setAttribute('fill','transparent');h.setAttribute('tabindex','0');h.setAttribute('role','button');h.setAttribute('aria-label',n.name);h.dataset.node=n.i;h.onclick=e=>{e.stopPropagation();if(moved<4&&nodePassesFilter(n))show(n)};h.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show(n)}};
 h.onmousemove=e=>showTip(e,n);h.onmouseleave=()=>tip.style.opacity=0;nodeLayer.appendChild(h)});
const selected=new Set(DATA.initial_outcome==null?[]:[DATA.initial_outcome]),choices=document.getElementById('outcomeChoices'),summary=document.getElementById('outcomeSummary');
const DISEASE_PALETTE=['#14B8A6','#3B82F6','#8B5CF6','#EC4899','#6366F1','#06B6D4','#10B981','#A855F7','#0EA5E9'];
const MODALITY_PALETTE=['#1E3A5F','#514B8A','#246B67','#475569','#075985','#374151'];
const outcomeColor=o=>DATA.outcome_colors?.[o]||DISEASE_PALETTE[DATA.outcomes.indexOf(o)%DISEASE_PALETTE.length];
const modalityColor=m=>MODALITY_PALETTE[(DATA.modalities||[]).indexOf(m)%MODALITY_PALETTE.length];
let colorMode='outcome';
function addOutcomeChoice(o){const label=document.createElement('label'),box=document.createElement('input'),swatch=document.createElement('i'),text=document.createElement('span');box.type='checkbox';box.value=o;box.checked=selected.has(o);swatch.className='dot';swatch.style.background=outcomeColor(o);swatch.style.margin='3px 0 0 0';text.textContent=o;label.append(box,swatch,text);choices.appendChild(label);box.onchange=()=>{box.checked?selected.add(o):selected.delete(o);updateSummary();recolor();applyFilters()}}
const groupedOutcomes=new Set();Object.entries(DATA.outcome_groups||{}).forEach(([group,outcomes])=>{const head=document.createElement('b');head.textContent=group;head.style.display='block';head.style.padding='9px 5px 3px';choices.appendChild(head);outcomes.forEach(o=>{groupedOutcomes.add(o);addOutcomeChoice(o)})});DATA.outcomes.filter(o=>!groupedOutcomes.has(o)).forEach(addOutcomeChoice);
function syncOutcomeBoxes(){choices.querySelectorAll('input').forEach(box=>box.checked=selected.has(box.value));updateSummary();recolor();applyFilters()}
document.getElementById('selectAllOutcomes').onclick=e=>{e.preventDefault();DATA.outcomes.forEach(o=>selected.add(o));syncOutcomeBoxes()};
document.getElementById('clearOutcomes').onclick=e=>{e.preventDefault();selected.clear();syncOutcomeBoxes()};
function updateSummary(){if(!DATA.outcomes.length){summary.textContent="No outcome labels";return}summary.textContent=selected.size===1?[...selected][0]:`${selected.size} outcomes selected`}
document.getElementById('outcomeSearch').oninput=e=>{const q=e.target.value.trim().toLowerCase();choices.querySelectorAll('label').forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(q))};
const contextPicker=document.getElementById('contextPicker'),contextChoices=document.getElementById('contextChoices'),contextSummary=document.getElementById('contextSummary');
function addContextChoice(label,value){const row=document.createElement('label'),radio=document.createElement('input'),text=document.createElement('span');radio.type='radio';radio.name='baselineContext';radio.checked=value===null;text.textContent=label;row.append(radio,text);contextChoices.appendChild(row);radio.onchange=()=>{if(!radio.checked)return;activeContext=value;contextSummary.textContent=value?`Context: ${value}`:'Context: all';updateAssociationMode()}}
if(DATA.contexts?.length){addContextChoice('All contexts',null);DATA.contexts.forEach(name=>addContextChoice(name,name))}else contextPicker.hidden=true;
const color=a=>!a||!a.significant?'#a8adb5':a.odds_ratio>=1?'#c73b3b':'#3478bd';
const selectedHits=n=>(A[n.i]||[]).filter(a=>selected.has(a.outcome)&&associationPasses(a,n));
function circle(cx,cy,r,fill,cls=''){const x=document.createElementNS(NS,'circle');x.setAttribute('cx',cx);x.setAttribute('cy',cy);x.setAttribute('r',r);x.setAttribute('fill',fill);if(cls)x.setAttribute('class',cls);return x}
function wedge(cx,cy,r,a0,a1,fill){const p=t=>[cx+r*Math.cos(t),cy+r*Math.sin(t)],s=p(a0),e=p(a1),x=document.createElementNS(NS,'path');x.setAttribute('d',`M${cx},${cy}L${s[0]},${s[1]}A${r},${r} 0 ${a1-a0>Math.PI?1:0} 1 ${e[0]},${e[1]}Z`);x.setAttribute('fill',fill);return x}
function protectiveFill(outcome){if(protectivePatterns.has(outcome))return`url(#${protectivePatterns.get(outcome)})`;const id=`protective-${protectivePatterns.size}`,pattern=document.createElementNS(NS,'pattern'),base=document.createElementNS(NS,'rect'),stripe=document.createElementNS(NS,'path'),hue=outcomeColor(outcome);pattern.id=id;pattern.setAttribute('patternUnits','userSpaceOnUse');pattern.setAttribute('width',7);pattern.setAttribute('height',7);base.setAttribute('width',7);base.setAttribute('height',7);base.setAttribute('fill',hue);base.setAttribute('fill-opacity','.18');stripe.setAttribute('d','M-2,7 L7,-2 M2,9 L9,2');stripe.setAttribute('stroke',hue);stripe.setAttribute('stroke-width','2');pattern.append(base,stripe);patternDefs.appendChild(pattern);protectivePatterns.set(outcome,id);return`url(#${id})`}
const associationFill=a=>a.odds_ratio<1?protectiveFill(a.outcome):outcomeColor(a.outcome);
function ringWedge(cx,cy,r0,r1,a0,a1,fill){const p=(r,t)=>[cx+r*Math.cos(t),cy+r*Math.sin(t)],a=p(r1,a0),b=p(r1,a1),c=p(r0,a1),d=p(r0,a0),x=document.createElementNS(NS,'path');x.setAttribute('d',`M${a}A${r1},${r1} 0 ${a1-a0>Math.PI?1:0} 1 ${b}L${c}A${r0},${r0} 0 ${a1-a0>Math.PI?1:0} 0 ${d}Z`);x.setAttribute('fill',fill);return x}
function drawNode(n){const hits=selectedHits(n),cx=n.x*S,cy=n.y*S,r=radius(n);n.el.replaceChildren();n.hit.setAttribute('r',Math.max(9,r+3));if(colorMode==='modality')n.el.appendChild(circle(cx,cy,r,modalityColor(n.modality)));else if(!hits.length)n.el.appendChild(circle(cx,cy,r,'#E2E8F0'));else if(hits.length===1)n.el.appendChild(circle(cx,cy,r,associationFill(hits[0])));else hits.forEach((a,i)=>n.el.appendChild(wedge(cx,cy,r,-Math.PI/2+i*2*Math.PI/hits.length,-Math.PI/2+(i+1)*2*Math.PI/hits.length,associationFill(a))));n.outline=circle(cx,cy,r,'none','node');n.el.appendChild(n.outline)}
function updateLegend(){const legend=document.getElementById('legend'),contextLabel=activeContext?` · ${escapeHTML(activeContext)} concepts (≥${DATA.context_min_activations??20} active)`:DATA.contexts?.length?' · all contexts':'',directionLabel=associationDirection==='risk'?'risk OR > 1':associationDirection==='protective'?'protective OR < 1':'risk + protective',modeLabel=extremeOnly?`extreme 10% ${directionLabel} (min 5)`:directionLabel;if(colorMode==='modality'){legend.innerHTML='<b>Color: modality'+contextLabel+'</b><br>Node area: activation prevalence<br>'+(DATA.modalities||[]).map(m=>`<span class="dot" style="background:${modalityColor(m)}"></span>${escapeHTML(m)}`).join('');return}const shown=[...selected].slice(0,selected.size<=12?12:8),extra=Math.max(0,selected.size-shown.length);legend.innerHTML=`<b>Color: outcome · ${modeLabel}${contextLabel}</b><span class="direction-key risk"></span>▲ predictive <span class="direction-key protective"></span>▼ protective<br>`+shown.map(o=>`<span class="dot" style="background:${outcomeColor(o)}"></span>${escapeHTML(o)}`).join('')+(extra?` &nbsp;+${extra} more`:'')+'<span class="dot" style="background:#E2E8F0"></span>none<br>Node area: activation prevalence'}
function recolor(){DATA.nodes.forEach(drawNode);updateLegend()}
function showTip(e,n){const hits=selectedHits(n),hitText=hits.length?hits.map(a=>`<span style="color:${outcomeColor(a.outcome)}">${a.odds_ratio<1?'▼':'▲'}</span> ${a.odds_ratio<1?'protective':'predictive'} · ${escapeHTML(a.outcome)} · OR ${a.odds_ratio.toFixed(2)}`).join('<br>'):'No significant selected outcomes',contextText=activeContext?`${escapeHTML(activeContext)} activation ${nodeRate(n)}% (${nodeActive(n).toLocaleString()})`:`activation ${n.rate}%`;tip.innerHTML=`<b>${escapeHTML(n.name)}</b><br>${escapeHTML(n.modality)} · ${escapeHTML(n.embedding_family)}<br>${escapeHTML(n.interpretation||'No interpretation')}<br><span style="opacity:.78">${contextText}<br>${hitText}</span>`;tip.style.opacity=1;let x=e.clientX+14,y=e.clientY+14;if(x+tip.offsetWidth>innerWidth-8)x=e.clientX-tip.offsetWidth-14;if(y+tip.offsetHeight>innerHeight-8)y=e.clientY-tip.offsetHeight-14;tip.style.left=x+'px';tip.style.top=y+'px'}
function wheel(n){
 const outcomeOrder=new Map(DATA.outcomes.map((outcome,index)=>[outcome,index]));
 const items=(A[n.i]||[]).slice().sort((a,b)=>(outcomeOrder.get(a.outcome)??999)-(outcomeOrder.get(b.outcome)??999));
 if(!items.length)return '<p class="meta">No outcome estimates are available for this concept.</p>';
 const D=400,c=D/2,R0=62,R1=174,SPOKE_LABEL_R=112,labels=items.length<=(DATA.wheel_label_limit??20),max=Math.max(Math.log(2),...items.map(a=>Math.abs(Math.log(a.odds_ratio)))),baseline=(R0+R1)/2,step=2*Math.PI/items.length,gap=Math.min(.045,step*.18);
 const point=(r,t)=>[c+r*Math.cos(t),c+r*Math.sin(t)];
 const sector=(r0,r1,a0,a1)=>{const a=point(r1,a0),b=point(r1,a1),d=point(r0,a1),e=point(r0,a0);return `M${a}A${r1},${r1} 0 ${a1-a0>Math.PI?1:0} 1 ${b}L${d}A${r0},${r0} 0 ${a1-a0>Math.PI?1:0} 0 ${e}Z`};
 let g=`<circle cx="${c}" cy="${c}" r="${R1}" fill="none" stroke="#f1f5f9"/>`;
 items.forEach((a,i)=>{
  const ang=i*step-Math.PI/2,a0=ang-step/2+gap/2,a1=ang+step/2-gap/2,end=baseline+(R1-R0)/2*Math.log(a.odds_ratio)/max;
  if(a.odds_ratio!==1)g+=`<path class="wheel-effect" data-or="${a.odds_ratio}" data-start-radius="${baseline}" data-end-radius="${end}" d="${sector(Math.min(baseline,end),Math.max(baseline,end),a0,a1)}" fill="${outcomeColor(a.outcome)}" fill-opacity="${a.significant?.9:.2}" pointer-events="none"/>`;
  g+=`<path class="wheel-spoke" data-association="${(A[n.i]||[]).indexOf(a)}" data-outcome="${escapeHTML(a.outcome)}" d="${sector(R0,R1,a0,a1)}" fill="transparent" tabindex="0" role="button" aria-describedby="tip" aria-label="${escapeHTML(a.outcome)}: OR ${formatOR(a.odds_ratio)}, ${a.significant?'significant':'not significant'}"><title>${escapeHTML(a.outcome)} · OR ${formatOR(a.odds_ratio)} · Bonferroni p ${formatP(a.corrected_p_value)}</title></path>`;
  if(labels){const label=point(SPOKE_LABEL_R,ang),left=Math.cos(ang)<0,rotation=ang*180/Math.PI+(left?180:0),name=a.outcome.length>27?a.outcome.slice(0,26)+'…':a.outcome;g+=`<text class="wheel-label" transform="translate(${label[0]} ${label[1]}) rotate(${rotation})" text-anchor="middle" dominant-baseline="middle" font-size="${items.length<=8?10.5:8.5}" font-weight="${a.significant?650:400}" fill="#334155" style="paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round">${a.odds_ratio<1?'▼':'▲'} ${escapeHTML(name)} · ${formatOR(a.odds_ratio)}</text>`}
 });
 g+=`<g pointer-events="none"><circle class="wheel-reference" data-or="${Math.exp(-max)}" cx="${c}" cy="${c}" r="${R0}" fill="none" stroke="#94a3b8" stroke-dasharray="3 4"/><circle class="wheel-reference" data-or="1" cx="${c}" cy="${c}" r="${baseline}" fill="none" stroke="#64748b" stroke-dasharray="4 4"/><text x="${c+7}" y="${c-R0+12}" font-size="10" fill="#64748b" style="paint-order:stroke;stroke:#fff;stroke-width:3px">${formatOR(Math.exp(-max))}</text><text x="${c+7}" y="${c-baseline-5}" font-size="10" fill="#475569" style="paint-order:stroke;stroke:#fff;stroke-width:3px">OR = 1</text><circle class="wheel-reference" data-or="${Math.exp(max)}" cx="${c}" cy="${c}" r="${R1}" fill="none" stroke="#94a3b8" stroke-dasharray="3 4"/><text x="${c+7}" y="${c-R1-5}" font-size="10" fill="#64748b" style="paint-order:stroke;stroke:#fff;stroke-width:3px">${formatOR(Math.exp(max))}</text></g><text x="${c}" y="${c-5}" text-anchor="middle" font-size="22" font-weight="650">${items.length}</text><text x="${c}" y="${c+16}" text-anchor="middle" font-size="11" fill="#64748b">outcomes</text>`;
 return `<svg class="wheel" viewBox="0 0 ${D} ${D}" role="group" aria-label="Outcome wheel for ${escapeHTML(n.name)}">${g}</svg>`;
}
function selectOutcome(outcome){selected.clear();selected.add(outcome);syncOutcomeBoxes()}
function conceptImages(n){
 const refs=(DATA.concept_images||{})[String(n.concept_id??n.feature_idx)]||(DATA.concept_images||{})[String(n.feature_idx)]||[];
 if(!refs.length)return '';
 const cards=refs.map(([key,activation])=>{const image=(DATA.image_bank||{})[key];if(!image)return '';return `<figure><img src="${escapeHTML(image.src)}" alt="High-activation example ${escapeHTML(image.sample_id)}" loading="lazy" class="${image.pixelated?'pixelated':''}" width="${image.width}" height="${image.height}"><figcaption>${escapeHTML(image.sample_id)}<span>${escapeHTML(image.split)} · activation ${Number(activation).toPrecision(3)}</span></figcaption></figure>`}).join('');
 return `<section class="concept-images"><h2>High-activation examples</h2><div class="example-grid">${cards}</div></section>`;
}
function renderSelected(){
 const detail=document.getElementById('detail'),intro=document.getElementById('intro');
 if(!selectedNode){detail.replaceChildren();intro.style.display='';return}
 const n=selectedNode,items=(A[n.i]||[]).slice().sort((a,b)=>Math.abs(Math.log(b.odds_ratio))-Math.abs(Math.log(a.odds_ratio)));
 const rows=items.map(a=>`<tr><td><button class="outcome-link" data-outcome="${escapeHTML(a.outcome)}">${escapeHTML(a.outcome)}</button></td><td>${a.odds_ratio<1?'▼':'▲'} ${formatOR(a.odds_ratio)}${a.ci_lower!=null&&a.ci_upper!=null?`<span class="ci">${formatOR(a.ci_lower)}–${formatOR(a.ci_upper)}</span>`:''}</td><td class="p-value">${formatP(a.corrected_p_value)}</td></tr>`).join('');
 const activation=activeContext?`${nodeActive(n).toLocaleString()} active in ${escapeHTML(activeContext)} · ${nodeRate(n)}%`:`${n.active.toLocaleString()} active samples · ${n.rate}%`;
 const family=n.embedding_family&&n.embedding_family!=='default'?` · ${escapeHTML(n.embedding_family)}`:'';
 intro.style.display='none';detail.innerHTML=`<div class="detail-actions"><button id="clearSelection">← Back to map</button><button id="exportConcept">Export CSV</button></div><h2>${escapeHTML(n.name)}</h2><div class="meta">${escapeHTML(n.modality)}${family} · ${activation}${n.accuracy==null?'':` · interpretation accuracy ${n.accuracy}%`}</div><p>${escapeHTML(n.interpretation||'No interpretation available.')}</p>${conceptImages(n)}<h2>Outcome wheel</h2><div class="meta">All ${items.length} outcomes · OR = 1 is the middle dashed circle. Spokes show departures from 1: inward below 1, outward above 1. Reciprocal odds ratios use equal distances. ${items.length<=(DATA.wheel_label_limit??20)?'Spokes show outcome, direction and OR. ':''}Hover or focus a spoke for statistics. Faded spokes are not Bonferroni-significant.</div>${wheel(n)}<h2>Outcome associations</h2><p class="assoc-note">Sorted by effect magnitude. Intervals, where available, are 95% CIs. Click an outcome to highlight it on the map.</p><table class="assoc"><thead><tr><th>Outcome</th><th>OR / 95% CI</th><th>Bonferroni p</th></tr></thead><tbody>${rows}</tbody></table>`;
 if(!DATA.outcomes.length){
 const wheelHeader=[...detail.querySelectorAll('h2')].find(x=>x.textContent==='Outcome wheel');
 while(wheelHeader?.nextSibling)wheelHeader.nextSibling.remove();
 if(wheelHeader){wheelHeader.textContent='Concept exploration';wheelHeader.insertAdjacentHTML('afterend','<p>No association testing has been performed. Add outcome labels to test associations.</p>')}
 }
 detail.querySelector('#clearSelection').onclick=clearSelection;
 detail.querySelector('#exportConcept').onclick=()=>exportConcept(n);
 detail.querySelectorAll('.outcome-link').forEach(button=>button.onclick=()=>selectOutcome(button.dataset.outcome));
 detail.querySelectorAll('.wheel-spoke').forEach(spoke=>{const a=A[n.i][Number(spoke.dataset.association)];spoke.onpointermove=e=>associationTip(e,a);spoke.onpointerleave=()=>tip.style.opacity=0;spoke.onfocus=e=>associationTip(e,a);spoke.onblur=()=>tip.style.opacity=0;spoke.onclick=()=>selectOutcome(a.outcome);spoke.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectOutcome(a.outcome)}}});
}
function exportConcept(n){const columns=['concept','outcome','odds_ratio','ci_lower','ci_upper','p_value','corrected_p_value','significant','count'],quote=v=>'"'+String(v??'').replaceAll('"','""')+'"',rows=(A[n.i]||[]).map(a=>columns.map(c=>quote(c==='concept'?n.concept_id??n.feature_idx:a[c])).join(','));const blob=new Blob(['\uFEFF'+columns.join(',')+'\r\n'+rows.join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='glyph-concept-'+String(n.feature_idx)+'.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function show(n){tip.style.opacity=0;selectedNode=n;setSideTab('detail');applyFilters()}
function clearSelection(){selectedNode=null;tip.style.opacity=0;applyFilters()}

const detailTab=document.getElementById('detailTab'),rankingTab=document.getElementById('rankingTab'),technicalTab=document.getElementById('technicalTab'),detailPanel=document.getElementById('detailPanel'),rankingPanel=document.getElementById('rankingPanel'),technicalPanel=document.getElementById('technicalPanel');
function setSideTab(tab){const ranking=tab==='ranking',technical=tab==='technical';detailPanel.hidden=ranking||technical;rankingPanel.hidden=!ranking;technicalPanel.hidden=!technical;detailTab.classList.toggle('active',!ranking&&!technical);rankingTab.classList.toggle('active',ranking);technicalTab.classList.toggle('active',technical);document.body.classList.toggle('ranking-open',ranking||technical)}
detailTab.onclick=()=>setSideTab('detail');rankingTab.onclick=()=>setSideTab('ranking');technicalTab.onclick=()=>setSideTab('technical');
const fmtMetric=v=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toFixed(3);
function renderTechnicalDetails(){const tech=DATA.technical_details;if(isSimple||!tech){technicalTab.hidden=true;return}technicalTab.hidden=false;document.getElementById('technicalNote').textContent=tech.method_note||'';const split=document.getElementById('technicalSplit').value,rows=tech.rows||[];const summary=rows.map(r=>`<tr><td>${escapeHTML(r.modality)}</td><td>${Number(r.input_dim).toLocaleString()}</td><td>${Number(r.n_alive).toLocaleString()} / ${Number(r.dict_size).toLocaleString()}</td><td>${r.top_k}</td><td>${Number(r.sample_counts?.[split]||0).toLocaleString()}</td><td>${fmtMetric(r.reconstruction_r2?.[split])}</td></tr>`).join('');const aucRows=rows.flatMap(r=>(r.aucs||[]).map(a=>{const raw=a.embedding?.[split],sae=a.sae?.[split],delta=raw==null||sae==null?null:sae-raw,cls=delta==null?'':delta>=0?'metric-good':'metric-bad';return`<tr><td>${escapeHTML(r.modality)}</td><td>${escapeHTML(a.outcome)}</td><td>${fmtMetric(raw)}</td><td>${fmtMetric(sae)}</td><td class="${cls}">${delta==null?'—':(delta>=0?'+':'')+delta.toFixed(3)}</td></tr>`})).join('');document.getElementById('technicalContent').innerHTML=`<h2>SAE diagnostics</h2><table class="tech-table"><thead><tr><th>Modality</th><th>Input dim</th><th>Alive / dictionary</th><th>Top-k</th><th>Samples</th><th>Reconstruction R²</th></tr></thead><tbody>${summary}</tbody></table><h2>Linear-probe AUROC · ${escapeHTML(split)}</h2><div class="tech-scroll"><table class="tech-table"><thead><tr><th>Modality</th><th>Outcome</th><th>Embedding</th><th>SAE</th><th>Δ SAE−embedding</th></tr></thead><tbody>${aucRows}</tbody></table></div>`}
document.getElementById('technicalSplit').onchange=renderTechnicalDetails;renderTechnicalDetails();
const stars=p=>p==null?'':p<.001?'***':p<.01?'**':p<.05?'*':'';
const significantByNode=DATA.nodes.map(n=>{const by={};(A[n.i]||[]).filter(a=>a.significant).forEach(a=>by[a.outcome]=a);return by});
const rankedNodes=DATA.nodes.filter(n=>Object.keys(significantByNode[n.i]).length);
const selectedModalities=new Set(DATA.modalities||[]),selectedFamilies=new Set(DATA.embedding_families||[]),facetChoices=document.getElementById('facetChoices');
let associationDirection='both',extremeOnly=!isSimple,extremeThresholds={};
document.getElementById('extremeOnly').checked=extremeOnly;
function recomputeExtremeThresholds(){extremeThresholds={};DATA.outcomes.forEach(outcome=>{const risk=[],protective=[];DATA.nodes.forEach(n=>{if(!contextPasses(n)||!selectedModalities.has(n.modality)||!selectedFamilies.has(n.embedding_family))return;const a=significantByNode[n.i][outcome];if(!a)return;if(a.odds_ratio>1)risk.push(a.odds_ratio);else if(a.odds_ratio<1)protective.push(a.odds_ratio)});risk.sort((a,b)=>a-b);protective.sort((a,b)=>a-b);const riskKeep=Math.min(risk.length,Math.max(5,Math.ceil(risk.length*.1))),protectiveKeep=Math.min(protective.length,Math.max(5,Math.ceil(protective.length*.1)));extremeThresholds[outcome]={risk:risk.length?risk[risk.length-riskKeep]:Infinity,protective:protective.length?protective[protectiveKeep-1]:-Infinity}})}
function associationPasses(a,n){if(!a||!a.significant)return false;const risk=a.odds_ratio>1,protective=a.odds_ratio<1;if(associationDirection==='risk'&&!risk)return false;if(associationDirection==='protective'&&!protective)return false;if(!risk&&!protective)return false;if(!extremeOnly)return true;const threshold=extremeThresholds[a.outcome]||{risk:Infinity,protective:-Infinity};return(risk&&a.odds_ratio>=threshold.risk)||(protective&&a.odds_ratio<=threshold.protective)}
function displayedAssociations(n){const shown={};Object.entries(significantByNode[n.i]).forEach(([outcome,a])=>{if(associationPasses(a,n))shown[outcome]=a});return shown}
function updateAssociationMode(){recomputeExtremeThresholds();const direction=associationDirection==='risk'?'risk · OR > 1':associationDirection==='protective'?'protective · OR < 1':'risk + protective',summary=extremeOnly?`Extreme 10% ${direction} · min 5`:direction;document.getElementById('settingsSummary').textContent=isSimple?'Options':summary;const tail=associationDirection==='risk'?'highest odds ratios':associationDirection==='protective'?'lowest odds ratios':'highest and lowest odds ratios',prefix=extremeOnly?`Most extreme 10% (minimum five) of globally significant associations per direction: ${tail}. `:`All globally significant ${direction} associations. `;document.getElementById('rankNote').textContent=prefix+'Stars use within-modality Bonferroni p: * <.05, ** <.01, *** <.001. Click an outcome header to sort by odds ratio.';recolor();renderRankings();applyFilters()}
function spotlight(n){edgeLayer.querySelectorAll('.edge').forEach(edge=>{edge.style.strokeOpacity=Number(edge.dataset.a)===n.i||Number(edge.dataset.b)===n.i?'.85':'.025'});DATA.nodes.forEach(m=>{m.el.style.opacity=m===n?1:.08;m.el.style.transform=m===n?'scale(2.15)':'scale(1)'});nodeLayer.appendChild(n.el)}
function clearSpotlight(){edgeLayer.querySelectorAll('.edge').forEach(edge=>edge.style.strokeOpacity='');DATA.nodes.forEach(n=>{n.el.style.transform='scale(1)'});applyFilters()}
let hoverOutcome=null,hoverOutcomeAdded=false;
function highlightOutcome(outcome){if(hoverOutcome)return;hoverOutcome=outcome;hoverOutcomeAdded=!selected.has(outcome);if(hoverOutcomeAdded)selected.add(outcome);DATA.nodes.forEach(drawNode);const visible=n=>contextPasses(n)&&selectedModalities.has(n.modality)&&selectedFamilies.has(n.embedding_family),isHit=n=>visible(n)&&associationPasses(significantByNode[n.i][outcome],n),hits=DATA.nodes.filter(isHit);DATA.nodes.forEach(n=>{const hit=isHit(n);n.el.style.opacity=hit?1:.045;n.el.style.transform=hit?'scale(1.38)':'scale(1)';n.el.style.filter=hit?`drop-shadow(0 0 4px ${outcomeColor(outcome)})`:''});hits.forEach(n=>nodeLayer.appendChild(n.el))}
function clearOutcomeHighlight(){if(!hoverOutcome)return;if(hoverOutcomeAdded)selected.delete(hoverOutcome);hoverOutcome=null;hoverOutcomeAdded=false;DATA.nodes.forEach(n=>{n.el.style.filter='';n.el.style.transform='scale(1)'});recolor();applyFilters()}
let activeRankSort=null;
function renderRankings(sortOutcome=activeRankSort){if(isSimple)return;activeRankSort=sortOutcome;clearOutcomeHighlight();const score=n=>{const by=displayedAssociations(n),values=sortOutcome?(by[sortOutcome]?[by[sortOutcome]]:[]):Object.values(by);if(!values.length)return associationDirection==='protective'?Infinity:-Infinity;if(associationDirection==='protective')return Math.min(...values.map(a=>a.odds_ratio));if(associationDirection==='both')return Math.max(...values.map(a=>Math.abs(Math.log(a.odds_ratio))));return Math.max(...values.map(a=>a.odds_ratio))},compare=(a,b)=>associationDirection==='protective'?score(a)-score(b):score(b)-score(a);const rows=rankedNodes.filter(n=>contextPasses(n)&&selectedModalities.has(n.modality)&&selectedFamilies.has(n.embedding_family)&&Object.keys(displayedAssociations(n)).length).sort(compare);const headers=DATA.outcomes.map(o=>`<th data-outcome="${escapeHTML(o)}" style="background:${outcomeColor(o)};color:#fff" title="Sort by ${escapeHTML(o)}">${escapeHTML(o)}</th>`).join('');const body=rows.map(n=>{const by=displayedAssociations(n),cells=DATA.outcomes.map(o=>{const a=by[o];return a?`<td class="rank-cell" style="color:${a.odds_ratio<1?'#2563B8':outcomeColor(o)}">${a.odds_ratio<1?'▼':'▲'} ${a.odds_ratio.toFixed(2)} <span class="stars">${stars(a.modality_corrected_p_value)}</span></td>`:'<td>—</td>'}).join('');return`<tr data-node="${n.i}"><td title="${escapeHTML(n.name)}">${escapeHTML(n.name)}</td><td style="color:${modalityColor(n.modality)};font-weight:700">${escapeHTML(n.modality)}</td>${cells}</tr>`}).join('');const table=document.getElementById('rankTable');table.innerHTML=`<thead><tr><th class="base">Concept</th><th class="base">Modality</th>${headers}</tr></thead><tbody>${body}</tbody>`;table.querySelectorAll('th[data-outcome]').forEach(th=>{th.onmouseenter=()=>highlightOutcome(th.dataset.outcome);th.onmouseleave=clearOutcomeHighlight;th.onclick=()=>renderRankings(th.dataset.outcome)});table.querySelectorAll('tbody tr').forEach(tr=>{const n=DATA.nodes[Number(tr.dataset.node)];tr.onmouseenter=()=>spotlight(n);tr.onmouseleave=clearSpotlight;tr.onclick=()=>{show(n);spotlight(n)}})}
renderRankings();
document.querySelectorAll('input[name="associationDirection"]').forEach(radio=>radio.onchange=()=>{associationDirection=radio.value;updateAssociationMode()});
document.getElementById('extremeOnly').onchange=e=>{extremeOnly=e.target.checked;updateAssociationMode()};
document.querySelectorAll('input[name="colorMode"]').forEach(radio=>radio.onchange=()=>{colorMode=radio.value;document.getElementById('colorSummary').textContent=colorMode==='outcome'?'Color: outcomes':'Color: modalities';recolor()});
function addFacet(title,values,set){const head=document.createElement('b');head.textContent=title;facetChoices.appendChild(head);values.forEach(v=>{const label=document.createElement('label'),box=document.createElement('input'),text=document.createElement('span');box.type='checkbox';box.checked=true;text.textContent=v;label.append(box,text);facetChoices.appendChild(label);box.onchange=()=>{box.checked?set.add(v):set.delete(v);updateAssociationMode()}})}
if(!isSimple||(DATA.modalities||[]).length>1)addFacet('Modalities',DATA.modalities||[],selectedModalities);
if(!isSimple||(DATA.embedding_families||[]).length>1)addFacet('Embedding families',DATA.embedding_families||[],selectedFamilies);
if(isSimple){
 const options=document.querySelector('#displaySettings>.outcomes'),colors=document.getElementById('colorSettings'),facets=document.getElementById('facetPicker');
 if((DATA.modalities||[]).length>1)options.append(...colors.querySelector('.outcomes').children);
 colors.hidden=true;facets.hidden=true;
 if(facetChoices.children.length)options.appendChild(facetChoices);
 if(DATA.contexts?.length){options.appendChild(contextChoices);contextPicker.hidden=true}
 const row=document.createElement('label'),checkbox=document.createElement('input'),text=document.createElement('span');
 checkbox.type='checkbox';checkbox.id='showEdges';checkbox.checked=true;text.textContent='Show connections (up to 8 per concept)';row.append(checkbox,text);options.appendChild(row);edgeLayer.style.display='';checkbox.onchange=()=>{if(checkbox.checked)renderEdges();edgeLayer.style.display=checkbox.checked?'':'none';updateEdgeVisibility()};
 document.querySelector('#displaySettings').setAttribute('aria-label','Display options');
 const filter=document.createElement('label');filter.innerHTML='<input type="checkbox" id="onlySignificant"><span>Only concepts with selected associations</span>';options.appendChild(filter);filter.querySelector('input').onchange=e=>{onlySignificant=e.target.checked;applyFilters()};
 const reset=document.createElement('button');reset.id='filterReset';reset.textContent='Reset filters';reset.onclick=()=>{onlySignificant=false;document.getElementById('onlySignificant').checked=false;document.getElementById('search').value='';associationDirection='both';extremeOnly=false;document.querySelector('input[name="associationDirection"][value="both"]').checked=true;document.getElementById('extremeOnly').checked=false;selectedModalities.clear();(DATA.modalities||[]).forEach(v=>selectedModalities.add(v));selectedFamilies.clear();(DATA.embedding_families||[]).forEach(v=>selectedFamilies.add(v));facetChoices.querySelectorAll('input').forEach(box=>box.checked=true);activeContext=null;const allContext=contextChoices.querySelector('input');if(allContext)allContext.checked=true;selected.clear();selected.add(DATA.initial_outcome);syncOutcomeBoxes();updateAssociationMode()};options.appendChild(reset);
}

function nodePassesFilter(n,q=document.getElementById('search').value.trim().toLowerCase()){return contextPasses(n)&&selectedModalities.has(n.modality)&&selectedFamilies.has(n.embedding_family)&&(!onlySignificant||selectedHits(n).length>0)&&(!q||n.name.toLowerCase().includes(q)||String(n.feature_idx).includes(q)||(n.interpretation||'').toLowerCase().includes(q))}
function updateEdgeVisibility(){edgeLayer.querySelectorAll('.edge').forEach(edge=>{const a=DATA.nodes[Number(edge.dataset.a)],b=DATA.nodes[Number(edge.dataset.b)];edge.style.display=nodePassesFilter(a)&&nodePassesFilter(b)?'':'none'})}
function applyFilters(){
 const q=document.getElementById('search').value.trim().toLowerCase();let matches=0,hits=0;
 if(selectedNode&&!nodePassesFilter(selectedNode,q))selectedNode=null;
 DATA.nodes.forEach(n=>{const facetVisible=contextPasses(n)&&selectedModalities.has(n.modality)&&selectedFamilies.has(n.embedding_family)&&(!onlySignificant||selectedHits(n).length>0),visible=nodePassesFilter(n,q);if(visible){matches++;if(selectedHits(n).length)hits++}n.el.style.display=facetVisible?'':'none';n.hit.style.display=facetVisible?'':'none';n.hit.setAttribute('tabindex',visible?'0':'-1');n.el.classList.toggle('is-selected',n===selectedNode);n.el.style.opacity=visible?(selectedNode?(n===selectedNode?1:.2):BASE_NODE_OPACITY):.06});
 document.getElementById('mapStatus').textContent=`${matches.toLocaleString()} of ${DATA.nodes.length.toLocaleString()} concepts · ${hits.toLocaleString()} with selected associations`;
 document.getElementById('emptyState').hidden=matches>0;updateEdgeVisibility();renderSelected();
}
document.getElementById('search').oninput=applyFilters;
let scale=1,tx=0,ty=0,drag=null,moved=0;
const apply=()=>camera.setAttribute('transform',`translate(${tx},${ty}) scale(${scale})`);
function svgPoint(e){const point=svg.createSVGPoint();point.x=e.clientX;point.y=e.clientY;return point.matrixTransform(svg.getScreenCTM().inverse())}
function zoomAt(point,factor){const next=Math.max(.5,Math.min(15,scale*factor));tx=point.x-(point.x-tx)*next/scale;ty=point.y-(point.y-ty)*next/scale;scale=next;apply()}
svg.addEventListener('wheel',e=>{e.preventDefault();zoomAt(svgPoint(e),e.deltaY<0?1.15:1/1.15)},{passive:false});
svg.style.touchAction='none';svg.onpointerdown=e=>{if(e.button!==0)return;const point=svgPoint(e);drag={x:point.x,y:point.y,tx,ty,clientX:e.clientX,clientY:e.clientY};moved=0};
window.addEventListener('pointerup',e=>{if(svg.hasPointerCapture(e.pointerId))svg.releasePointerCapture(e.pointerId);drag=null});
window.addEventListener('pointercancel',()=>drag=null);
window.addEventListener('pointermove',e=>{if(!drag)return;const point=svgPoint(e);moved=Math.max(moved,Math.abs(e.clientX-drag.clientX)+Math.abs(e.clientY-drag.clientY));if(moved<=4)return;svg.setPointerCapture(e.pointerId);tx=drag.tx+point.x-drag.x;ty=drag.ty+point.y-drag.y;apply()});
svg.onclick=()=>{if(moved<=4)clearSelection()};
document.getElementById('reset').onclick=()=>{scale=1;tx=0;ty=0;apply();clearSelection()};
document.getElementById('zoomIn').onclick=()=>zoomAt({x:S/2,y:S/2},1.3);
document.getElementById('zoomOut').onclick=()=>zoomAt({x:S/2,y:S/2},1/1.3);
document.addEventListener('click',e=>{document.querySelectorAll('#toolbar details[open]').forEach(menu=>{if(!menu.contains(e.target))menu.open=false})});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.querySelectorAll('details[open]').forEach(menu=>menu.open=false);clearSelection();endTour()}});
if(!DATA.outcomes.length){
 document.getElementById('outcomePicker').hidden=true;
 document.querySelectorAll('#displaySettings input[name="associationDirection"],#extremeOnly,#onlySignificant').forEach(x=>x.closest('label').hidden=true);
 colorMode='modality';
 document.getElementById('intro').textContent='Select a concept to explore its activation frequency and available interpretation. Add outcome labels for association discovery.';
}
updateSummary();updateAssociationMode();apply();

const TOUR=[
 {target:'stage',title:'Read the concept map',text:'Each node is a learned concept. Solid outcome color means predictive risk (OR > 1); diagonal stripes mean protective association (OR < 1). Pie sectors show concepts associated with multiple selected outcomes. Drag to pan and scroll to zoom.'},
 {target:'displaySettings',title:'Choose what to display',text:'Choose outcomes and use Options to filter association direction, change colors or show connections. Solid nodes indicate higher odds; striped nodes indicate lower odds.'},
 {target:'side',title:'Explore a concept',text:'Select a concept to see its interpretation, outcome wheel and strongest associations.'}
];
let tourIndex=0,tourTarget=null;const tourShade=document.getElementById('tourShade'),tourCard=document.getElementById('tourCard'),TOUR_STORAGE_KEY=`glyph-tour-v2:${DATA.title}`;
function placeTour(){if(!tourTarget)return;const r=tourTarget.getBoundingClientRect(),w=tourCard.offsetWidth,h=tourCard.offsetHeight;let left,top;if(r.width>500){left=r.left+28;top=r.top+82}else{left=Math.min(innerWidth-w-16,Math.max(16,r.left));top=r.bottom+14;if(top+h>innerHeight-16)top=Math.max(16,r.top-h-14)}tourCard.style.left=left+'px';tourCard.style.top=top+'px'}
function showTourStep(){if(tourTarget)tourTarget.classList.remove('tour-target');const step=TOUR[tourIndex];tourTarget=document.getElementById(step.target);tourTarget.classList.add('tour-target');document.getElementById('tourKicker').textContent=`Quick tour · ${tourIndex+1} of ${TOUR.length}`;document.getElementById('tourTitle').textContent=step.title;document.getElementById('tourText').textContent=step.text;document.getElementById('tourDots').textContent=TOUR.map((_,i)=>i===tourIndex?'●':'○').join(' ');document.getElementById('tourBack').hidden=tourIndex===0;document.getElementById('tourNext').textContent=tourIndex===TOUR.length-1?'Done':'Next';requestAnimationFrame(placeTour)}
function startTour(){tourIndex=0;tourShade.hidden=false;tourCard.hidden=false;showTourStep()}
function endTour(){tourShade.hidden=true;tourCard.hidden=true;if(tourTarget)tourTarget.classList.remove('tour-target');tourTarget=null;try{localStorage.setItem(TOUR_STORAGE_KEY,'done')}catch(e){}}
function tourWasSeen(){try{return localStorage.getItem(TOUR_STORAGE_KEY)==='done'}catch(e){return false}}
document.getElementById('tourNext').onclick=()=>{if(tourIndex===TOUR.length-1)endTour();else{tourIndex++;showTourStep()}};document.getElementById('tourBack').onclick=()=>{tourIndex--;showTourStep()};document.getElementById('tourSkip').onclick=endTour;document.getElementById('tourButton').onclick=startTour;window.addEventListener('resize',placeTour);if(!isSimple&&!tourWasSeen())setTimeout(startTour,250);
</script></body></html>'''
