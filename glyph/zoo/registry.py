"""Registry of pretrained SAEs available to glyph.

Concept indices are only meaningful relative to a specific encoder and SAE
configuration, so each entry pins the encoder, dimensions and a HuggingFace
revision. Entries are versioned (`name@N`); a retrained SAE gets a new version
rather than redefining the concepts of an existing one.
"""

from dataclasses import dataclass

HF_REPO = "RobbieHolland/HypothesisExtractor"
HF_REVISION = "23c5da2d6d415e88255912da398e83035d507069"


@dataclass(frozen=True)
class ZooEntry:
    name: str
    modality: str
    encoder: str
    input_dim: int
    n_concepts: int
    top_k: int
    weights: str
    repo: str
    revision: str


ZOO = {
    "ct-merlin@1": ZooEntry(
        name="ct-merlin@1",
        modality="image",
        encoder="merlin-vlm: Merlin(ImageEmbedding=True)",
        input_dim=2048,
        n_concepts=8192,
        top_k=40,
        weights="sae_image.ckpt",
        repo=HF_REPO,
        revision=HF_REVISION,
    ),
    "findings-qwen3@1": ZooEntry(
        name="findings-qwen3@1",
        modality="findings",
        encoder="Qwen/Qwen3-Embedding-8B (last-token pooled)",
        input_dim=4096,
        n_concepts=8192,
        top_k=40,
        weights="sae_findings.ckpt",
        repo=HF_REPO,
        revision=HF_REVISION,
    ),
    "labs-loinc@1": ZooEntry(
        name="labs-loinc@1",
        modality="labs",
        encoder="glyph labs encoder (LOINC-coded quarterly panels)",
        input_dim=768,
        n_concepts=8192,
        top_k=40,
        weights="sae_labs.ckpt",
        repo=HF_REPO,
        revision=HF_REVISION,
    ),
}


def resolve(name):
    """Resolve a zoo name, defaulting to the highest version when unversioned."""
    if name in ZOO:
        return ZOO[name]
    versions = sorted(k for k in ZOO if k.rsplit("@", 1)[0] == name)
    if not versions:
        raise KeyError(
            f"Unknown pretrained SAE {name!r}. Available: {sorted(ZOO)}"
        )
    return ZOO[max(versions, key=lambda key: int(key.rsplit("@", 1)[1]))]


def available():
    """Names of all registered pretrained SAEs."""
    return sorted(ZOO)
