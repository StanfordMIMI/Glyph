"""PheWAS (Phenome-Wide Association Study) for hypothesis discovery."""

from glyph.phewas.phewas import PheWAS
from glyph.phewas.statistics import (
    compute_odds_ratio,
    compute_feature_prevalence,
    compute_outcome_prevalence,
)

__all__ = [
    "PheWAS",
    "compute_odds_ratio",
    "compute_feature_prevalence", 
    "compute_outcome_prevalence",
]
