"""
Glyph: Mechanistic Science - Domain-Agnostic Hypothesis Generation

A toolkit for discovering interpretable features and generating scientific hypotheses
from multimodal data using Sparse Autoencoders and statistical analysis.
"""

__version__ = "0.1.0"

__all__ = ["Glyph"]


def __getattr__(name):
    if name == 'Glyph':
        from glyph.core.glyph import Glyph
        globals()['Glyph'] = Glyph
        return Glyph
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
