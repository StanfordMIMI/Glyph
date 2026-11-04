"""Visualizations and optional viewer hosting, loaded on demand."""
from importlib import import_module

_EXPORTS = {
    'plot_concept_network': 'glyph.visualization.concept_network',
    'plot_feature_lollipop': 'glyph.visualization.feature_lollipop',
    'build_interactive_concept_atlas': 'glyph.visualization.interactive_concept_network',
    'build_interactive_concept_network': 'glyph.visualization.interactive_concept_network',
    'serve_html': 'glyph.visualization.hosting',
    'serve_over_ssh': 'glyph.visualization.hosting',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
