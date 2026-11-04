"""Prevalence-proportional node areas for Glyph concept maps.

Prevalence is a fraction in [0, 1]. Density scaling multiplies all areas by
one common factor; it never normalizes prevalence against the largest node.
"""
import numpy as np


def node_area_parameters(n_nodes, full_prevalence_area=400., minimum_area=.25):
    density_scale=min(1.,2048./max(1,int(n_nodes)))
    return dict(full_prevalence_area=float(full_prevalence_area*density_scale),
                minimum_area=float(minimum_area),unknown_area=float(12.*density_scale))


def prevalence_node_areas(prevalence, n_nodes=None):
    rates=np.asarray(prevalence,dtype=float)
    if ((rates[np.isfinite(rates)]<0)|(rates[np.isfinite(rates)]>1)).any():
        raise ValueError('Node prevalence must be a fraction in [0, 1]')
    settings=node_area_parameters(rates.size if n_nodes is None else n_nodes)
    return np.where(np.isfinite(rates),
        np.maximum(settings['minimum_area'],settings['full_prevalence_area']*rates),
        max(settings['minimum_area'],settings['unknown_area']))
