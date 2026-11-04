import numpy as np
import pytest
from glyph.visualization.node_sizes import node_area_parameters,prevalence_node_areas


def test_area_tracks_absolute_prevalence_without_a_large_offset():
    sizes=prevalence_node_areas([.01,.05,.25,1.])
    np.testing.assert_allclose(sizes,[4.,20.,100.,400.])
    assert sizes[2]/sizes[1]==5
    # Adding a more prevalent concept cannot rescale existing values.
    np.testing.assert_array_equal(sizes[:3],prevalence_node_areas([.01,.05,.25]))


def test_large_maps_share_one_density_factor_and_keep_rare_nodes_small():
    sizes=prevalence_node_areas([.003,.077,.45],n_nodes=8192)
    np.testing.assert_allclose(sizes,[.3,7.7,45.])
    assert sizes[1]/sizes[2]==pytest.approx(.077/.45)
    assert node_area_parameters(8192)['full_prevalence_area']==100.
    np.testing.assert_array_equal(prevalence_node_areas([0,1e-6],n_nodes=8192),[.25,.25])
    assert np.isfinite(prevalence_node_areas([np.nan])).all()


@pytest.mark.parametrize('rates',[[-.1],[1.1],[10.]])
def test_rates_are_fractions(rates):
    with pytest.raises(ValueError,match='fraction'):prevalence_node_areas(rates)


def test_correlation_line_width_and_opacity_are_small_and_proportional():
    from glyph.visualization.correlation_edges import correlation_edge_style
    widths,opacity=correlation_edge_style([.25,.5,-1.])
    np.testing.assert_allclose(widths,[.075,.15,.3])
    np.testing.assert_allclose(opacity,[.035,.07,.14])


def test_outcome_palette_is_distinct_and_does_not_merge_digits():
    from glyph.visualization.colors import outcome_palette
    names=[f'digit{i}' for i in range(10)]
    assert len(set(outcome_palette(names).values()))==10
    assert outcome_palette(names)==outcome_palette(list(reversed(names)))
    assert len(set(outcome_palette([str(i) for i in range(150)]).values()))==150


def test_digit_colors_follow_a_fixed_order_even_for_a_subset():
    from glyph.visualization.colors import outcome_palette,DIGIT_COLORS
    palette=outcome_palette([f'is_digit_{i}' for i in range(10)])
    assert list(palette.values())==DIGIT_COLORS
    assert outcome_palette(['digit7','digit2'])=={'digit2':DIGIT_COLORS[2],'digit7':DIGIT_COLORS[7]}
