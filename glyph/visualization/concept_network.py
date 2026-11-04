import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from pathlib import Path


def plot_concept_network(
    sae_features,
    alive_mask,
    phewas_results,
    outcome,
    output_path,
    split='train',
    correlation_percentile=0.975,
    bonferroni_alpha=0.05,
    figsize=(14, 14),
    dpi=150,
    layout_k=None,
    seed=42,
    autointerp_results=None,
    modality=None,
    top_k=None,
):
    """
    Force-directed graph of the SAE concept space.

    Nodes are alive SAE features. Edges connect highly correlated features
    (top correlation_percentile of pairwise correlations). Node color encodes
    the odds ratio for `outcome` on `split`, but only for features whose
    p-value survives Bonferroni correction; all others are grey.
    Node size encodes activation rate.

    Args:
        sae_features: Dict[split] -> tensor (n_samples, n_features)
        alive_mask: boolean tensor/array of shape (n_features,)
        phewas_results: DataFrame from PheWAS with columns
            feature_idx, outcome, split, odds_ratio, odds_ratio_p_value, c_f
        outcome: outcome column name to color by
        output_path: path to save the JPG
        split: which split to use for correlation and coloring
        correlation_percentile: quantile threshold for edges
        bonferroni_alpha: alpha before Bonferroni correction
        figsize: matplotlib figure size
        dpi: output resolution
        layout_k: spring layout k parameter (None = auto)
        seed: random seed for layout
    """
    if 'odds_ratio_p_value_bonferroni' not in phewas_results:
        from glyph.phewas.phewas import PheWAS
        phewas_results = PheWAS._add_multiple_testing_corrections(phewas_results)
    for column, value in [('modality', modality), ('top_k', top_k)]:
        if column in phewas_results:
            if value is None and phewas_results[column].nunique() > 1:
                raise ValueError(f"Specify {column} when plotting multiple dictionaries")
            if value is not None:
                phewas_results = phewas_results[phewas_results[column] == value]
    if phewas_results.duplicated(['split', 'outcome', 'feature_idx']).any():
        raise ValueError("Ambiguous duplicate hypotheses; select one dictionary/run")
    if autointerp_results is not None:
        autointerp_results = autointerp_results.copy()
        for column, value in [('modality', modality), ('top_k', top_k)]:
            if column in autointerp_results and value is not None:
                autointerp_results = autointerp_results[autointerp_results[column] == value]

    # --- Prepare activations ---
    feats = sae_features[split]
    if isinstance(feats, torch.Tensor):
        feats = feats.cpu().numpy()

    if isinstance(alive_mask, torch.Tensor):
        alive_mask = alive_mask.cpu().numpy()

    # Restrict to features that appear in phewas_results for this split
    # (these are the features that passed min_activations in PheWAS)
    phewas_feature_indices = set(
        phewas_results[phewas_results['split'] == split]['feature_idx'].unique()
    )
    alive_indices = np.array([
        i for i in np.where(alive_mask)[0] if i in phewas_feature_indices
    ])

    if len(alive_indices) == 0:
        raise ValueError("No supported concepts for the selected plot")
    feats_alive = feats[:, alive_indices]  # (n_samples, n_alive)
    n_alive = len(alive_indices)

    # --- Correlation matrix for edges ---
    # Use pandas for NaN-safe corr, but numpy is fine here since alive features are non-degenerate
    corr = np.corrcoef(feats_alive.T) if n_alive > 1 else np.ones((1, 1))
    corr = np.nan_to_num(corr)  # (n_alive, n_alive)
    corr_abs = np.abs(corr)
    np.fill_diagonal(corr_abs, 0)

    upper = corr_abs[np.triu_indices(n_alive, k=1)]
    upper = upper[~np.isnan(upper)]
    threshold = np.quantile(upper, correlation_percentile) if len(upper) > 0 else 0.0

    # --- PheWAS lookup for this outcome/split ---
    mask = (phewas_results['outcome'] == outcome) & (phewas_results['split'] == split)
    df_sub = phewas_results[mask].set_index('feature_idx')


    # --- Build graph ---
    G = nx.Graph()

    for local_idx, global_idx in enumerate(alive_indices):
        if global_idx in df_sub.index:
            row = df_sub.loc[global_idx]
            c_f = row['c_f']
            activation_rate = c_f / feats.shape[0]
            p_val = row.get('odds_ratio_p_value_bonferroni', np.nan)
            odds_ratio = row['odds_ratio'] if not np.isnan(row['odds_ratio']) else 1.0
            significant = (not np.isnan(p_val)) and (p_val < bonferroni_alpha)
        else:
            activation_rate = (feats_alive[:, local_idx] > 1e-8).mean()
            odds_ratio = 1.0
            significant = False

        G.add_node(
            local_idx,
            global_idx=global_idx,
            activation_rate=float(activation_rate),
            odds_ratio=float(odds_ratio),
            significant=significant,
        )

    for i in range(n_alive):
        for j in range(i + 1, n_alive):
            w = corr_abs[i, j]
            if not np.isnan(w) and w >= threshold:
                G.add_edge(i, j, weight=float(w))

    # Ensure no isolated nodes (connect each to its strongest neighbour)
    for i in range(n_alive):
        if n_alive > 1 and G.degree(i) == 0:
            best_j = int(np.argmax([corr_abs[i, j] if j != i else -1 for j in range(n_alive)]))
            G.add_edge(i, best_j, weight=float(corr_abs[i, best_j]))

    from glyph.visualization.correlation_edges import limit_correlation_edges
    kept=limit_correlation_edges([[a,b,attr['weight']] for a,b,attr in G.edges(data=True)],n_alive)
    G.remove_edges_from(list(G.edges()))
    G.add_weighted_edges_from(kept)

    # --- Layout ---
    from glyph.visualization.interactive_concept_network import _layout
    dist_matrix = (1 - corr_abs).clip(0, 1)
    np.fill_diagonal(dist_matrix, 0)
    embedding = _layout(1 - dist_matrix, seed)
    pos = {i: (float(embedding[i, 0]), float(embedding[i, 1])) for i in range(n_alive)}

    # Marker area is proportional to absolute prevalence, with a tiny floor.
    from glyph.visualization.node_sizes import prevalence_node_areas
    rates = np.array([G.nodes[n]['activation_rate'] for n in G.nodes()])
    node_sizes = prevalence_node_areas(rates)

    # --- Node colors ---
    # Significant nodes: map log(OR) through diverging colormap centred at 0
    # Non-significant: grey
    sig_flags = np.array([G.nodes[n]['significant'] for n in G.nodes()])
    or_vals = np.array([G.nodes[n]['odds_ratio'] for n in G.nodes()])

    log_or = np.where(sig_flags & (or_vals > 0), np.log(np.clip(or_vals, 1e-6, None)), 0.0)
    # Symmetric colormap range
    vmax = np.abs(log_or[sig_flags]).max() if sig_flags.any() else 1.0
    vmax = max(vmax, 0.1)

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    node_colors = []
    for n in G.nodes():
        if G.nodes[n]['significant']:
            node_colors.append(cmap(norm(np.log(max(G.nodes[n]['odds_ratio'], 1e-6)))))
        else:
            node_colors.append((0.75, 0.75, 0.75, 0.6))

    from glyph.visualization.correlation_edges import correlation_edge_style
    edge_widths,edge_opacity=correlation_edge_style([G[a][b]['weight'] for a,b in G.edges()])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_opacity, width=edge_widths, edge_color='#aaaaaa')
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.85,
        linewidths=0.2,
        edgecolors='#333333',
    )

    # --- Autointerp labels ---
    if autointerp_results is not None and not autointerp_results.empty:
        interp_lookup = {}
        for _, row in autointerp_results.iterrows():
            acc = row.get('test_accuracy', 0.0)
            if acc is None or acc <= 0.5:
                continue
            name = row.get('feature_name', '')
            if not isinstance(name, str) or not name.startswith('feature_'):
                continue
            try:
                fidx = int(name.split('_')[1])
            except (IndexError, ValueError):
                continue
            interp = str(row.get('interpretation', ''))
            if interp:
                interp_lookup[fidx] = (interp, acc)

        for local_idx, global_idx in enumerate(alive_indices):
            if not G.nodes[local_idx]['significant']:
                continue
            if global_idx not in interp_lookup:
                continue
            interp, acc = interp_lookup[global_idx]
            for prefix in ('This cohort is characterized by ', 'This cohort is characterized by a ',
                           'This cohort is characterized by an '):
                if interp.lower().startswith(prefix.lower()):
                    interp = interp[len(prefix)].upper() + interp[len(prefix)+1:]
                    break
            label = (interp[:35] + '…' if len(interp) > 35 else interp) + f' ({acc:.0%})'
            x, y = pos[local_idx]
            ax.text(
                x, y + 0.03, label,
                fontsize=5.5, ha='center', va='bottom',
                color='#111111',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='none'),
            )

    # Colorbar for significant nodes
    if sig_flags.any():
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
        cbar.set_label(f'log(OR) — {outcome}', fontsize=10)
        n_sig = int(sig_flags.sum())
        ax.set_title(
            f'Concept space  |  {n_alive} features  |  {n_sig} significant (Bonferroni p<{bonferroni_alpha})',
            fontsize=12,
        )
    else:
        ax.set_title(
            f'Concept space  |  {n_alive} features  |  no Bonferroni-significant features for "{outcome}"',
            fontsize=12,
        )

    ax.set_axis_off()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format='jpeg')
    plt.close()
    print(f"  ✓ Concept network saved to: {output_path}")
