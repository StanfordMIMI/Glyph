import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def plot_feature_lollipop(
    phewas_results,
    outcome,
    output_path,
    split='train',
    autointerp_results=None,
    figsize=(16, 7),
    dpi=150,
):
    if autointerp_results is None or autointerp_results.empty:
        print("  No autointerp results — skipping lollipop.")
        return

    # Build interp lookup: feature_idx -> (short_label, acc)
    interp_lookup = {}
    for _, row in autointerp_results.iterrows():
        acc = row.get('test_accuracy', 0.0) or 0.0
        name = str(row.get('feature_name', ''))
        if not name.startswith('feature_'):
            continue
        try:
            fidx = int(name.split('_')[1])
        except (IndexError, ValueError):
            continue
        if acc <= 0.5:
            interp_lookup[fidx] = ('Undocumented AI concept', acc)
        else:
            interp = str(row.get('interpretation', ''))
            for prefix in ('This cohort is characterized by a ',
                           'This cohort is characterized by an ',
                           'This cohort is characterized by '):
                if interp.lower().startswith(prefix.lower()):
                    interp = interp[len(prefix)].upper() + interp[len(prefix)+1:]
                    break
            if interp:
                interp_lookup[fidx] = (interp, acc)

    if not interp_lookup:
        print("  No interpreted features with acc > 50% — skipping lollipop.")
        return

    df = phewas_results[
        (phewas_results['outcome'] == outcome) &
        (phewas_results['split'] == split) &
        (phewas_results['feature_idx'].isin(interp_lookup.keys()))
    ].copy().sort_values('odds_ratio', ascending=True).reset_index(drop=True)

    if df.empty:
        print("  No PheWAS rows for interpreted features — skipping lollipop.")
        return

    labels = []
    for _, row in df.iterrows():
        interp, acc = interp_lookup[int(row['feature_idx'])]
        wrapped = textwrap.fill(f"{interp}  [{acc:.0%}]", width=60)
        labels.append(wrapped)

    or_vals = df['odds_ratio'].values
    log_or = np.log(np.clip(or_vals, 1e-6, None))
    vmax = max(np.abs(log_or).max(), 0.1)

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    colors = [cmap(norm(lo)) for lo in log_or]

    n_total = phewas_results[phewas_results['split'] == split]['c_f'].max()
    rates = df['c_f'].values / max(n_total, 1)
    r_min, r_max = rates.min(), rates.max()
    sizes = 80 + 320 * (rates - r_min) / max(r_max - r_min, 1e-8)

    height = len(df) * 0.6 + 2
    fig, ax = plt.subplots(figsize=(figsize[0], height))
    y = np.arange(len(df))

    for yi, or_val, color in zip(y, or_vals, colors):
        ax.hlines(yi, 1.0, or_val, colors=color, linewidths=2.2, alpha=0.75)

    ax.scatter(or_vals, y, s=sizes, c=colors, zorder=3, linewidths=0.6, edgecolors='#333333')

    ax.axvline(1.0, color='#555555', linewidth=1.0, linestyle='--', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Odds ratio (log scale)', fontsize=10)
    ax.set_title(f'{outcome}  ·  {split} split', fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.6)
    cbar.set_label('log(OR)', fontsize=9)

    for rate, label in [(r_min, 'low activation'), (r_max, 'high activation')]:
        s = 80 + 320 * (rate - r_min) / max(r_max - r_min, 1e-8)
        ax.scatter([], [], s=s, c='grey', alpha=0.7, edgecolors='#333333', label=label)
    ax.legend(title='Activation rate', fontsize=8, title_fontsize=8,
              loc='lower right', framealpha=0.85)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.25, linestyle=':')
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format='jpeg')
    plt.close()
    print(f"  ✓ Feature lollipop saved to: {output_path}")
