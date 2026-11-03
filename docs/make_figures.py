"""Figures of the README and the documentation, from the synthetic results of simulation v2.

Run from the repository root after ``python -m simulation_v2.run summary --level evaluation``:

    python docs/make_figures.py
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'simulation_v2' / 'results' / 'evaluation' / 'methods.csv'
OUT = ROOT / 'docs' / 'images'
METHODS = [('rabmv', 'RabMV (rule while fitting)', '#1b6ca8'),
           ('ebm_mice', 'EBM + chained imputation', '#7a7a7a'),
           ('rabmv_pattern', 'Pattern submodels', '#9e9e9e'),
           ('ebm_native', 'EBM, native missing bin', '#d95f02'),
           ('ebm_indicator', 'EBM + missing indicator', '#e6ab02'),
           ('trees_native', 'Boosted trees, native NaN', '#a6761d')]


def transport():
    t = pd.read_csv(SOURCE)
    t = t[(t['events'] == 180) & (t['metric'] == 'excess_logloss')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, gap, title in zip(axes, ('education', 'occupation'), ('Gap in education', 'Gap in occupation')):
        q = t[(t['condition'] == f'E_pi100_{gap}')].set_index(['target', 'method'])
        for k, (m, label, color) in enumerate(METHODS):
            vals = [q.loc[(tg, m), 'mean'] for tg in ('T0', 'T3')]
            errs = [1.96 * q.loc[(tg, m), 'mcse'] for tg in ('T0', 'T3')]
            ax.bar([k - 0.2, k + 0.2 + len(METHODS)], vals, 0.4, yerr=errs, color=color, label=label,
                   edgecolor='white', linewidth=0.5)
        ax.set_xticks([(len(METHODS) - 1) / 2, len(METHODS) + (len(METHODS) - 1) / 2],
                      ['Training registry (T0)', 'Another municipality (T3)'])
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Excess log-loss over the true risk')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle('The form equals the class: a model that learns the missingness fails when the registry changes',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'transport.png', dpi=200)
    fig.savefig(OUT / 'transport.svg')
    plt.close(fig)


if __name__ == '__main__':
    transport()
    print(f'written {OUT}')
