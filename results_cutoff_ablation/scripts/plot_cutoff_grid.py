#!/usr/bin/env python3
"""Projected Hessian MAE vs molecule size for each checkpoint and inference Hessian-graph cutoff.

Usage: plot_cutoff_grid.py <cutoff-grid dir> <outlier_samples.csv> <out prefix>
Left panel: released hip_v2 / hip_v3. Right panel: cutoff ablations. Line style = model,
colour = inference cutoff (sequential). Median per 10-atom bin (90-100 inclusive), plotted at the bin's mean N, 51 reference outliers excluded.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

grid, outl, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
bad = set(pd.read_csv(outl).sample_name)
CUTS = ['4', '5', '6', '7', '9', '12', '16', '32', 'full']
cmap = plt.get_cmap('viridis'); COL = {c: cmap(0.9 * i / (len(CUTS) - 1)) for i, c in enumerate(CUTS)}
PANELS = [('Released checkpoints', {'hip_v2': ('-', 'hip_v2 (trained 5 / 5)'), 'hip_v3': ('--', 'hip_v3 (trained 5 / full)')}),
          ('Cutoff ablations (epoch 100)', {'abl_5_5': ('-', 'backbone 5 / Hessian 5'), 'abl_5_all': ('--', 'backbone 5 / Hessian full'),
                                            'abl_12_12': (':', 'backbone 12 / Hessian 12')})]
METRIC = 'projected_hessian_mae_ev_a2_amu'
bins = [30, 40, 50, 60, 70, 80, 90, 101]  # last bin includes N = 100
rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
for ax, (title, models) in zip(axes, PANELS):
    for m, (ls, _) in models.items():
        for c in CUTS:
            f = grid / f'{m}_H{c}' / 'metrics.csv'
            if not f.exists(): continue
            df = pd.read_csv(f); df = df[(df.status == 'ok') & ~df.sample_name.isin(bad)]
            df['bin'] = pd.cut(df.natoms, bins, right=False)
            g = df.groupby('bin', observed=True).agg(y=(METRIC, 'median'), x=('natoms', 'mean'))
            g = g.set_axis(['y', 'x'], axis=1); x = g.x.values; g = g.y
            ax.plot(x, g.values, ls=ls, color=COL[c], lw=1.6)
            rows.append(dict(model=m, cutoff=c, n=len(df), mean=df[METRIC].mean(), median=df[METRIC].median()))
    ax.set_title(title, fontsize=11); ax.set_yscale('log'); ax.set_xlabel('number of atoms N')
    ax.grid(True, which='major', color='0.9', lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(handles=[Line2D([], [], ls=ls, color='0.25', lw=1.6, label=lab) for ls, lab in models.values()],
              loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, columnspacing=1.2, handlelength=2.6, fontsize=8, frameon=False, title='line style = model', title_fontsize=8)
axes[0].set_ylabel('projected Hessian MAE, median per 10-atom bin\n[eV / (Å² amu)]')
fig.legend(handles=[Line2D([], [], color=COL[c], lw=2.2, label=('fully connected' if c == 'full' else f'{c} Å')) for c in CUTS],
           title='Hessian graph at inference', loc='center right', fontsize=8, title_fontsize=8, frameon=False)
fig.suptitle('PubChem 30–100 atoms (n = 659, reference outliers excluded): error vs size by inference Hessian cutoff', fontsize=11)
fig.tight_layout(rect=(0, 0.02, 0.86, 0.95))
for ext in ('png', 'pdf'): fig.savefig(f'{out}.{ext}', dpi=200)
t = pd.DataFrame(rows); t.to_csv(f'{out}-summary.csv', index=False)
print(t.pivot(index='model', columns='cutoff', values='median')[CUTS].to_string(float_format=lambda v: f'{v:.4g}'))
