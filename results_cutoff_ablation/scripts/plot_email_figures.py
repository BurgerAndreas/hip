#!/usr/bin/env python3
"""Figures and markdown tables for the email reply on HIP size extrapolation (PubChem 30-100 atoms).

Run from results_cutoff_ablation/:
  uv run --no-project --with pandas,matplotlib,numpy python scripts/plot_email_figures.py
Outputs: pubchem/email-figures/fig{1..4}_*.{png,pdf} and pubchem/email-figures/tables.md
Conventions follow scripts/plot_cutoff_grid.py (bins [30,40,...,90,101], median per bin, log y,
51 reference outliers excluded -> clean n = 659).
"""
import re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

P = Path('pubchem'); OUT = P / 'email-figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight'})
# Okabe-Ito colorblind-safe palette
OI = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00', purple='#CC79A7',
          sky='#56B4E9', yellow='#F0E442', black='#000000')
PH, EV, COS = 'projected_hessian_mae_ev_a2_amu', 'eckart_eigval_mae_ev_a2', 'eckart_abs_cos_v1'
U = 'eV/(Å² amu)'
FLAT = ['PubChem_' + s for s in '5280344 5280349 5280489 5280805 5281673 5281675 5281792 5281794'.split()]
BAD = set(pd.read_csv(P / 'outlier_samples.csv').sample_name)
CUTS = ['4', '5', '6', '7', '9', '12', '16', '32', 'full']
XPOS = {c: (64.0 if c == 'full' else float(c)) for c in CUTS}
BINS = [30, 40, 50, 60, 70, 80, 90, 101]
T1X_MAX = 12.21


def load(path):
    df = pd.read_csv(path)
    assert len(df) == 710 and (df.status == 'ok').all(), path
    return df


def clean(df):
    return df[~df.sample_name.isin(BAD)]


def nonflat(df):
    return df[~df.sample_name.isin(FLAT)]


def grid(m, c):
    return load(P / 'cutoff-grid' / f'{m}_H{c}' / 'metrics.csv')


def style(ax):
    ax.grid(True, which='major', color='0.9', lw=0.6); ax.set_axisbelow(True)


def save(fig, name):
    for ext in ('png', 'pdf'): fig.savefig(OUT / f'{name}.{ext}', dpi=200)
    plt.close(fig)


GRID_MODELS = {'hip_v2': ('hip_v2 (released; trained 5 / 5 Å)', OI['blue'], '-', 'o', 2.0),
               'hip_v3': ('hip_v3 (released; trained 5 Å / full)', OI['red'], '-', 's', 2.0),
               'abl_5_5': ('ablation 5 / 5 Å', OI['sky'], '--', 'o', 1.2),
               'abl_5_all': ('ablation 5 Å / full', OI['orange'], '--', 's', 1.2),
               'abl_12_12': ('ablation 12 / 12 Å', OI['green'], '--', '^', 1.2)}
G = {(m, c): grid(m, c) for m in GRID_MODELS for c in CUTS}

# sanity: cutoff-grid 'full' == released maxnb1000 (fully connected default)
for m in ('hip_v2', 'hip_v3'):
    r = load(P / 'released' / f'{m}_maxnb1000' / 'metrics.csv')
    a, b = clean(G[(m, 'full')])[PH].median(), clean(r)[PH].median()
    print(f'check {m}: grid full median {a:.5f} vs released maxnb1000 {b:.5f}')

# ---------------- Figure 1: inference Hessian-graph cutoff sweep ----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (met, lab) in zip(axes, [(PH, f'projected Hessian MAE [{U}]'), (EV, f'Eckart eigenvalue MAE [{U}]')]):
    for m, (name, col, ls, mk, lw) in GRID_MODELS.items():
        y = [clean(G[(m, c)])[met].median() for c in CUTS]
        x = [XPOS[c] for c in CUTS]
        ax.plot(x[:-1], y[:-1], ls=ls, color=col, lw=lw, marker=mk, ms=4 if lw > 1.5 else 3, label=name)
        ax.plot(x[-2:], y[-2:], ls=':', color=col, lw=lw * 0.8)  # 32 -> full (broken axis)
        ax.plot(x[-1], y[-1], ls='none', color=col, marker=mk, ms=4 if lw > 1.5 else 3)
    ax.axvline(7.5, color='0.45', lw=0.9, ls='-.')
    ax.axvline(T1X_MAX, color='0.2', lw=0.9, ls='-.')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([XPOS[c] for c in CUTS]); ax.set_xticklabels([('full' if c == 'full' else c) for c in CUTS])
    ax.minorticks_off(); ax.set_yscale('log')
    ax.set_xlabel('Hessian-graph cutoff at inference [Å]')
    ax.set_ylabel(f'median over clean molecules\n{lab}')
    style(ax)
    yl = ax.get_ylim()
    ax.text(7.5 * 0.97, yl[1] * 0.85, '7.5 Å', ha='right', va='top', fontsize=7, color='0.35')
    ax.text(T1X_MAX * 1.03, yl[0] * 1.3, 'T1x max pair\ndistance 12.21 Å', ha='left', va='bottom', fontsize=7, color='0.2')
axes[0].set_title('(a) projected Hessian MAE'); axes[1].set_title('(b) Eckart eigenvalue MAE')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
fig.suptitle('PubChem 30–100 atoms, clean set (n = 659); backbone graph as trained, max_neighbors 1000', fontsize=10)
fig.tight_layout()
save(fig, 'fig1_cutoff_sweep')

# ---------------- Figure 2: error vs N ----------------
SIZE = [('hip_v3', 'full', 'hip_v3, fully connected (code default)', OI['red'], ':'),
        ('hip_v3', '7', 'hip_v3, 7 Å Hessian graph', OI['red'], '-'),
        ('hip_v3', '5', 'hip_v3, 5 Å Hessian graph', OI['orange'], '-'),
        ('hip_v2', 'full', 'hip_v2, fully connected (code default)', OI['blue'], ':'),
        ('hip_v2', '5', 'hip_v2, 5 Å Hessian graph', OI['blue'], '-')]
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for m, c, name, col, ls in SIZE:
    df = clean(G[(m, c)]).copy(); df['bin'] = pd.cut(df.natoms, BINS, right=False)
    g = df.groupby('bin', observed=True).agg(x=('natoms', 'mean'), y=(PH, 'median'),
                                            lo=(PH, lambda s: s.quantile(.25)), hi=(PH, lambda s: s.quantile(.75)))
    ax.fill_between(g.x, g.lo, g.hi, color=col, alpha=0.15, lw=0)
    ax.plot(g.x, g.y, ls=ls, color=col, lw=1.8, marker='o', ms=3.5, label=name)
ax.set_yscale('log'); style(ax)
ax.set_xlabel('number of atoms N (median per 10-atom bin, plotted at bin mean N)')
ax.set_ylabel(f'projected Hessian MAE [{U}]')
ax.set_title('Error vs molecule size, clean set (n = 659); shading = interquartile range')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
save(fig, 'fig2_error_vs_size')

# ---------------- Figure 3 / Table B: retraining ablation ----------------
pm = pd.read_csv(P / 'projected-analysis' / 'per-molecule.csv')
orig = {k: pm[pm.model == k].reset_index(drop=True) for k in ('5A_5A', '5A_all', '12A_12A')}
for v in orig.values(): assert len(v) == 710
A2 = lambda n: load(P / 'ablations2' / n / 'metrics.csv')
ABL = [('5 / 5', orig['5A_5A']), ('5 / 12', A2('b5h12')), ('12 / 5', A2('b12h5')), ('12 / 12', orig['12A_12A']),
       ('5 / 5 + envelope', A2('b5h5-env')), ('12 / 12 + envelope', A2('b12h12-env')),
       ('5 / 5, Hessian RBF 0–5 Å', A2('b5h5-rbf5')), ('5 / full', orig['5A_all'])]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ypos = np.arange(len(ABL))[::-1]
abl_rows = []
for yy, (name, df) in zip(ypos, ABL):
    d = clean(df)
    mean, med = d[PH].mean(), d[PH].median()
    ax.plot([min(mean, med), max(mean, med)], [yy, yy], color='0.75', lw=1, zorder=1)
    ax.scatter(mean, yy, marker='D', color=OI['blue'], s=28, zorder=3, label='mean' if yy == ypos[0] else None)
    ax.scatter(med, yy, marker='o', color=OI['orange'], s=28, zorder=3, label='median' if yy == ypos[0] else None)
    abl_rows.append((name, mean, med, d[EV].mean(), d[EV].median(), d[COS].mean(), d[COS].median()))
ax.set_yticks(ypos); ax.set_yticklabels([n for n, _ in ABL])
ax.set_xscale('log'); style(ax)
ax.set_xlabel(f'projected Hessian MAE [{U}], clean set (n = 659)')
ax.set_ylabel('backbone / Hessian graph cutoff')
ax.set_title('Retraining ablation (100 epochs each), evaluated with its training graph')
ax.legend(loc='lower right', frameon=True, framealpha=0.9)
save(fig, 'fig3_ablation')

# ---------------- Figure 4: outlier diagnostic ----------------
d = G[('hip_v3', '7')]
isflat, isbad = d.sample_name.isin(FLAT), d.sample_name.isin(BAD)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.scatter(d.natoms[~isbad], d[PH][~isbad], s=9, color='0.65', alpha=0.7, lw=0, label=f'clean (n = {(~isbad).sum()})')
o = isbad & ~isflat
ax.scatter(d.natoms[o], d[PH][o], s=22, marker='^', facecolor='none', edgecolor=OI['red'], lw=1,
           label=f'other reference outliers (n = {o.sum()})')
ax.scatter(d.natoms[isflat], d[PH][isflat], s=34, marker='X', color=OI['blue'], lw=0.3,
           label=f'flat 2D drawings, z = 0 (n = {isflat.sum()})')
ax.set_yscale('log'); style(ax)
ax.set_xlabel('number of atoms N'); ax.set_ylabel(f'projected Hessian MAE [{U}]')
ax.set_title('hip_v3, 7 Å Hessian graph at inference: per-molecule error (all 710)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, columnspacing=1.0)
save(fig, 'fig4_outliers')

# ---------------- Tables ----------------
f = lambda v: f'{v:.3g}'
L = ['# Tables for email reply: HIP on PubChem 30–100 atoms', '',
     'Units. `projected_hessian_mae_ev_a2_amu` and `eckart_eigval_mae_ev_a2` (despite its column name) are both '
     'computed on the **mass-weighted** Hessian, so both are in eV/(Å² amu). Definitions in '
     '`scripts/eval_projected_pubchem.py`: M^-1/2 H M^-1/2 is projected onto the reduced Eckart (vibrational) '
     'basis Q (translations/rotations removed); the eigenvalue MAE is mean_i |λ_i^pred − λ_i^ref| over all 3N−6 '
     'Eckart eigenvalues; projected Hessian MAE is the elementwise MAE of QᵀQ (H_pred − H_ref)_mw QᵀQ over all (3N)² '
     'entries; |cos v1| is the overlap of the lowest Eckart eigenvectors. Clean set: 710 minus 51 reference outliers '
     '(n = 659). Non-flat set: 710 minus the 8 z = 0 files (n = 702). All models use the backbone graph they were '
     'trained with (5 Å, or 12 Å for abl 12/12) with max_neighbors 1000; only the Hessian graph cutoff at inference varies.', '']
L += ['## A. Inference Hessian-graph cutoff sweep (released checkpoints)', '']
for m in ('hip_v2', 'hip_v3'):
    L += [f'### {m} ({GRID_MODELS[m][0].split("; ")[1].rstrip(")")})', '',
          '| Hessian cutoff | median proj. Hessian MAE (clean) | mean proj. Hessian MAE (clean) | median eigval MAE (clean) '
          '| median abs cos v1 (clean) | mean eigval MAE (non-flat, n = 702) |', '|---|---|---|---|---|---|']
    for c in CUTS:
        g = G[(m, c)]; cl = clean(g); nf = nonflat(g)
        assert len(cl) == 659 and len(nf) == 702
        L.append(f'| {"fully connected" if c == "full" else c + " Å"} | {f(cl[PH].median())} | {f(cl[PH].mean())} | '
                 f'{f(cl[EV].median())} | {cl[COS].median():.3f} | {f(nf[EV].mean())} |')
    L.append('')
L += ['For comparison, the external researcher reports mean Eckart eigenvalue MAE on 702 molecules of 1.8234 '
      '(default 100 Å cutoff) and 0.0131 (7.5 Å + 32 sorted neighbours). Units/normalisation of his metric should '
      'be checked against ours before comparing directly.', '']
L += ['## B. Retraining ablation (backbone / Hessian graph, Å), each evaluated with its training graph, clean set (n = 659)', '',
      '| backbone / Hessian | mean proj. Hessian MAE | median proj. Hessian MAE | mean eigval MAE | median eigval MAE '
      '| mean abs cos v1 | median abs cos v1 |', '|---|---|---|---|---|---|---|']
for name, a, b, c, d_, e, g in abl_rows:
    L.append(f'| {name} | {f(a)} | {f(b)} | {f(c)} | {f(d_)} | {e:.3f} | {g:.3f} |')
L += ['', 'All variants: 100 epochs with limit_train_batches=0.1 on T1x (269,500 steps), otherwise identical settings (see README). "envelope" = smooth cutoff envelope on '
      'Hessian edges; "Hessian RBF 0–5 Å" = Hessian-edge distance expansion restricted to 0–5 Å.', '']
txt = (P / 't1x-max-pair-distance.out').read_text()
num = lambda pat: re.search(pat, txt).group(1)
pct = re.search(r'percentiles 50/90/99/99.9: \[([^\]]+)\]', txt).group(1).split(',')
L += ['## C. T1x training set: pair-distance statistics (all training geometries)', '',
      '| quantity | value |', '|---|---|',
      f'| number of geometries | {int(num(r"samples (\d+)")):,} |',
      f'| atoms per molecule (max / mean) | {num(r"natoms max (\d+)")} / {num(r"mean ([\d.]+)")} |',
      f'| largest pair distance in the set | {num(r"MAX pair distance ([\d.]+)")} Å |',
      '| per-molecule max pair distance, percentiles 50 / 90 / 99 / 99.9 | ' + ' / '.join(p.strip() for p in pct) + ' Å |']
for thr in ('5', '7.5', '10', '12', '14.36'):
    v = float(num(r'any pair > ' + re.escape(thr) + r' A: ([\d.]+)'))
    L.append(f'| fraction of molecules with any pair > {thr} Å | {v:.5f} ({100 * v:.3g} %) |')
L += ['', 'So a fully connected (or 100 Å) Hessian graph at inference produces edges far beyond any distance '
      'seen in training (> 12.21 Å never occurs; > 7.5 Å occurs in 1.3 % of molecules).', '']
n_of = G[('hip_v3', '7')].set_index('sample_name').natoms
L += ['## D. The 8 flat (z = 0) PubChem files', '', '| sample_name | natoms | in 51-outlier list |', '|---|---|---|']
for s in FLAT:
    L.append(f'| {s} | {n_of[s]} | {"yes" if s in BAD else "no"} |')
(OUT / 'tables.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L[:60]))
