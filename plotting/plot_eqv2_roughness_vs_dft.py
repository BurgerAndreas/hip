#!/usr/bin/env python
"""Robustness (distributions) + causation (roughness vs autograd-Hessian DFT error) for eqv2.

Reads the per-sample table from scripts/eqv2_roughness_vs_dft.py and renders:
  1. eqv2_rough_distributions.png  - histograms of the roughness and accuracy metrics.
  2. eqv2_rough_vs_dft_scatter.png  - roughness (fd_plateau, asym) vs autograd-Hessian error
                                      vs DFT (hess_rel_err, vib_eigval_mae), with Spearman rho.
  3. eqv2_rough_controls.png        - does roughness predict Hessian error beyond general model
                                      inaccuracy? (force error controls).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_style import AD_COLOR, HIP_COLOR, finish_axis


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _spearman(a, b) -> float:
    ra, rb = pd.Series(np.asarray(a)).rank(), pd.Series(np.asarray(b)).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def _scatter(ax, x, y, c, label_x, label_y):
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=28, alpha=0.8, edgecolor="k", linewidth=0.3)
    rho = _spearman(x, y)
    # robust trend via rank-space linear fit, drawn in value space order
    order = np.argsort(x)
    sns.lineplot(
        x=np.asarray(x)[order],
        y=np.poly1d(np.polyfit(x, y, 1))(np.asarray(x)[order]),
        ax=ax,
        color=HIP_COLOR,
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
    )
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(rf"Spearman $\rho$ = {rho:+.2f}", fontsize=10)
    finish_axis(ax)
    return sc


def fig_distributions(df, out):
    metrics = [
        ("fd_plateau", "FD self-inconsistency (roughness)"),
        ("asym", r"non-conservativeness $\|H-H^T\|/\|H\|$"),
        ("hess_rel_err", "autograd-H vs DFT  (rel. Frobenius)"),
        ("eigval_mae", "eigenvalue MAE vs DFT [eV/$\\AA^2$]"),
        ("vib_eigval_mae", "vib. eigenvalue MAE vs DFT"),
        ("force_rel_err", "force vs DFT (rel.)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (col, lab) in zip(axes.ravel(), metrics):
        sns.histplot(df, x=col, bins=24, ax=ax, color=AD_COLOR, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(df[col].median(), color=HIP_COLOR, ls="--", lw=1.4, label=f"median={df[col].median():.3g}")
        ax.set_xlabel(lab)
        ax.set_ylabel("count")
        ax.legend(fontsize=8, frameon=True, edgecolor="none")
        finish_axis(ax)
    fig.suptitle(
        f"eqv2 roughness & autograd-Hessian accuracy — {len(df)} HORM samples", fontsize=12
    )
    fig.tight_layout(pad=0.01)
    p = out / "eqv2_rough_distributions.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_scatter(df, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    combos = [
        ("fd_plateau", "hess_rel_err", "FD self-inconsistency (roughness)", "autograd-H vs DFT (rel.)"),
        ("fd_plateau", "vib_eigval_mae", "FD self-inconsistency (roughness)", "vib. eigval MAE vs DFT"),
        ("asym", "hess_rel_err", r"non-conservativeness $\|H-H^T\|/\|H\|$", "autograd-H vs DFT (rel.)"),
        ("asym", "vib_eigval_mae", r"non-conservativeness $\|H-H^T\|/\|H\|$", "vib. eigval MAE vs DFT"),
    ]
    sc = None
    for ax, (xc, yc, xl, yl) in zip(axes.ravel(), combos):
        sc = _scatter(ax, df[xc].values, df[yc].values, df["natoms"].values, xl, yl)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("n atoms")
    fig.suptitle(
        f"Does eqv2 force-field roughness predict autograd-Hessian error vs DFT? "
        f"({len(df)} HORM samples)",
        fontsize=12,
    )
    p = out / "eqv2_rough_vs_dft_scatter.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_controls(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    _scatter(axes[0], df["force_rel_err"].values, df["hess_rel_err"].values,
             df["natoms"].values, "force vs DFT (rel.)", "autograd-H vs DFT (rel.)")
    axes[0].set_title("control: is H-error just overall inaccuracy?\n" + axes[0].get_title(), fontsize=9)
    _scatter(axes[1], df["fd_plateau"].values, df["force_rel_err"].values,
             df["natoms"].values, "FD self-inconsistency (roughness)", "force vs DFT (rel.)")
    axes[1].set_title("roughness vs force accuracy\n" + axes[1].get_title(), fontsize=9)
    # neg-mode classification accuracy as a function of roughness (binned)
    ax = axes[2]
    q = pd.qcut(df["fd_plateau"], q=min(5, df["fd_plateau"].nunique()), duplicates="drop")
    agree = df.groupby(q, observed=True)["neg_match"].mean()
    centers = [iv.mid for iv in agree.index]
    sns.lineplot(x=centers, y=agree.values * 100, ax=ax, marker="o", color=AD_COLOR)
    ax.set_xlabel("FD self-inconsistency (roughness), binned")
    ax.set_ylabel("neg-mode agreement w/ DFT [%]")
    ax.set_ylim(0, 105)
    ax.set_title("does roughness break TS/min classification?", fontsize=9)
    finish_axis(ax)
    fig.suptitle(f"eqv2 controls — {len(df)} HORM samples", fontsize=12)
    fig.tight_layout(pad=0.01)
    p = out / "eqv2_rough_controls.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=_project_root() / "runs" / "eqv2_roughness_vs_dft" / "eqv2_sample100.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=_project_root() / "runs" / "eqv2_roughness_vs_dft" / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    written = [fig_distributions(df, args.out_dir), fig_scatter(df, args.out_dir),
               fig_controls(df, args.out_dir)]
    print(f"Loaded {len(df)} samples from {args.csv}")
    print("Spearman correlations (roughness vs accuracy):")
    for rough in ["fd_plateau", "asym"]:
        for acc in ["hess_rel_err", "eigval_mae", "vib_eigval_mae", "force_rel_err"]:
            print(f"  rho({rough:11s}, {acc:14s}) = {_spearman(df[rough], df[acc]):+.3f}")
    print(f"neg-mode agreement with DFT: {df['neg_match'].mean() * 100:.1f}%")
    print("Wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
