#!/usr/bin/env python
"""Dense 1D view of why EQV2 autograd (AD) Hessians fail along the glycine PT path.

Single model, single comparison: EQV2 AD Hessian vs DFT, along the antisymmetric proton
coordinate xi = q_NH - q_OH. Consumes ``path_arrays.npz`` from the enhanced
``scripts/glycine_pt_path_scan.py`` (full EQV2 Hessians + vib eigenvalues + n_negative +
asymmetry) and, when present, the dense DFT reference ``orca_vib_cache.npz`` on the same
grid.

Works in two modes:
  - MLIP only (no DFT yet): AD eigenvalue traces, n_negative, and asymmetry along xi.
  - With DFT: adds AD-vs-DFT eigenvalue errors, neg-count vs truth, and the Frobenius
    error decomposition  ||H_AD-H*||^2 = ||asym(H_AD)||^2 + ||sym(H_AD)-H*||^2.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from plot_style import ACCENT_COLOR, AD_COLOR, DFT_COLOR, HIP_COLOR, LINE_WIDTH, THIN_LINE_WIDTH, finish_axis

EQV2_COLOR = AD_COLOR
NEG_THRESHOLD = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-dir", type=Path, default=Path("runs/glycine_pt_path_dft"))
    parser.add_argument("--path-arrays", type=Path, default=None)
    parser.add_argument("--orca-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-modes", type=int, default=4, help="lowest vib modes to trace")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def fro(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x.reshape(x.shape[0], -1), axis=1)


def load_dft(cache_path: Path) -> dict[str, np.ndarray] | None:
    if not cache_path.exists():
        return None
    c = np.load(cache_path, allow_pickle=True)
    xi = np.asarray(c["q_nh"], dtype=float) - np.asarray(c["q_oh"], dtype=float)
    order = np.argsort(xi)
    out = {
        "xi": xi[order],
        "evals": np.asarray(c["vib_evals_ev_ang2_amu"], dtype=float)[order],
        "n_negative": np.asarray(c["n_negative"], dtype=int)[order],
        "hessian": 0.5 * (np.asarray(c["hessian_ev_ang2"], dtype=float)
                          + np.swapaxes(np.asarray(c["hessian_ev_ang2"], dtype=float), -1, -2))[order],
    }
    return out


def main() -> None:
    args = parse_args()
    path_dir = args.path_dir
    arrays_path = args.path_arrays or path_dir / "path_arrays.npz"
    cache_path = args.orca_cache or path_dir / "orca_vib_cache.npz"
    output_dir = args.output_dir or path_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(arrays_path)
    xi = np.asarray(data["xi"], dtype=float)
    order = np.argsort(xi)

    def col(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=float)[order]

    xi = xi[order]
    ad_evals = col("eqv2_evals")
    ad_nneg = col("eqv2_n_negative")
    ad_asym = col("eqv2_asym")
    have_full_h = "eqv2_hessian_cartesian" in data.files
    if have_full_h:
        h_raw = np.asarray(data["eqv2_hessian_cartesian"], dtype=float)[order]

    dft = load_dft(cache_path)
    aligned = dft is not None and dft["xi"].shape == xi.shape and np.allclose(dft["xi"], xi, atol=1e-6)
    if dft is not None and not aligned:
        print("[warn] DFT grid does not match MLIP grid 1:1; using DFT as overlay only.")

    k = int(min(args.n_modes, ad_evals.shape[1]))
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.4))
    xlabel = r"$\xi = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]"

    # A: lowest-k eigenvalue traces, AD vs DFT
    ax = axes[0, 0]
    for m in range(k):
        sns.lineplot(x=xi, y=ad_evals[:, m], ax=ax, color=EQV2_COLOR, lw=LINE_WIDTH, alpha=0.85, label="EQV2 AD" if m == 0 else None)
    if dft is not None:
        for m in range(min(k, dft["evals"].shape[1])):
            sns.lineplot(x=dft["xi"], y=dft["evals"][:, m], ax=ax, color=DFT_COLOR, lw=LINE_WIDTH, ls="--", label="DFT" if m == 0 else None)
    ax.axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    ax.set_title(f"A  lowest {k} vibrational eigenvalues")
    ax.set_ylabel(r"$\lambda$ [eV $\AA^{-2}$ amu$^{-1}$]")
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8, frameon=True, edgecolor="none")

    # B: negative-mode count along the path
    ax = axes[0, 1]
    sns.lineplot(x=xi, y=ad_nneg, ax=ax, color=EQV2_COLOR, lw=LINE_WIDTH, label="EQV2 AD")
    if dft is not None:
        sns.lineplot(x=dft["xi"], y=dft["n_negative"], ax=ax, color=DFT_COLOR, lw=LINE_WIDTH, ls="--", label="DFT")
    ax.set_title("B  number of negative modes")
    ax.set_ylabel(r"$n_\mathrm{neg}$")
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8, frameon=True, edgecolor="none")

    # C: AD asymmetry (non-conservativeness)
    ax = axes[0, 2]
    sns.lineplot(x=xi, y=ad_asym, ax=ax, color=EQV2_COLOR, lw=LINE_WIDTH)
    ax.set_title(r"C  AD asymmetry $\|H_{AD}-H_{AD}^\top\|/\|H_{AD}\|$")
    ax.set_ylabel("fraction (non-conservativeness)")
    ax.set_xlabel(xlabel)

    # D: eigenvalue error vs DFT (lowest k)
    ax = axes[1, 0]
    if aligned:
        kk = min(k, dft["evals"].shape[1])
        for m in range(kk):
            sns.lineplot(x=xi, y=ad_evals[:, m] - dft["evals"][:, m], ax=ax, lw=THIN_LINE_WIDTH, label=f"mode {m}")
        ax.axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
        ax.set_title("D  AD eigenvalue error vs DFT")
        ax.set_ylabel(r"$\lambda_{AD}-\lambda^{*}$")
        ax.legend(fontsize=7, ncol=2, frameon=True, edgecolor="none")
    else:
        ax.text(0.5, 0.5, "needs DFT on the same grid", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("D  AD eigenvalue error vs DFT")
    ax.set_xlabel(xlabel)

    # E: neg-count agreement (AD - DFT)
    ax = axes[1, 1]
    if aligned:
        delta = ad_nneg - dft["n_negative"]
        sns.lineplot(x=xi, y=delta, ax=ax, color=EQV2_COLOR, lw=LINE_WIDTH)
        ax.axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
        frac = float((delta == 0).mean())
        ax.set_title(f"E  neg-count error (correct {100*frac:.0f}%)")
        ax.set_ylabel(r"$n_\mathrm{neg}^{AD}-n_\mathrm{neg}^{*}$")
    else:
        ax.text(0.5, 0.5, "needs DFT on the same grid", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("E  neg-count error vs DFT")
    ax.set_xlabel(xlabel)

    # F: Frobenius error decomposition vs DFT
    ax = axes[1, 2]
    if aligned and have_full_h:
        h_star = dft["hessian"]
        asym = 0.5 * (h_raw - np.swapaxes(h_raw, -1, -2))   # raw AD antisymmetric part
        sym = 0.5 * (h_raw + np.swapaxes(h_raw, -1, -2))    # raw AD symmetric part
        norm_star = np.maximum(fro(h_star), 1e-12)
        total_rel = fro(h_raw - h_star) / norm_star
        sym_rel = fro(sym - h_star) / norm_star
        asym_rel = fro(asym) / norm_star
        sns.lineplot(x=xi, y=total_rel, ax=ax, color=EQV2_COLOR, lw=LINE_WIDTH, label=r"total $\|H_{AD}-H^*\|/\|H^*\|$")
        sns.lineplot(x=xi, y=sym_rel, ax=ax, color=HIP_COLOR, lw=LINE_WIDTH, label=r"sym part (unsupervised)")
        sns.lineplot(x=xi, y=asym_rel, ax=ax, color=ACCENT_COLOR, lw=LINE_WIDTH, label=r"asym part (non-cons.)")
        ax.set_title("F  AD vs DFT error decomposition")
        ax.set_ylabel("relative Frobenius")
        ax.legend(fontsize=7, frameon=True, edgecolor="none")
    else:
        ax.text(0.5, 0.5, "needs DFT + full AD Hessians", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("F  AD vs DFT error decomposition")
    ax.set_xlabel(xlabel)

    for ax in axes.ravel():
        finish_axis(ax)

    status = "with DFT" if aligned else "MLIP only (no DFT yet)"
    fig.suptitle(f"EQV2 autograd Hessian failure along the glycine PT path ({status})", fontsize=13)
    fig.tight_layout(pad=0.01)
    out_path = output_dir / "glycine_pt_path_ad_failure.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
