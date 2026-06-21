#!/usr/bin/env python
"""Method-comparison diagnostics on the relaxed glycine PT surface.

Unlike ``plot_glycine_pt_dft_cv_diagnostics.py`` (which assumes a regular
``q_nh x q_oh`` rectangle), the relaxed scan is laid out on a regular grid in the
difference coordinates ``s = q_nh - q_oh`` and ``sigma = q_nh + q_oh``. Heavy-atom
relaxation makes ``(q_nh, q_oh)`` irregular, so every panel here pivots on
``(s, sigma)`` instead.

Compares DFT (ORCA), HIP (predicted Hessian) and eqv2 (autograd Hessian) on the
identical relaxed geometries:

* relaxed DFT energy surface,
* lowest projected (vibrational) Hessian eigenvalue ``lambda_min``,
* number of negative eigenvalues (spurious-mode speckle for autograd),
* alignment of the softest mode with the proton-transfer direction,
* Hessian mean-absolute error vs DFT,
* force mean-absolute error vs DFT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT), str(ROOT / "plotting")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from scripts.cache_glycine_pt_orca_vibrations import (  # noqa: E402
    NEG_EIGVAL_THRESHOLD,
    mode_alignment,
    vibrational_eigh,
)

HARTREE_TO_KCAL = 627.5094740631
EV_ANG2 = r"eV Å$^{-2}$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=ROOT / "runs" / "glycine_pt_scan_relaxed")
    parser.add_argument("--vib-cache", type=Path, default=None)
    parser.add_argument("--hip-arrays", type=Path, default=None)
    parser.add_argument("--eqv2-arrays", type=Path, default=None)
    parser.add_argument("--stationary-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--lambda-clip-pct", type=float, default=2.0,
                        help="Percentile clip for the lambda_min colour scale (clipped variant).")
    return parser.parse_args()


STATIONARY_STYLE = {
    "reactant": dict(marker="o", label="reactant", color="#1b9e77"),
    "product": dict(marker="s", label="product", color="#7570b3"),
    "ts": dict(marker="*", label="TS (1 imag.)", color="#d95f02"),
}


def overlay_stationary(ax, pts):
    if not pts:
        return
    for key, style in STATIONARY_STYLE.items():
        if key not in pts:
            continue
        ax.scatter(
            [pts[key]["s"]], [pts[key]["sigma"]],
            marker=style["marker"], s=170 if style["marker"] == "*" else 90,
            facecolor=style["color"], edgecolor="white", linewidth=1.1,
            zorder=6, label=style["label"],
        )


def make_grid(s: np.ndarray, sigma: np.ndarray, values: np.ndarray):
    """Pivot node values onto the regular (s, sigma) grid; gaps become masked."""
    s_r = np.round(np.asarray(s, dtype=float), 6)
    sig_r = np.round(np.asarray(sigma, dtype=float), 6)
    xs = np.unique(s_r)
    ys = np.unique(sig_r)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    grid = np.full((ys.size, xs.size), np.nan, dtype=float)
    for sv, gv, val in zip(s_r, sig_r, np.asarray(values, dtype=float)):
        grid[yi[gv], xi[sv]] = val
    return xs, ys, np.ma.masked_invalid(grid)


def draw_panel(ax, xs, ys, grid, *, cmap, norm=None, vmin=None, vmax=None, title=None):
    mesh = ax.pcolormesh(xs, ys, grid, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_xlabel(r"$s = q_{NH} - q_{OH}$  (Å)")
    ax.set_ylabel(r"$\sigma = q_{NH} + q_{OH}$  (Å)")
    if title:
        ax.set_title(title)
    ax.axvline(0.0, color="0.35", lw=0.6, ls="--", alpha=0.7)
    return mesh


def vib_metrics(hessians_ev, coords, masses, pt_dirs):
    """lambda_min, n_negative, |softest-mode alignment with PT| per node."""
    n = hessians_ev.shape[0]
    lam = np.empty(n)
    nneg = np.empty(n, dtype=int)
    align = np.empty(n)
    for i in range(n):
        evals, modes = vibrational_eigh(hessians_ev[i], coords[i], masses)
        lam[i] = evals[0]
        nneg[i] = int((evals < -NEG_EIGVAL_THRESHOLD).sum())
        align[i] = abs(mode_alignment(modes[:, 0], pt_dirs[i], masses))
    return lam, nneg, align


def main() -> None:
    args = parse_args()
    scan = args.scan_dir
    vib_path = args.vib_cache or scan / "orca_vib_cache.npz"
    hip_path = args.hip_arrays or scan / "hip_v2_arrays.npz"
    eqv2_path = args.eqv2_arrays or scan / "eqv2_autograd_arrays.npz"
    out_dir = args.output_dir or scan / "plots_relaxed"
    out_dir.mkdir(parents=True, exist_ok=True)

    stat_path = args.stationary_json or scan / "stationary_points.json"
    stationary = json.loads(stat_path.read_text()) if stat_path.exists() else {}

    dft = np.load(vib_path)
    hip = np.load(hip_path)
    eqv2 = np.load(eqv2_path)

    # All three pipelines sort by grid_id; verify identical ordering.
    gid = dft["grid_id"]
    for other, name in ((hip, "hip"), (eqv2, "eqv2")):
        if not np.array_equal(gid, other["grid_id"]):
            raise ValueError(f"grid_id ordering mismatch between DFT and {name}")

    s = dft["s"]
    sigma = dft["sigma"]
    masses = dft["masses_amu"]
    coords = dft["coords_angstrom"]
    pt_dirs = dft["pt_direction"]

    # --- DFT metrics (precomputed in the cache) ---
    lam_dft = dft["vib_evals_ev_ang2_amu"][:, 0]
    nneg_dft = dft["n_negative"]
    align_dft = dft["unstable_mode_pt_abs_alignment"]
    H_dft = dft["hessian_ev_ang2"]
    F_dft = dft["forces_ev_ang"]
    e_rel = (dft["energy_hartree_engrad"] - dft["energy_hartree_engrad"].min()) * HARTREE_TO_KCAL

    # --- HIP / eqv2 metrics (recomputed on identical geometries) ---
    lam_hip, nneg_hip, align_hip = vib_metrics(hip["hessians_cartesian"], coords, masses, pt_dirs)
    lam_eqv2, nneg_eqv2, align_eqv2 = vib_metrics(eqv2["hessians_cartesian"], coords, masses, pt_dirs)

    hmae_hip = np.abs(hip["hessians_cartesian"] - H_dft).mean(axis=(1, 2))
    hmae_eqv2 = np.abs(eqv2["hessians_cartesian"] - H_dft).mean(axis=(1, 2))
    fmae_hip = np.abs(hip["forces"] - F_dft).mean(axis=(1, 2))
    fmae_eqv2 = np.abs(eqv2["forces"] - F_dft).mean(axis=(1, 2))

    def grids(values):
        return make_grid(s, sigma, values)

    # === Figure 1: relaxed DFT energy surface ===
    xs, ys, z = grids(e_rel)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    mesh = draw_panel(ax, xs, ys, z, cmap="viridis")
    levels = np.arange(0, np.nanmax(z.filled(np.nan)) + 10, 10.0)
    ax.contour(xs, ys, z, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
    overlay_stationary(ax, stationary)
    ax.legend(loc="upper center", ncol=3, fontsize=8, framealpha=0.85,
              frameon=True, edgecolor="none")
    fig.draw_without_rendering()
    cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0])
    fig.colorbar(mesh, cax=cax, label=r"$E - E_{\min}$  (kcal/mol)")
    fig.tight_layout(pad=0.01)
    fname = out_dir / "relaxed_energy_surface.png"
    fig.savefig(fname, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    print(f"Wrote {fname}")
    plt.close(fig)

    def panel_row(metric_by_method, labels, fname, *, cbar_label, cmap, diverging=False,
                  discrete=False, vmin=None, vmax=None, clip_pct=None,
                  mark_stationary=False):
        gridded = [grids(m) for m in metric_by_method]
        finite = np.concatenate([g[2].compressed() for g in gridded])
        if clip_pct is not None:
            lo, hi = np.percentile(finite, [clip_pct, 100 - clip_pct])
            if vmin is None:
                vmin = float(lo)
            if vmax is None:
                vmax = float(hi)
        if vmin is None:
            vmin = float(finite.min())
        if vmax is None:
            vmax = float(finite.max())
        norm = None
        if discrete:
            hi = int(max(1, np.ceil(vmax)))
            bounds = np.arange(-0.5, hi + 1.5, 1.0)
            cmap_obj = plt.get_cmap(cmap, len(bounds) - 1)
            norm = BoundaryNorm(bounds, cmap_obj.N)
            cmap = cmap_obj
            vmin = vmax = None
        elif diverging:
            span = max(abs(vmin), abs(vmax))
            norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
            vmin = vmax = None
        n = len(metric_by_method)
        fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), squeeze=False)
        axes = axes[0]
        mesh = None
        for ax, (xs_, ys_, z_), label in zip(axes, gridded, labels):
            mesh = draw_panel(ax, xs_, ys_, z_, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, title=label)
            if mark_stationary:
                overlay_stationary(ax, stationary)
        if mark_stationary:
            axes[0].legend(loc="upper center", ncol=3, fontsize=7, framealpha=0.85,
                           frameon=True, edgecolor="none")
        fig.draw_without_rendering()
        cax = axes[-1].inset_axes([1.05, 0.0, 0.05, 1.0])
        cbar = fig.colorbar(mesh, cax=cax, label=cbar_label)
        if discrete:
            cbar.set_ticks(np.arange(0, int(np.ceil(vmax if vmax else 0)) + 1)) if False else None
        fig.tight_layout(pad=0.01)
        fname = out_dir / fname
        print(f"Wrote {fname}")
        fig.savefig(fname, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)

    # === Figure 2: lambda_min (3 panels), full diverging scale ===
    panel_row(
        [lam_dft, lam_hip, lam_eqv2], ["DFT", "HIP (predicted)", "eqv2 (autograd)"],
        "relaxed_lambda_min.png",
        cbar_label=r"$\lambda_{\min}$ (eV Å$^{-2}$ amu$^{-1}$)",
        cmap="RdBu_r", diverging=True, mark_stationary=True,
    )

    # === Figure 2b: lambda_min, percentile-clipped to reveal DFT/HIP structure ===
    panel_row(
        [lam_dft, lam_hip, lam_eqv2], ["DFT", "HIP (predicted)", "eqv2 (autograd)"],
        "relaxed_lambda_min_clipped.png",
        cbar_label=r"$\lambda_{\min}$ (eV Å$^{-2}$ amu$^{-1}$, clipped)",
        cmap="RdBu_r", diverging=True, clip_pct=args.lambda_clip_pct,
        mark_stationary=True,
    )

    # === Figure 3: number of negative eigenvalues (3 panels) ===
    nmax = int(max(nneg_dft.max(), nneg_hip.max(), nneg_eqv2.max()))
    panel_row(
        [nneg_dft, nneg_hip, nneg_eqv2], ["DFT", "HIP (predicted)", "eqv2 (autograd)"],
        "relaxed_n_negative.png",
        cbar_label="# negative eigenvalues",
        cmap="magma_r", discrete=True, vmin=0, vmax=max(1, nmax),
        mark_stationary=True,
    )

    # === Figure 4: softest-mode / PT-direction alignment (3 panels) ===
    panel_row(
        [align_dft, align_hip, align_eqv2], ["DFT", "HIP (predicted)", "eqv2 (autograd)"],
        "relaxed_mode_alignment.png",
        cbar_label=r"$|\langle$ softest mode $|\, q_{NH}-q_{OH}\rangle|$",
        cmap="viridis", vmin=0.0, vmax=1.0,
    )

    # === Figure 5: Hessian MAE vs DFT (2 panels) ===
    panel_row(
        [hmae_hip, hmae_eqv2], ["HIP (predicted)", "eqv2 (autograd)"],
        "relaxed_hessian_mae.png",
        cbar_label=rf"Hessian MAE vs DFT ({EV_ANG2})",
        cmap="inferno", mark_stationary=True,
    )

    # === Figure 6: force MAE vs DFT (2 panels) ===
    panel_row(
        [fmae_hip, fmae_eqv2], ["HIP (predicted)", "eqv2 (autograd)"],
        "relaxed_force_mae.png",
        cbar_label=r"Force MAE vs DFT (eV Å$^{-1}$)",
        cmap="inferno", vmin=0.0, vmax=0.10,
    )

    # --- console summary (median over the surface) ---
    def med(x):
        return float(np.median(x))

    print("Wrote plots to", out_dir)
    print(f"{'metric':<28}{'HIP':>14}{'eqv2(AD)':>14}")
    print(f"{'Hessian MAE vs DFT':<28}{med(hmae_hip):>14.4f}{med(hmae_eqv2):>14.4f}")
    print(f"{'Force MAE vs DFT':<28}{med(fmae_hip):>14.4f}{med(fmae_eqv2):>14.4f}")
    print(f"{'mean # neg eigs':<28}{nneg_hip.mean():>14.3f}{nneg_eqv2.mean():>14.3f}")
    print(f"{'(DFT mean # neg eigs)':<28}{nneg_dft.mean():>14.3f}")
    print(f"{'nodes with >1 neg eig':<28}{int((nneg_hip>1).sum()):>14d}{int((nneg_eqv2>1).sum()):>14d}")
    print(f"{'(DFT nodes >1 neg eig)':<28}{int((nneg_dft>1).sum()):>14d}")


if __name__ == "__main__":
    main()
