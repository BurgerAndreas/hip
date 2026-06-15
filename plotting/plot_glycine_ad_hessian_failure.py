#!/usr/bin/env python
"""Why EQV2 autograd (AD) Hessians fail, on the existing glycine 2D scan.

Single model, single comparison: EQV2's AD Hessian ``H_AD = -dF/dx`` (the Jacobian of
EQV2's *own* direct forces) versus the DFT Hessian ``H*``. EQV2's forces are accurate
(shown), yet the Hessian obtained by differentiating them is not.

Because ``H*`` is symmetric, the AD error decomposes exactly into two physically
distinct failures::

    ||H_AD - H*||^2 = ||asym(H_AD)||^2 + ||sym(H_AD) - H*||^2
                      \\__ non-conservativeness __/   \\__ unsupervised-Jacobian __/

- asym(H_AD) = 0.5 (H_AD - H_AD^T): exists only because EQV2's forces are
  non-conservative; a true Hessian has none.
- sym(H_AD) - H*: the symmetric part is still wrong because the force Jacobian was
  never supervised in training.

Inputs (already aligned on the 36-grid, no new compute):
  - EQV2 AD arrays:  runs/glycine_pt_eqv2_autograd_n36/eqv2_autograd_arrays.npz
  - EQV2 predictions (grid_id order): .../eqv2_autograd_predictions.csv
  - DFT cache:       runs/glycine_pt_scan_n36/orca_vib_cache.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


NEG_THRESHOLD = 1e-6
MASS_BY_Z = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 15: 30.974, 16: 32.065, 17: 35.453}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ad-arrays", type=Path, default=Path("runs/glycine_pt_eqv2_autograd_n36/eqv2_autograd_arrays.npz"))
    parser.add_argument("--ad-predictions", type=Path, default=Path("runs/glycine_pt_eqv2_autograd_n36/eqv2_autograd_predictions.csv"))
    parser.add_argument("--orca-cache", type=Path, default=Path("runs/glycine_pt_scan_n36/orca_vib_cache.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/glycine_pt_path/plots"))
    parser.add_argument("--n-eigs", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def symmetrize(h: np.ndarray) -> np.ndarray:
    return 0.5 * (h + np.swapaxes(h, -1, -2))


def eckart_generators(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    masses = np.asarray(masses, dtype=float).reshape(-1)
    n_atoms = xyz.shape[0]
    sqrt_m = np.sqrt(masses)
    sqrt_m3 = np.repeat(sqrt_m, 3)
    com = (xyz * masses[:, None]).sum(axis=0) / masses.sum()
    rel = xyz - com[None, :]
    cols = []
    for axis in np.eye(3):
        col = sqrt_m3 * np.tile(axis, n_atoms)
        cols.append(col / max(float(np.linalg.norm(col)), 1e-12))
    rx, ry, rz = rel[:, 0], rel[:, 1], rel[:, 2]
    for rot in (
        np.stack([np.zeros_like(rx), -rz, ry], axis=1),
        np.stack([rz, np.zeros_like(ry), -rx], axis=1),
        np.stack([-ry, rx, np.zeros_like(rz)], axis=1),
    ):
        col = (rot * sqrt_m[:, None]).reshape(-1)
        norm = np.linalg.norm(col)
        if norm > 1e-12:
            cols.append(col / norm)
    return np.stack(cols, axis=1)


def vibrational_evals(hessian_ev_ang2: np.ndarray, coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    n_atoms = masses.size
    hessian = symmetrize(np.asarray(hessian_ev_ang2, dtype=float).reshape(3 * n_atoms, 3 * n_atoms))
    m3 = np.repeat(masses, 3)
    hessian_mw = hessian / np.sqrt(np.outer(m3, m3))
    generators = eckart_generators(coords, masses)
    q, r = np.linalg.qr(generators, mode="reduced")
    rank = max(int((np.abs(np.diag(r)) > 1e-6).sum()), 1)
    u, _, _ = np.linalg.svd(q[:, :rank], full_matrices=True)
    q_vib = u[:, rank:]
    reduced = symmetrize(q_vib.T @ hessian_mw @ q_vib)
    return np.linalg.eigvalsh(reduced)


def to_grid(q_nh: np.ndarray, q_oh: np.ndarray, values: np.ndarray):
    df = pd.DataFrame({"q_nh": q_nh, "q_oh": q_oh, "v": values})
    pivot = df.pivot_table(index="q_oh", columns="q_nh", values="v").sort_index()
    return pivot.columns.to_numpy(float), pivot.index.to_numpy(float), pivot.to_numpy(float)


def heatmap(ax, q_nh, q_oh, values, title, cbar_label, cmap="viridis", vmin=None, vmax=None, contour=None):
    x, y, z = to_grid(q_nh, q_oh, values)
    mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    if contour is not None:
        _, _, cz = to_grid(q_nh, q_oh, contour)
        ax.contour(x, y, cz, levels=12, colors="k", linewidths=0.4, alpha=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$q_\mathrm{NH}$ [$\AA$]")
    ax.set_ylabel(r"$q_\mathrm{OH}$ [$\AA$]")
    plt.colorbar(mesh, ax=ax).set_label(cbar_label, fontsize=8)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ad = np.load(args.ad_arrays)
    pred = pd.read_csv(args.ad_predictions)
    orca = np.load(args.orca_cache, allow_pickle=True)

    h_ad_all = np.asarray(ad["hessians_cartesian"], dtype=float)
    f_ad_all = np.asarray(ad["forces"], dtype=float)
    ad_grid = pred["grid_id"].to_numpy(int)
    ad_order = np.argsort(ad_grid)

    orca_grid = np.asarray(orca["grid_id"], dtype=int)
    orca_order = np.argsort(orca_grid)
    q_nh = np.asarray(orca["q_nh"], dtype=float)[orca_order]
    q_oh = np.asarray(orca["q_oh"], dtype=float)[orca_order]
    coords = np.asarray(orca["coords_angstrom"], dtype=float)[orca_order]
    h_dft = symmetrize(np.asarray(orca["hessian_ev_ang2"], dtype=float)[orca_order])
    f_dft = np.asarray(orca["forces_ev_ang"], dtype=float)[orca_order] if "forces_ev_ang" in orca.files else None
    masses = np.asarray(orca["masses_amu"], dtype=float)
    dft_evals = np.asarray(orca["vib_evals_ev_ang2_amu"], dtype=float)[orca_order]
    dft_nneg = np.asarray(orca["n_negative"], dtype=int)[orca_order]

    h_ad = h_ad_all[ad_order]
    f_ad = f_ad_all[ad_order]
    if not np.array_equal(ad_grid[ad_order], orca_grid[orca_order]):
        raise ValueError("EQV2 AD grid_id order does not match ORCA cache grid_id order.")

    n = h_ad.shape[0]
    asym = 0.5 * (h_ad - np.swapaxes(h_ad, -1, -2))
    sym = 0.5 * (h_ad + np.swapaxes(h_ad, -1, -2))

    def fro(x):
        return np.linalg.norm(x.reshape(n, -1), axis=1)

    norm_dft = np.maximum(fro(h_dft), 1e-12)
    total_err = fro(h_ad - h_dft)
    asym_norm = fro(asym)
    sym_err = fro(sym - h_dft)

    total_rel = total_err / norm_dft
    sym_rel = sym_err / norm_dft
    asym_frac = asym_norm / np.maximum(fro(h_ad), 1e-12)
    asym_contrib = asym_norm**2 / np.maximum(total_err**2, 1e-24)
    force_mae = np.full(n, np.nan) if f_dft is None else np.mean(np.abs(f_ad - f_dft), axis=(1, 2))

    ad_evals = np.stack([vibrational_evals(h_ad[i], coords[i], masses) for i in range(n)])
    ad_nneg = (ad_evals < -NEG_THRESHOLD).sum(axis=1)
    nneg_delta = ad_nneg - dft_nneg
    eig0_err = ad_evals[:, 0] - dft_evals[:, 0]

    print(f"[ad-fail] N={n}")
    for label, v in (
        ("force MAE [eV/A]", force_mae),
        ("total H rel err", total_rel),
        ("sym-part rel err", sym_rel),
        ("asym fraction", asym_frac),
        ("asym contribution to err^2", asym_contrib),
        ("|eig0 err|", np.abs(eig0_err)),
    ):
        print(f"  {label:28s} med={np.nanmedian(v):.4g}  p90={np.nanquantile(v,0.9):.4g}")
    print(f"  neg-mode count correct: {(nneg_delta == 0).mean():.3f}")

    fig, axes = plt.subplots(2, 4, figsize=(21.5, 9.6))

    heatmap(axes[0, 0], q_nh, q_oh, force_mae, "A  EQV2 force MAE vs DFT (premise: forces OK)",
            "eV/$\\AA$", cmap="viridis")
    heatmap(axes[0, 1], q_nh, q_oh, total_rel, r"B  AD Hessian error $\|H_{AD}-H^*\|/\|H^*\|$",
            "rel. Frobenius", cmap="magma")
    heatmap(axes[0, 2], q_nh, q_oh, nneg_delta.astype(float),
            f"C  neg-mode count error (correct {100*(nneg_delta==0).mean():.0f}%)",
            r"$n_\mathrm{neg}^{AD}-n_\mathrm{neg}^{*}$", cmap="coolwarm", vmin=-3, vmax=3)
    heatmap(axes[0, 3], q_nh, q_oh, eig0_err, r"D  lowest-mode eigenvalue error",
            r"$\lambda_0^{AD}-\lambda_0^{*}$", cmap="coolwarm",
            vmin=-np.nanquantile(np.abs(eig0_err), 0.95), vmax=np.nanquantile(np.abs(eig0_err), 0.95))

    heatmap(axes[1, 0], q_nh, q_oh, asym_frac, r"E  AD asymmetry $\|\mathrm{asym}(H_{AD})\|/\|H_{AD}\|$",
            "fraction (non-conservativeness)", cmap="viridis")
    heatmap(axes[1, 1], q_nh, q_oh, asym_contrib,
            r"F  asym share of error  $\|\mathrm{asym}\|^2/\|H_{AD}-H^*\|^2$",
            "fraction of total error", cmap="viridis", vmin=0, vmax=1)
    heatmap(axes[1, 2], q_nh, q_oh, sym_rel, r"G  symmetric-part error $\|\mathrm{sym}(H_{AD})-H^*\|/\|H^*\|$",
            "rel. Frobenius (unsupervised Jacobian)", cmap="magma")

    # H: eigenvalue spectrum where AD invents a spurious mode (n_neg too high), zoomed to soft modes.
    ax = axes[1, 3]
    spurious = np.where(nneg_delta > 0)[0]
    if spurious.size:
        # most physically relevant: closest to the symmetric proton-transfer valley.
        sel = int(spurious[np.argmin(np.abs(q_nh[spurious] - q_oh[spurious]))])
    else:
        idx1 = np.where(dft_nneg == 1)[0]
        sel = int(idx1[np.argmin(dft_evals[idx1, 0])]) if idx1.size else int(np.argmin(dft_evals[:, 0]))
    k = min(args.n_eigs, dft_evals.shape[1], ad_evals.shape[1])
    modes = np.arange(k)
    ax.plot(modes, dft_evals[sel, :k], "o-", color="k", label=f"DFT ($n_-$={dft_nneg[sel]})")
    ax.plot(modes, ad_evals[sel, :k], "s--", color="tab:red", label=f"EQV2 AD ($n_-$={ad_nneg[sel]})")
    ax.axhline(0.0, color="grey", lw=0.8)
    lo = min(float(dft_evals[sel, :k].min()), float(ad_evals[sel, :k].min()))
    ax.set_ylim(max(lo, -5.0) - 0.5, float(np.maximum(dft_evals[sel, :k], ad_evals[sel, :k]).max()) * 1.1 + 0.5)
    ax.set_title(f"H  low-mode spectrum, AD adds a mode\n($q_{{NH}}$={q_nh[sel]:.2f}, $q_{{OH}}$={q_oh[sel]:.2f})", fontsize=10)
    ax.set_xlabel("vibrational mode index")
    ax.set_ylabel(r"$\lambda$ [eV $\AA^{-2}$ amu$^{-1}$]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "EQV2 autograd Hessian failure: accurate forces, inaccurate force-Jacobian "
        "(non-conservative + unsupervised)",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = args.output_dir / "glycine_ad_hessian_failure.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
