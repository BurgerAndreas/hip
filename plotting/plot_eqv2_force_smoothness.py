#!/usr/bin/env python
"""Insight figures for the smoothness of the *eqv2* (baseline EquiformerV2) force field.

Reads the arrays/summaries produced by ``scripts/force_smoothness_scan.py`` for the
``eqv2`` checkpoint (tags ``eqv2_test_s0_{transition_state,reactant}_ext``) and renders
several standalone figures, each focused on one insight. No model evaluation is needed
here -- everything is computed from the saved arrays, so this runs on CPU in seconds.

Key fact this set of figures is built around: for eqv2's *direct, (nearly) non-conservative*
forces, the autograd Hessian is H = -dF/dx (the Jacobian of the force field). Its reliability
is therefore entirely governed by how smooth F(x) is.

Figures:
  1. fd_convergence      - autograd-H vs finite-difference error vs step size h (the headline:
                           a smooth field would track O(h^2) down to the float-noise floor).
  2. force_spectrum      - power spectrum of the directional force d.F (broadband high-frequency
                           content == roughness).
  3. conservativeness    - directional force d.F vs -dE/dl along the scan, and their residual
                           (non-conservativeness of the direct forces).
  4. hessian_asymmetry   - heatmaps of the autograd Hessian, its symmetric and antisymmetric
                           parts (a true PES Hessian is symmetric; the differentiated force
                           field is not).
  5. vib_eigenspectrum   - mass-weighted, Eckart-projected vibrational eigenvalues from the
                           (symmetrised) autograd Hessian, and the implied stationary-point
                           classification (TS should have exactly one negative mode).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from hip.frequency_analysis import analyze_frequencies_torch
from plot_style import AD_COLOR, ACCENT_COLOR, GUIDE_COLOR, GUIDE_LINE_WIDTH, HIP_COLOR, LINE_WIDTH, THIN_LINE_WIDTH, finish_axis

GEOMS = ["transition_state", "reactant"]
GEOM_LABEL = {"transition_state": "transition state (saddle)", "reactant": "reactant (minimum)"}
DIRS = ["lowest_hess", "random"]
DIR_COLOR = {"lowest_hess": AD_COLOR, "random": HIP_COLOR}
MODEL = "EquiformerV2 baseline (eqv2.ckpt)"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_summary(summary_path: Path) -> dict:
    s = json.loads(summary_path.read_text())
    s["_atomic_numbers"] = np.array([int(x) for x in re.findall(r"-?\d+", s["atomic_numbers"])])
    coords = np.array([float(x) for x in re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", s["coords0"])])
    s["_coords0"] = coords.reshape(-1, 3)
    return s


def _load(run_dir: Path, geom: str) -> tuple[dict, dict]:
    tag = f"eqv2_test_s0_{geom}_ext"
    npz = dict(np.load(run_dir / f"{tag}_arrays.npz"))
    summary = _parse_summary(run_dir / f"{tag}_summary.json")
    return npz, summary


# ---------------------------------------------------------------------------
def fig_fd_convergence(data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, geom in zip(axes, GEOMS):
        npz, summ = data[geom]
        h = npz["h_values"]
        # ax.loglog(h, npz["dir_err"], "o-", color="C0", ms=4,
        #           label=r"directional  $\|H_{fd}\,d - H d\| / \|H d\|$")
        sns.lineplot(
            x=npz["h_full"],
            y=npz["full_err"],
            ax=ax,
            marker="s",
            color=HIP_COLOR,
            label=r"full Jacobian  $\|H_{fd} - H\| / \|H\|$",
        )
        # ax.loglog(h, npz["noise_floor"], ":", color="grey",
        #           label=r"$\sim$ float-noise floor ($\epsilon|F|/h$)")
        # O(h^2) guide anchored at the smallest-h directional point
        anchor = npz["dir_err"][0]
        sns.lineplot(x=h, y=anchor * (h / h[0]) ** 2, ax=ax, color=GUIDE_COLOR, linestyle="--", linewidth=GUIDE_LINE_WIDTH, alpha=0.7, label=r"$O(h^2)$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{GEOM_LABEL[geom]}", fontsize=10)
        ax.set_xlabel(r"FD step size  $h$  [$\AA$]")
        finish_axis(ax)
    axes[0].set_ylabel("relative error vs autograd Hessian")
    axes[0].legend(fontsize=8, loc="lower right")
    # fig.suptitle(
    #     f"Autograd Hessian never converges to finite differences — {MODEL}\n"
    #     "flat & far above the noise floor (ignores the $O(h^2)$ guide) ⇒ force field is not smooth",
    #     fontsize=11,
    # )
    fig.tight_layout(pad=0.01)
    p = out / "eqv2_fd_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_force_spectrum(data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, geom in zip(axes, GEOMS):
        npz, summ = data[geom]
        for kind in DIRS:
            freqs = npz[f"scan_{kind}_freqs"]
            mag = npz[f"scan_{kind}_mag"]
            hf = summ["hf_power_fraction"][kind]
            sns.lineplot(x=freqs, y=mag + 1e-30, ax=ax, lw=THIN_LINE_WIDTH, color=DIR_COLOR[kind], label=f"{kind}  (HF frac={hf:.1e})")
        ax.set_yscale("log")
        ax.axvline(20.0, color="grey", ls=":", lw=1)
        ax.set_title(GEOM_LABEL[geom], fontsize=10)
        ax.set_xlabel("spatial frequency [cycles/$\\AA$]")
        finish_axis(ax)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$|\mathrm{FFT}(d\cdot F)|$")
    fig.suptitle(
        f"Force power spectrum along a line scan"
    )
    fig.tight_layout(pad=0.01)
    p = out / "eqv2_force_spectrum.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_conservativeness(data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for col, geom in enumerate(GEOMS):
        npz, summ = data[geom]
        kind = "random"  # cleaner, well-excited direction
        lam = npz[f"scan_{kind}_lam"]
        g = npz[f"scan_{kind}_g"]            # d.F  (directional force)
        E = npz[f"scan_{kind}_E"]
        dl = float(lam[1] - lam[0])
        minus_dEdl = -np.gradient(E, dl)     # for a conservative field == g
        resid = g - minus_dEdl

        top, bot = axes[0, col], axes[1, col]
        sns.lineplot(x=lam, y=g, ax=top, lw=LINE_WIDTH, color=AD_COLOR, label=r"$d\cdot F$ (directional force)")
        sns.lineplot(x=lam, y=minus_dEdl, ax=top, lw=LINE_WIDTH, ls="--", color=HIP_COLOR, label=r"$-dE/d\lambda$ (from energy)")
        top.set_title(f"{GEOM_LABEL[geom]}", fontsize=10)
        finish_axis(top)
        top.legend(fontsize=8)

        sns.lineplot(x=lam, y=resid, ax=bot, lw=THIN_LINE_WIDTH, color=ACCENT_COLOR)
        bot.axhline(0, color="k", lw=GUIDE_LINE_WIDTH, alpha=0.5)
        bot.set_xlabel(r"random displacement $\lambda$ [$\AA$]")
        finish_axis(bot)
    axes[0, 0].set_ylabel("force [eV/$\\AA$]")
    axes[1, 0].set_ylabel(r"residual $d\cdot F - (-dE/d\lambda)$ [eV/$\AA$]")
    # fig.suptitle(
    #     f"Are the direct forces conservative? — {MODEL}\n"
    #     r"gap between $d\cdot F$ and $-dE/d\lambda$ = non-conservativeness (random direction)",
    #     fontsize=11,
    # )
    fig.tight_layout(pad=0.01)
    p = out / "eqv2_conservativeness.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_hessian_asymmetry(data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for row, geom in enumerate(GEOMS):
        npz, summ = data[geom]
        H = npz["H_autograd"]
        sym = 0.5 * (H + H.T)
        anti = 0.5 * (H - H.T)
        vmax = float(np.abs(H).max())
        panels = [(H, "autograd H  $(-dF/dx)$"), (sym, "symmetric part"), (anti, "antisymmetric part")]
        for col, (M, title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{GEOM_LABEL[geom]}\nrow index (3N)", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        anti_frac = float(np.linalg.norm(anti) / (np.linalg.norm(H) + 1e-30))
        axes[row, 2].text(0.5, -0.14, f"antisym fraction = {anti_frac:.3f}",
                          transform=axes[row, 2].transAxes, ha="center", fontsize=9, color=HIP_COLOR)
    fig.suptitle(
        f"Autograd Hessian symmetry",
        fontsize=11,
    )
    fig.tight_layout(pad=0.01, rect=[0, 0.02, 1, 1])
    p = out / "eqv2_hessian_asymmetry.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_vib_eigenspectrum(data: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows = []
    for ax, geom in zip(axes, GEOMS):
        npz, summ = data[geom]
        H = torch.tensor(npz["H_autograd"], dtype=torch.float64)
        Hsym = 0.5 * (H + H.T)  # symmetrise before physical analysis
        coords = torch.tensor(summ["_coords0"], dtype=torch.float64)
        z = torch.tensor(summ["_atomic_numbers"], dtype=torch.long)
        freq = analyze_frequencies_torch(Hsym, coords, z)
        ev = freq["eigvals"].detach().cpu().numpy()
        neg = int(freq["neg_num"])
        idx = np.arange(ev.size)
        colors = np.where(ev < 0, HIP_COLOR, AD_COLOR)
        sns.barplot(x=idx, y=ev, ax=ax, palette=colors.tolist(), hue=idx, legend=False)
        ax.axhline(0, color="k", lw=0.7)
        expected = 1 if geom == "transition_state" else 0
        ok = "✓" if neg == expected else "✗"
        ax.set_title(f"{GEOM_LABEL[geom]}\nnegative modes: {neg} (expected {expected}) {ok}", fontsize=10)
        ax.set_xlabel("mode index (sorted)")
        finish_axis(ax)
        rows.append((geom, neg, expected))
    axes[0].set_ylabel("Eigenvalue (mass-weighted, Eckart-projected)")
    # fig.suptitle(
    #     f"Stationary-point classification from the autograd Hessian — {MODEL}\n"
    #     "red = negative (imaginary) modes; non-smoothness can inject/remove spurious curvature",
    #     fontsize=11,
    # )
    fig.tight_layout(pad=0.01)
    p = out / "eqv2_vib_eigenspectrum.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=_project_root() / "runs" / "force_smoothness")
    ap.add_argument("--out-dir", type=Path, default=_project_root() / "runs" / "force_smoothness" / "eqv2_figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = {g: _load(args.run_dir, g) for g in GEOMS}

    written = []
    written.append(fig_fd_convergence(data, args.out_dir))
    written.append(fig_force_spectrum(data, args.out_dir))
    written.append(fig_conservativeness(data, args.out_dir))
    written.append(fig_hessian_asymmetry(data, args.out_dir))
    vib_png, vib_rows = fig_vib_eigenspectrum(data, args.out_dir)
    written.append(vib_png)

    print("Negative-mode check (autograd Hessian, symmetrised):")
    for geom, neg, exp in vib_rows:
        print(f"  {geom:18s}: {neg} negative (expected {exp})")
    print("\nWrote figures:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
