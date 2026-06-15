#!/usr/bin/env python
"""Full-Hessian diagnostics along the glycine proton-transfer path (MLIP-only).

The reaction-coordinate curvature (``plot_glycine_pt_path_mechanism.py``) showed no
difference between EQV2 autograd and HIP direct Hessians. The autograd degradation lives
in the *rest* of the Hessian. This figure compares EQV2 (autograd) vs HIP (direct) on:

A  lowest vibrational eigenvalues lambda_0..2(xi)   -- soft/reaction modes
B  number of negative modes(xi)                      -- saddle character
C  per-mode RMS eigenvalue difference |dlambda|      -- which modes disagree
D  autograd non-conservativeness ||H-H^T||/||H||     -- structural defect of the force Jacobian
E  ||H_AD - H_HIP||_F / ||H_HIP||_F (xi)             -- where the models disagree
F  text summary

No DFT is needed: this localizes the EQV2-vs-HIP discrepancy. Anchor to DFT afterwards
with the 1000-point ORCA scan if the differences are convincing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HIP_COLOR = "tab:blue"
EQV2_COLOR = "tab:red"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-dir", type=Path, default=Path("runs/glycine_pt_path"))
    parser.add_argument("--path-arrays", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--neg-thresh", type=float, default=-1e-6, help="eigenvalue sign threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays_path = args.path_arrays or args.path_dir / "path_arrays.npz"
    output_dir = args.output_dir or args.path_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(arrays_path)
    if "eqv2_evals" not in data.files:
        raise SystemExit(
            f"{arrays_path} lacks 'eqv2_evals'; re-run glycine_pt_path_scan.py (enhanced version)."
        )

    xi = np.asarray(data["xi"], dtype=float)
    order = np.argsort(xi)
    xi = xi[order]

    def col(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=float)[order]

    eqv2_evals = np.asarray(data["eqv2_evals"], dtype=float)[order]
    hip_evals = np.asarray(data["hip_evals"], dtype=float)[order]
    eqv2_neg, hip_neg = col("eqv2_n_negative"), col("hip_n_negative")
    eqv2_asym, hip_asym = col("eqv2_asym"), col("hip_asym")
    h_diff = col("h_diff_frob_rel")

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.6))

    # A: lowest eigenvalues
    ax = axes[0, 0]
    for k, ls in zip(range(3), ("-", "--", ":"), strict=True):
        ax.plot(xi, eqv2_evals[:, k], color=EQV2_COLOR, ls=ls, lw=1.2,
                label=f"EQV2 $\\lambda_{k}$")
        ax.plot(xi, hip_evals[:, k], color=HIP_COLOR, ls=ls, lw=1.2,
                label=f"HIP $\\lambda_{k}$")
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_title(r"A  Lowest vibrational eigenvalues")
    ax.set_ylabel(r"$\lambda$ [eV/$\AA^2$/amu]")
    ax.legend(fontsize=7, ncol=3)

    # B: negative-mode count
    ax = axes[0, 1]
    ax.step(xi, hip_neg, where="mid", color=HIP_COLOR, lw=1.4, label="HIP")
    ax.step(xi, eqv2_neg, where="mid", color=EQV2_COLOR, lw=1.4, alpha=0.8, label="EQV2")
    ax.set_title("B  Number of negative modes")
    ax.set_ylabel("count")
    ax.set_yticks(range(int(max(eqv2_neg.max(), hip_neg.max())) + 1))
    ax.legend(fontsize=8)

    # C: per-mode RMS eigenvalue difference
    ax = axes[0, 2]
    per_mode_rms = np.sqrt(np.mean((eqv2_evals - hip_evals) ** 2, axis=0))
    ax.bar(np.arange(per_mode_rms.size), per_mode_rms, color="tab:purple", alpha=0.8)
    ax.set_title("C  Per-mode RMS eigenvalue diff (EQV2 vs HIP)")
    ax.set_xlabel("mode index (ascending)")
    ax.set_ylabel(r"RMS $|\Delta\lambda|$ [eV/$\AA^2$/amu]")

    # D: autograd non-conservativeness
    ax = axes[1, 0]
    ax.plot(xi, hip_asym, color=HIP_COLOR, lw=1.2, label="HIP")
    ax.plot(xi, eqv2_asym, color=EQV2_COLOR, lw=1.2, label="EQV2 (autograd)")
    ax.set_title(r"D  Non-conservativeness $\|H-H^\top\|/\|H\|$")
    ax.set_xlabel(r"$\xi = q_\mathrm{NH}-q_\mathrm{OH}$ [$\AA$]")
    ax.set_ylabel("relative asymmetry")
    ax.legend(fontsize=8)

    # E: model-model Hessian difference
    ax = axes[1, 1]
    ax.plot(xi, h_diff, color="tab:green", lw=1.2)
    ax.set_title(r"E  $\|H_\mathrm{AD}-H_\mathrm{HIP}\|_F/\|H_\mathrm{HIP}\|_F$")
    ax.set_xlabel(r"$\xi = q_\mathrm{NH}-q_\mathrm{OH}$ [$\AA$]")
    ax.set_ylabel("relative difference")

    # F: text summary
    ax = axes[1, 2]
    ax.axis("off")
    disagree = float(np.mean(eqv2_neg != hip_neg))
    summary = [
        "EQV2 autograd vs HIP direct (MLIP-only)",
        "",
        f"neg-mode count disagree: {disagree * 100:.1f}% of path",
        f"asymmetry  EQV2 med: {np.median(eqv2_asym):.3e}",
        f"asymmetry  HIP  med: {np.median(hip_asym):.3e}",
        f"||H_AD-H_HIP||/||H_HIP|| med: {np.median(h_diff):.3e}",
        f"  p90: {np.quantile(h_diff, 0.9):.3e}",
        "",
        "softest-mode disagreement and",
        "autograd asymmetry localize the",
        "autograd Hessian defect; confirm",
        "vs DFT with the 1000-pt ORCA scan.",
    ]
    ax.text(0.02, 0.98, "\n".join(summary), transform=ax.transAxes,
            fontsize=10, va="top", family="monospace")

    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xlabel(r"$\xi = q_\mathrm{NH}-q_\mathrm{OH}$ [$\AA$]")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    fig.suptitle("Glycine PT path: where the EQV2 autograd Hessian departs from HIP direct", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / "glycine_pt_path_hessian_diag.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
