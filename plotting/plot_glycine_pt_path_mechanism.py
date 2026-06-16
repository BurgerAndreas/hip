#!/usr/bin/env python
"""AD-only figures: why autograd Hessians from forces fail.

Consumes the dense 1D proton-transfer path from ``scripts/glycine_pt_path_scan.py``
(``path_arrays.npz``) and writes several focused figures. Only the EQV2 model is
plotted, and it is labeled as "AD" because its Hessian is obtained by differentiating
the force field.

Figures:

- energy / force sanity checks: the model looks smooth at the usual level.
- force residual: a small high-frequency wiggle remains after removing the trend.
- curvature from force: differentiating the same force exposes the wiggle.
- spectra: differentiation weights high frequencies by k, amplifying tiny force noise.
- full-Hessian AD diagnostics: optional eigenvalue/asymmetry plots if present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from plot_style import ACCENT_COLOR, AD_COLOR, DFT_COLOR, LINE_WIDTH, THIN_LINE_WIDTH, finish_axis



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-dir", type=Path, default=Path("runs/glycine_pt_path"))
    parser.add_argument(
        "--path-arrays",
        type=Path,
        default=None,
        help="Dense MLIP npz. Defaults to path-dir/path_arrays.npz.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--detrend-degree", type=int, default=6)
    return parser.parse_args()


def detrend(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    degree = min(degree, max(1, y.size - 1))
    return y - np.polyval(np.polyfit(x, y, degree), x)


def savefig(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout(pad=0.01)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {path}")


def finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def plot_energy_force(output_dir: Path, dpi: int, xi: np.ndarray, energy: np.ndarray, force: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    xlabel = r"$\xi = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]"

    sns.lineplot(x=xi, y=energy - np.nanmin(energy), ax=axes[0], color=AD_COLOR, lw=LINE_WIDTH, label="AD")
    axes[0].set_title("Energy Looks Smooth")
    axes[0].set_ylabel(r"$E - \min E$ [eV]")
    axes[0].set_xlabel(xlabel)
    axes[0].legend(fontsize=8, frameon=True, edgecolor="none")

    sns.lineplot(x=xi, y=force, ax=axes[1], color=AD_COLOR, lw=LINE_WIDTH, label="AD")
    axes[1].set_title(r"Projected Force Looks Smooth")
    axes[1].set_ylabel(r"$g=\hat t\cdot F$ [eV/$\AA$]")
    axes[1].set_xlabel(xlabel)
    axes[1].legend(fontsize=8, frameon=True, edgecolor="none")

    for ax in axes:
        finish_axis(ax)
    fig.suptitle("AD force field passes the usual 1D sanity checks", fontsize=12)
    savefig(fig, output_dir / "glycine_pt_ad_energy_force.png", dpi)


def plot_force_residual(
    output_dir: Path,
    dpi: int,
    xi: np.ndarray,
    force: np.ndarray,
    residual: np.ndarray,
    detrend_degree: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    xlabel = r"$\xi = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]"
    trend = force - residual

    sns.lineplot(x=xi, y=force, ax=axes[0], color=AD_COLOR, lw=LINE_WIDTH, alpha=0.55, label="AD force")
    sns.lineplot(x=xi, y=trend, ax=axes[0], color=DFT_COLOR, lw=LINE_WIDTH, label=f"degree-{detrend_degree} trend")
    axes[0].set_title("Force = Smooth Trend + Small Wiggle")
    axes[0].set_ylabel(r"$g$ [eV/$\AA$]")
    axes[0].set_xlabel(xlabel)
    axes[0].legend(fontsize=8, frameon=True, edgecolor="none")

    sns.lineplot(x=xi, y=residual, ax=axes[1], color=AD_COLOR, lw=THIN_LINE_WIDTH, label="AD residual")
    axes[1].axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    axes[1].set_title("Hidden High-Frequency Force Residual")
    axes[1].set_ylabel(r"$g - \mathrm{trend}$ [eV/$\AA$]")
    axes[1].set_xlabel(xlabel)
    axes[1].legend(fontsize=8, frameon=True, edgecolor="none")

    for ax in axes:
        finish_axis(ax)
    fig.suptitle("The force error can be visually small before differentiation", fontsize=12)
    savefig(fig, output_dir / "glycine_pt_ad_force_residual.png", dpi)


def plot_curvature_failure(
    output_dir: Path,
    dpi: int,
    xi: np.ndarray,
    kappa_auto: np.ndarray,
    kappa_fd: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    xlabel = r"$\xi = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]"

    sns.lineplot(x=xi, y=kappa_auto, ax=axes[0], color=AD_COLOR, lw=LINE_WIDTH, label=r"AD Hessian $\hat t^\top H\hat t$")
    sns.lineplot(x=xi, y=kappa_fd, ax=axes[0], color=AD_COLOR, lw=THIN_LINE_WIDTH, alpha=0.35, label=r"AD force FD $-dg/d\lambda$")
    axes[0].axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    axes[0].set_title("Differentiating Force Exposes Wiggle")
    axes[0].set_ylabel(r"$\kappa$ [eV/$\AA^2$]")
    axes[0].set_xlabel(xlabel)
    axes[0].legend(fontsize=8, frameon=True, edgecolor="none")

    mismatch = kappa_auto - kappa_fd
    sns.lineplot(x=xi, y=mismatch, ax=axes[1], color=ACCENT_COLOR, lw=THIN_LINE_WIDTH)
    axes[1].axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    axes[1].set_title("AD Hessian vs Sampled Force Derivative")
    axes[1].set_ylabel(r"$\kappa_\mathrm{AD} - (-dg/d\lambda)$ [eV/$\AA^2$]")
    axes[1].set_xlabel(xlabel)

    for ax in axes:
        finish_axis(ax)
    fig.suptitle("Autograd Hessians inherit and amplify force-field roughness", fontsize=12)
    savefig(fig, output_dir / "glycine_pt_ad_curvature_failure.png", dpi)


def plot_spectral_amplification(
    output_dir: Path,
    dpi: int,
    freqs: np.ndarray,
    mag: np.ndarray,
    cutoff: float,
    meta: dict[str, float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    weighted = (2.0 * np.pi * freqs) ** 2 * mag

    sns.lineplot(x=freqs, y=mag + 1e-30, ax=axes[0], color=AD_COLOR, lw=THIN_LINE_WIDTH, label="AD")
    axes[0].set_yscale("log")
    axes[0].axvline(cutoff, color="grey", ls=":", lw=1)
    axes[0].set_title(r"Force Spectrum $|\mathrm{FFT}(g)|$")
    axes[0].set_xlabel(r"spatial frequency [cycles/$\AA$]")
    axes[0].set_ylabel(r"$|\mathrm{FFT}(g)|$")
    axes[0].legend(fontsize=8, frameon=True, edgecolor="none")

    sns.lineplot(x=freqs, y=weighted + 1e-30, ax=axes[1], color=AD_COLOR, lw=THIN_LINE_WIDTH, label="AD")
    axes[1].set_yscale("log")
    axes[1].axvline(cutoff, color="grey", ls=":", lw=1)
    axes[1].set_title(r"Curvature-Weighted Spectrum $(2\pi k)^2|\mathrm{FFT}(g)|$")
    axes[1].set_xlabel(r"spatial frequency [cycles/$\AA$]")
    axes[1].set_ylabel(r"curvature-amplified amplitude")
    axes[1].legend(fontsize=8, frameon=True, edgecolor="none")

    force_hf = meta.get("hf_fraction_force_eqv2")
    curvature_hf = meta.get("hf_fraction_curvature_eqv2")
    if force_hf is not None and curvature_hf is not None:
        ratio = curvature_hf / max(force_hf, 1e-30)
        axes[1].text(
            0.97,
            0.95,
            "AD high-frequency power\n"
            f"force: {force_hf:.2e}\n"
            f"after differentiation: {curvature_hf:.2e}\n"
            f"amplification: {ratio:.1f}x",
            transform=axes[1].transAxes,
            fontsize=8,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85),
        )

    for ax in axes:
        finish_axis(ax)
    fig.suptitle("Differentiation turns small high-frequency force noise into Hessian noise", fontsize=12)
    savefig(fig, output_dir / "glycine_pt_ad_spectral_amplification.png", dpi)


def plot_full_hessian_diagnostics(
    output_dir: Path,
    dpi: int,
    xi: np.ndarray,
    data: np.lib.npyio.NpzFile,
    order: np.ndarray,
) -> None:
    if not {"eqv2_evals", "eqv2_n_negative", "eqv2_asym"}.issubset(data.files):
        return

    evals = np.asarray(data["eqv2_evals"], dtype=float)[order]
    n_negative = np.asarray(data["eqv2_n_negative"], dtype=float)[order]
    asym = np.asarray(data["eqv2_asym"], dtype=float)[order]
    n_modes = min(4, evals.shape[1])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    xlabel = r"$\xi = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]"

    for mode_idx, ls in zip(range(n_modes), ("-", "--", ":", "-."), strict=False):
        sns.lineplot(x=xi, y=evals[:, mode_idx], ax=axes[0, 0], color=AD_COLOR, ls=ls, lw=LINE_WIDTH, label=fr"AD $\lambda_{mode_idx}$")
    axes[0, 0].axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    axes[0, 0].set_title("Lowest AD Vibrational Eigenvalues")
    axes[0, 0].set_ylabel(r"$\lambda$ [eV/$\AA^2$/amu]")
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=True, edgecolor="none")

    axes[0, 1].step(xi, n_negative, where="mid", color=AD_COLOR, lw=LINE_WIDTH, label="AD")
    axes[0, 1].set_title("AD Negative-Mode Count")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].legend(fontsize=8, frameon=True, edgecolor="none")

    sns.lineplot(x=xi, y=asym, ax=axes[1, 0], color=AD_COLOR, lw=LINE_WIDTH, label="AD")
    axes[1, 0].set_title(r"AD Hessian Asymmetry $\|H-H^\top\|/\|H\|$")
    axes[1, 0].set_ylabel("relative asymmetry")
    axes[1, 0].legend(fontsize=8, frameon=True, edgecolor="none")

    axes[1, 1].axis("off")
    emin, emax = finite_range(evals[:, 0])
    summary = [
        "AD full-Hessian diagnostics",
        "",
        f"lowest eigenvalue range: {emin:.3e} to {emax:.3e}",
        f"negative-mode count median: {np.median(n_negative):.1f}",
        f"negative-mode count max: {np.nanmax(n_negative):.0f}",
        f"H asymmetry median: {np.median(asym):.3e}",
        f"H asymmetry p90: {np.quantile(asym, 0.9):.3e}",
    ]
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(summary),
        transform=axes[1, 1].transAxes,
        fontsize=10,
        va="top",
        family="monospace",
    )

    for ax in axes.ravel():
        if ax is not axes[1, 1]:
            ax.set_xlabel(xlabel)
            finish_axis(ax)
    fig.suptitle("AD Hessian failure also appears in the full vibrational spectrum", fontsize=12)
    savefig(fig, output_dir / "glycine_pt_ad_full_hessian_diagnostics.png", dpi)


def main() -> None:
    args = parse_args()
    path_dir = args.path_dir
    arrays_path = args.path_arrays or path_dir / "path_arrays.npz"
    output_dir = args.output_dir or Path("plots") / path_dir.name / "mechanism"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(arrays_path)
    xi = np.asarray(data["xi"], dtype=float)
    order = np.argsort(xi)
    xi = xi[order]

    def col(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=float)[order]

    e_ad = col("eqv2_energy")
    g_ad = col("eqv2_g")
    x_along = col("x_along")
    kappa_auto = col("eqv2_kappa_auto")
    kappa_fd = col("eqv2_kappa_fd")

    meta = {}
    meta_path = path_dir / "path_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    cutoff = float(meta.get("spectrum_cutoff", 8.0))

    freqs_eqv2 = np.asarray(data["eqv2_g_freqs"], dtype=float)
    mag_eqv2 = np.asarray(data["eqv2_g_mag"], dtype=float)
    resid_ad = detrend(xi, g_ad, args.detrend_degree)
    dlam = float(np.mean(np.diff(x_along)))
    kappa_fd_check = -np.gradient(g_ad, dlam)
    if not np.allclose(kappa_fd, kappa_fd_check, rtol=1e-4, atol=1e-6):
        kappa_fd = kappa_fd_check

    plot_energy_force(output_dir, args.dpi, xi, e_ad, g_ad)
    plot_force_residual(output_dir, args.dpi, xi, g_ad, resid_ad, args.detrend_degree)
    plot_curvature_failure(output_dir, args.dpi, xi, kappa_auto, kappa_fd)
    plot_spectral_amplification(output_dir, args.dpi, freqs_eqv2, mag_eqv2, cutoff, meta)
    plot_full_hessian_diagnostics(output_dir, args.dpi, xi, data, order)


if __name__ == "__main__":
    main()
