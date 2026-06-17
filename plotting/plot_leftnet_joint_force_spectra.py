#!/usr/bin/env python
"""Joint median force-spectrum plot for LeftNet-CF/DF and their orig variants."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_style import DFT_COLOR, GUIDE_COLOR, LEFTNET_CF_FORCE_COLOR, LEFTNET_DF_FORCE_COLOR, finish_axis


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_ANALYSIS_DIR = (
    project_root()
    / "runs"
    / "t1x_val_force_spectra_100x2x51"
    / "t1x_val_force_spectra_leftnet"
    / "force_spectra_analysis"
)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def tagged_name(stem: str, suffix: str) -> str:
    return f"{stem}{suffix}.png"


def plot_with_iqr(
    ax,
    freqs: np.ndarray,
    mags: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    median = np.median(mags, axis=0)
    q25 = np.quantile(mags, 0.25, axis=0)
    q75 = np.quantile(mags, 0.75, axis=0)
    ax.plot(freqs, median + 1e-30, color=color, label=label, linewidth=2.2, linestyle=linestyle)
    ax.fill_between(freqs, q25 + 1e-30, q75 + 1e-30, color=color, alpha=0.13)


def plot_error_with_iqr(
    ax,
    freqs: np.ndarray,
    dft_mag: np.ndarray,
    model_mag: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    err = np.abs(model_mag - dft_mag)
    median = np.median(err, axis=0)
    q25 = np.quantile(err, 0.25, axis=0)
    q75 = np.quantile(err, 0.75, axis=0)
    ax.plot(freqs, median + 1e-30, color=color, label=label, linewidth=2.2, linestyle=linestyle)
    ax.fill_between(freqs, q25 + 1e-30, q75 + 1e-30, color=color, alpha=0.13)


def plot_log_ratio_with_iqr(
    ax,
    freqs: np.ndarray,
    dft_mag: np.ndarray,
    model_mag: np.ndarray,
    *,
    color: str,
    label: str,
    eps: float,
    linestyle: str = "-",
) -> None:
    ratio = np.log10((model_mag + eps) / (dft_mag + eps))
    median = np.median(ratio, axis=0)
    q25 = np.quantile(ratio, 0.25, axis=0)
    q75 = np.quantile(ratio, 0.75, axis=0)
    ax.plot(freqs, median, color=color, label=label, linewidth=2.2, linestyle=linestyle)
    ax.fill_between(freqs, q25, q75, color=color, alpha=0.13)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--cutoff", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--error-output", type=Path, default=None)
    parser.add_argument("--log-ratio-output", type=Path, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--ratio-eps", type=float, default=1e-16)
    args = parser.parse_args()

    arrays = {
        "LeftNet-CF": load_arrays(args.analysis_dir / "leftnet-cf" / "force_spectra_arrays.npz"),
        "LeftNet-DF": load_arrays(args.analysis_dir / "leftnet-df" / "force_spectra_arrays.npz"),
        "LeftNet-CF (no H)": load_arrays(args.analysis_dir / "leftnet-cf-orig" / "force_spectra_arrays.npz"),
        "LeftNet-DF (no H)": load_arrays(args.analysis_dir / "leftnet-df-orig" / "force_spectra_arrays.npz"),
    }
    freqs = arrays["LeftNet-CF"]["freqs"]
    dft_mag = arrays["LeftNet-CF"]["dft_mag"]
    for label, data in arrays.items():
        if not np.allclose(freqs, data["freqs"]):
            raise ValueError(f"{label} frequency grid differs")
        if not np.allclose(dft_mag, data["dft_mag"]):
            raise ValueError(f"{label} DFT spectra differ")

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    plot_with_iqr(ax, freqs, dft_mag, color=DFT_COLOR, label="DFT")
    plot_with_iqr(ax, freqs, arrays["LeftNet-CF"]["eqv2_mag"], color=LEFTNET_CF_FORCE_COLOR, label="LeftNet-CF")
    plot_with_iqr(ax, freqs, arrays["LeftNet-DF"]["eqv2_mag"], color=LEFTNET_DF_FORCE_COLOR, label="LeftNet-DF")
    plot_with_iqr(
        ax,
        freqs,
        arrays["LeftNet-CF (no H)"]["eqv2_mag"],
        color=LEFTNET_CF_FORCE_COLOR,
        label="LeftNet-CF (no H)",
        linestyle=":",
    )
    plot_with_iqr(
        ax,
        freqs,
        arrays["LeftNet-DF (no H)"]["eqv2_mag"],
        color=LEFTNET_DF_FORCE_COLOR,
        label="LeftNet-DF (no H)",
        linestyle=":",
    )
    ax.axvline(args.cutoff, color=GUIDE_COLOR, linestyle="--", linewidth=1.2, label=f"{args.cutoff:g} cyc/Å")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e0)
    ax.set_xlim(0, 100)
    ax.set_xlabel(r"spatial frequency [cycles/$\AA$]")
    ax.set_ylabel(r"$|\mathrm{FFT}(d \cdot F)|$")
    # ax.set_title("Median force spectrum across 200 lines")
    finish_axis(ax, legend=True)
    fig.tight_layout(pad=0.01)

    out_path = args.output or project_root() / "plots" / "t1x_val_force_spectra_100x2x51" / tagged_name(
        "leftnet_cf_df_median_force_spectra", args.output_suffix
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    plot_error_with_iqr(
        ax, freqs, dft_mag, arrays["LeftNet-CF"]["eqv2_mag"], color=LEFTNET_CF_FORCE_COLOR, label="LeftNet-CF"
    )
    plot_error_with_iqr(
        ax, freqs, dft_mag, arrays["LeftNet-DF"]["eqv2_mag"], color=LEFTNET_DF_FORCE_COLOR, label="LeftNet-DF"
    )
    plot_error_with_iqr(
        ax,
        freqs,
        dft_mag,
        arrays["LeftNet-CF (no H)"]["eqv2_mag"],
        color=LEFTNET_CF_FORCE_COLOR,
        label="LeftNet-CF (no H)",
        linestyle=":",
    )
    plot_error_with_iqr(
        ax,
        freqs,
        dft_mag,
        arrays["LeftNet-DF (no H)"]["eqv2_mag"],
        color=LEFTNET_DF_FORCE_COLOR,
        label="LeftNet-DF (no H)",
        linestyle=":",
    )
    ax.axvline(args.cutoff, color=GUIDE_COLOR, linestyle="--", linewidth=1.2, label=f"{args.cutoff:g} cyc/Å")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e0)
    ax.set_xlim(0, 100)
    ax.set_xlabel(r"spatial frequency [cycles/$\AA$]")
    ax.set_ylabel(r"$||\mathrm{FFT}(d \cdot F)|_\mathrm{model} - |\mathrm{FFT}(d \cdot F)|_\mathrm{DFT}|$")
    finish_axis(ax, legend=True)
    fig.tight_layout(pad=0.01)

    err_path = (
        args.error_output
        or project_root()
        / "plots"
        / "t1x_val_force_spectra_100x2x51"
        / tagged_name("leftnet_cf_df_median_force_spectra_error_vs_dft", args.output_suffix)
    )
    err_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(err_path, dpi=180)
    plt.close(fig)
    print(f"Saved {err_path}")

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    plot_log_ratio_with_iqr(
        ax,
        freqs,
        dft_mag,
        arrays["LeftNet-CF"]["eqv2_mag"],
        color=LEFTNET_CF_FORCE_COLOR,
        label="LeftNet-CF",
        eps=args.ratio_eps,
    )
    plot_log_ratio_with_iqr(
        ax,
        freqs,
        dft_mag,
        arrays["LeftNet-DF"]["eqv2_mag"],
        color=LEFTNET_DF_FORCE_COLOR,
        label="LeftNet-DF",
        eps=args.ratio_eps,
    )
    plot_log_ratio_with_iqr(
        ax,
        freqs,
        dft_mag,
        arrays["LeftNet-CF (no H)"]["eqv2_mag"],
        color=LEFTNET_CF_FORCE_COLOR,
        label="LeftNet-CF (no H)",
        eps=args.ratio_eps,
        linestyle=":",
    )
    plot_log_ratio_with_iqr(
        ax,
        freqs,
        dft_mag,
        arrays["LeftNet-DF (no H)"]["eqv2_mag"],
        color=LEFTNET_DF_FORCE_COLOR,
        label="LeftNet-DF (no H)",
        eps=args.ratio_eps,
        linestyle=":",
    )
    ax.axhline(0.0, color=GUIDE_COLOR, linestyle="-", linewidth=1.0)
    ax.axvline(args.cutoff, color=GUIDE_COLOR, linestyle="--", linewidth=1.2, label=f"{args.cutoff:g} cyc/Å")
    ax.set_xlim(0, 100)
    ax.set_xlabel(r"spatial frequency [cycles/$\AA$]")
    ax.set_ylabel(r"$\log_{10}\left((|\mathrm{FFT}|_\mathrm{model}+\epsilon)/(|\mathrm{FFT}|_\mathrm{DFT}+\epsilon)\right)$")
    finish_axis(ax, legend=True)
    fig.tight_layout(pad=0.01)

    ratio_path = (
        args.log_ratio_output
        or project_root()
        / "plots"
        / "t1x_val_force_spectra_100x2x51"
        / tagged_name("leftnet_cf_df_median_force_spectra_log_ratio_vs_dft", args.output_suffix)
    )
    ratio_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ratio_path, dpi=180)
    plt.close(fig)
    print(f"Saved {ratio_path}")


if __name__ == "__main__":
    main()
