#!/usr/bin/env python
"""Joint median force-spectrum plot for EqV2, EqV2 (no H), and HIP-v2."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_style import DFT_COLOR, EQV2_FORCE_COLOR, EQV2_NO_H_FORCE_COLOR, GUIDE_COLOR, HIP_FORCE_COLOR, finish_axis


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_SCAN_DIR = project_root() / "runs" / "t1x_val_force_spectra_100x2x51"


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


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
    ax.fill_between(freqs, q25 + 1e-30, q75 + 1e-30, color=color, alpha=0.14)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--cutoff", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    arrays = {
        "EqV2": load_arrays(args.scan_dir / "force_spectra_analysis" / "force_spectra_arrays.npz"),
        "EqV2 (no H)": load_arrays(args.scan_dir / "force_spectra_analysis_eqv2_orig" / "force_spectra_arrays.npz"),
        "HIP": load_arrays(args.scan_dir / "force_spectra_analysis_hip_v2" / "force_spectra_arrays.npz"),
    }
    freqs = arrays["EqV2"]["freqs"]
    dft_mag = arrays["EqV2"]["dft_mag"]
    for label, data in arrays.items():
        if not np.allclose(freqs, data["freqs"]):
            raise ValueError(f"{label} frequency grid differs")
        if not np.allclose(dft_mag, data["dft_mag"]):
            raise ValueError(f"{label} DFT spectra differ")

    fig, ax = plt.subplots(figsize=(9, 5.4))
    plot_with_iqr(ax, freqs, dft_mag, color=DFT_COLOR, label="DFT")
    plot_with_iqr(ax, freqs, arrays["EqV2"]["eqv2_mag"], color=EQV2_FORCE_COLOR, label="EqV2")
    plot_with_iqr(
        ax,
        freqs,
        arrays["EqV2 (no H)"]["eqv2_mag"],
        color=EQV2_NO_H_FORCE_COLOR,
        label="EqV2 (no H)",
        linestyle=":",
    )
    plot_with_iqr(ax, freqs, arrays["HIP"]["eqv2_mag"], color=HIP_FORCE_COLOR, label="HIP")
    ax.axvline(args.cutoff, color=GUIDE_COLOR, linestyle="--", linewidth=1.2, label=f"{args.cutoff:g} cyc/Å")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e0)
    ax.set_xlim(0, 100)
    ax.set_xlabel(r"spatial frequency [cycles/$\AA$]")
    ax.set_ylabel(r"$|\mathrm{FFT}(d \cdot F)|$")
    # ax.set_title("Median force spectrum across 200 lines")
    finish_axis(ax, legend=True)
    fig.tight_layout(pad=0.01)

    out_path = args.output or args.scan_dir / "eqv2_eqv2_orig_hip_v2_median_force_spectra.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
