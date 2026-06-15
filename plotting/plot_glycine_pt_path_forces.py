#!/usr/bin/env python
"""Forces-only view of the dense glycine proton-transfer path.

Uses only what ``scripts/glycine_pt_path_scan.py`` already saved in ``path_arrays.npz``
(no new compute): the path-projected force ``g = t.F`` (H9 along the N->O axis), the
max Cartesian force component ``fmax`` (sensitive to *off-axis* wiggle), the energy, and
the precomputed ``g`` spectrum.

Panels
  A  g(xi)                 projected force, EQV2 vs HIP
  B  residual(g)           detrended -> the on-axis wiggle
  C  |FFT(g)|              spectrum of the projected force
  D  fmax(xi)              max |force| component (off-axis sensitive)
  E  residual(fmax)        detrended -> global/off-axis wiggle
  F  g + dE/dlambda        non-conservativeness (0 for a conservative field)
"""
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--detrend-degree", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def detrend(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    degree = min(degree, max(1, y.size - 1))
    return y - np.polyval(np.polyfit(x, y, degree), x)


def wiggle_metrics(x_along: np.ndarray, y: np.ndarray, resid: np.ndarray, cutoff: float) -> dict[str, float]:
    dlam = float(np.mean(np.diff(x_along)))
    base = np.arange(y.size)
    win = np.hanning(y.size)
    mag = np.abs(np.fft.rfft((y - np.polyval(np.polyfit(base, y, 3), base)) * win))
    freqs = np.fft.rfftfreq(y.size, d=dlam)
    power = mag**2
    hf = float(power[freqs >= cutoff].sum() / max(power[1:].sum(), 1e-30))
    return {
        "resid_rms": float(np.sqrt(np.mean(resid**2))),
        "resid_max_abs": float(np.max(np.abs(resid))),
        "tv_per_ang": float(np.sum(np.abs(np.diff(y))) / (x_along[-1] - x_along[0])),
        "hf_frac": hf,
    }


def main() -> None:
    args = parse_args()
    path_dir = args.path_dir
    arrays_path = args.path_arrays or path_dir / "path_arrays.npz"
    output_dir = args.output_dir or path_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(arrays_path)
    order = np.argsort(np.asarray(data["xi"], dtype=float))

    def col(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=float)[order]

    xi = col("xi")
    x_along = col("x_along")
    g_eqv2, g_hip = col("eqv2_g"), col("hip_g")
    fmax_eqv2, fmax_hip = col("eqv2_fmax"), col("hip_fmax")
    e_eqv2, e_hip = col("eqv2_energy"), col("hip_energy")

    meta = {}
    meta_path = path_dir / "path_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    cutoff = float(meta.get("spectrum_cutoff", 8.0))

    res_g_eqv2 = detrend(xi, g_eqv2, args.detrend_degree)
    res_g_hip = detrend(xi, g_hip, args.detrend_degree)
    res_fmax_eqv2 = detrend(xi, fmax_eqv2, args.detrend_degree)
    res_fmax_hip = detrend(xi, fmax_hip, args.detrend_degree)

    # non-conservativeness: conservative field has g = -dE/dlambda
    dlam = float(np.mean(np.diff(x_along)))
    nc_eqv2 = g_eqv2 + np.gradient(e_eqv2, dlam)
    nc_hip = g_hip + np.gradient(e_hip, dlam)

    print(f"[forces] cutoff={cutoff:g} cyc/A   (detrend degree {args.detrend_degree})")
    for label, y, res in (
        ("EQV2 g", g_eqv2, res_g_eqv2),
        ("HIP  g", g_hip, res_g_hip),
        ("EQV2 fmax", fmax_eqv2, res_fmax_eqv2),
        ("HIP  fmax", fmax_hip, res_fmax_hip),
    ):
        m = wiggle_metrics(x_along, y, res, cutoff)
        print(
            f"  {label:10s} resid_rms={m['resid_rms']:.3e}  resid_max={m['resid_max_abs']:.3e}  "
            f"TV/A={m['tv_per_ang']:.3e}  HF_frac={m['hf_frac']:.3e} eV/A"
        )
    print(f"  non-conservativeness |g+dE/dl| max: EQV2={np.max(np.abs(nc_eqv2)):.3e}  HIP={np.max(np.abs(nc_hip)):.3e} eV/A")

    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.2))
    xlabel = r"$\xi = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]"

    ax = axes[0, 0]
    ax.plot(xi, g_hip, color=HIP_COLOR, lw=1.1, label="HIP")
    ax.plot(xi, g_eqv2, color=EQV2_COLOR, lw=1.1, label="EQV2")
    ax.set_title(r"A  Projected force $g=\hat t\cdot F$ (H9 along N$\to$O)")
    ax.set_ylabel(r"$g$ [eV/$\AA$]")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(xi, res_g_hip, color=HIP_COLOR, lw=0.9, alpha=0.85, label="HIP")
    ax.plot(xi, res_g_eqv2, color=EQV2_COLOR, lw=0.9, label="EQV2")
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_title(f"B  Projected-force residual (deg-{args.detrend_degree})")
    ax.set_ylabel(r"$g - \mathrm{trend}$ [eV/$\AA$]")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.semilogy(np.asarray(data["hip_g_freqs"]), np.asarray(data["hip_g_mag"]) + 1e-30,
                color=HIP_COLOR, lw=0.9, label="HIP")
    ax.semilogy(np.asarray(data["eqv2_g_freqs"]), np.asarray(data["eqv2_g_mag"]) + 1e-30,
                color=EQV2_COLOR, lw=0.9, label="EQV2")
    ax.axvline(cutoff, color="grey", ls=":", lw=1)
    ax.set_title("C  Force spectrum |FFT($g$)|")
    ax.set_xlabel(r"spatial frequency [cycles/$\AA$]")
    ax.set_ylabel(r"|FFT($g$)|")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(xi, fmax_hip, color=HIP_COLOR, lw=1.1, label="HIP")
    ax.plot(xi, fmax_eqv2, color=EQV2_COLOR, lw=1.1, label="EQV2")
    ax.set_title(r"D  Max force component $\max_i|F_i|$ (off-axis sensitive)")
    ax.set_ylabel(r"$f_\mathrm{max}$ [eV/$\AA$]")
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(xi, res_fmax_hip, color=HIP_COLOR, lw=0.9, alpha=0.85, label="HIP")
    ax.plot(xi, res_fmax_eqv2, color=EQV2_COLOR, lw=0.9, label="EQV2")
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_title(f"E  $f_\\mathrm{{max}}$ residual (deg-{args.detrend_degree})")
    ax.set_ylabel(r"$f_\mathrm{max} - \mathrm{trend}$ [eV/$\AA$]")
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(xi, nc_hip, color=HIP_COLOR, lw=0.9, alpha=0.85, label="HIP")
    ax.plot(xi, nc_eqv2, color=EQV2_COLOR, lw=0.9, label="EQV2")
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.set_title(r"F  Non-conservativeness  $g + dE/d\lambda$")
    ax.set_ylabel(r"$g + dE/d\lambda$ [eV/$\AA$]")
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xlabel(xlabel)

    fig.suptitle("Glycine proton transfer: MLIP forces along the path", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / "glycine_pt_path_forces.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
