#!/usr/bin/env python
"""Analyze EqV2 vs ORCA/DFT force spectra for the T1x-val line scans."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from plot_style import (
    AD_COLOR,
    DFT_COLOR,
    EQV2_FORCE_COLOR,
    EQV2_NO_H_FORCE_COLOR,
    GUIDE_COLOR,
    HIP_FORCE_COLOR,
    HIP_COLOR,
    LEFTNET_CF_FORCE_COLOR,
    LEFTNET_DF_FORCE_COLOR,
    finish_axis,
)

HARTREE_TO_EV = 27.211386245988


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_SCAN_DIR = project_root() / "runs" / "t1x_val_force_spectra_100x2x51"


def model_slug(model_label: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in model_label.strip())
    return "_".join(part for part in slug.split("_") if part) or "model"


def resolve_existing(path_value: str | Path, fallback_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    if fallback_dir is not None:
        fallback = fallback_dir / path.name
        if fallback.exists():
            return fallback
    return path


def line_npz_path(base: Path, geom_rank: int, dataset_idx: int, direction_id: int) -> Path:
    return base / f"g{geom_rank:04d}_idx{dataset_idx:06d}_d{direction_id}.npz"


def detrended_spectrum(signal: np.ndarray, dlam: float, detrend_degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(signal.size, dtype=np.float64)
    degree = min(detrend_degree, signal.size - 2)
    if degree >= 0:
        coeffs = np.polyfit(x, signal, degree)
        resid = signal - np.polyval(coeffs, x)
    else:
        resid = signal - np.mean(signal)
    window = np.hanning(signal.size)
    spec = np.fft.rfft(resid * window)
    freqs = np.fft.rfftfreq(signal.size, d=dlam)
    mag = np.abs(spec)
    return freqs, mag, resid


def high_frequency_fraction(freqs: np.ndarray, mag: np.ndarray, cutoff: float) -> float:
    power = mag**2
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    return float(np.sum(power[freqs >= cutoff]) / total)


def projected_force(forces: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return np.einsum("pij,ij->p", forces, direction)


def model_color(model_label: str) -> str:
    normalized = model_label.casefold()
    if "leftnet" in normalized and "cf" in normalized:
        return LEFTNET_CF_FORCE_COLOR
    if "leftnet" in normalized and "df" in normalized:
        return LEFTNET_DF_FORCE_COLOR
    if "hip" in normalized:
        return HIP_FORCE_COLOR
    if "no h" in normalized or "orig" in normalized:
        return EQV2_NO_H_FORCE_COLOR
    if "eqv2" in normalized:
        return EQV2_FORCE_COLOR
    return AD_COLOR


def save_sanity_plot(
    row: pd.Series,
    lam: np.ndarray,
    g_dft: np.ndarray,
    g_eqv2: np.ndarray,
    freqs: np.ndarray,
    mag_dft: np.ndarray,
    mag_eqv2: np.ndarray,
    cutoff: float,
    out_dir: Path,
    model_label: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    color = model_color(model_label)
    ax = axes[0]
    sns.lineplot(x=lam, y=g_dft, ax=ax, color=DFT_COLOR, label="DFT")
    sns.lineplot(x=lam, y=g_eqv2, ax=ax, color=color, label=model_label)
    ax.set_xlabel(r"$\lambda$ [$\AA$]")
    ax.set_ylabel(r"$d \cdot F$ [eV/$\AA$]")
    ax.set_title(f"Single-line sanity: g{int(row.geom_rank):04d} d{int(row.direction_id)}")
    finish_axis(ax, legend=True)

    ax = axes[1]
    sns.lineplot(x=freqs, y=mag_dft + 1e-30, ax=ax, color=DFT_COLOR, label="DFT")
    sns.lineplot(x=freqs, y=mag_eqv2 + 1e-30, ax=ax, color=color, label=model_label)
    ax.axvline(cutoff, color=GUIDE_COLOR, linestyle="--", linewidth=1.2, label=f"{cutoff:g} cyc/Å")
    ax.set_yscale("log")
    ax.set_xlabel(r"spatial frequency [cycles/$\AA$]")
    ax.set_ylabel(r"$|\mathrm{FFT}(d \cdot F)|$")
    ax.set_title("Same detrending/windowing")
    finish_axis(ax, legend=True)

    fig.tight_layout(pad=0.01)
    path = out_dir / f"{model_slug(model_label)}_single_line_sanity.png"
    fig.savefig(path, dpi=180)
    print(f"Saved {path}")
    plt.close(fig)
    return path


def save_median_spectra(
    freqs: np.ndarray,
    dft_mags: np.ndarray,
    eqv2_mags: np.ndarray,
    cutoff: float,
    out_dir: Path,
    model_label: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mags, color, label in [(dft_mags, DFT_COLOR, "DFT"), (eqv2_mags, model_color(model_label), model_label)]:
        median = np.median(mags, axis=0)
        q25 = np.quantile(mags, 0.25, axis=0)
        q75 = np.quantile(mags, 0.75, axis=0)
        ax.plot(freqs, median + 1e-30, color=color, label=label)
        ax.fill_between(freqs, q25 + 1e-30, q75 + 1e-30, color=color, alpha=0.18)
    ax.axvline(cutoff, color=GUIDE_COLOR, linestyle="--", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e0)
    ax.set_xlabel(r"spatial frequency [cycles/$\AA$]")
    ax.set_ylabel(r"$|\mathrm{FFT}(d \cdot F)|$")
    # ax.set_title("Median force spectrum across 200 lines")
    ax.set_xlim(0, 100)
    finish_axis(ax, legend=True)
    fig.tight_layout(pad=0.01)
    path = out_dir / f"{model_slug(model_label)}_median_force_spectra.png"
    fig.savefig(path, dpi=180)
    print(f"Saved {path}")
    plt.close(fig)
    return path


def save_hf_scatter(summary: pd.DataFrame, out_dir: Path, model_label: str) -> Path:
    eps = 1e-16
    x = summary["hf_fraction_dft"].to_numpy(dtype=float) + eps
    y = summary["hf_fraction_eqv2"].to_numpy(dtype=float) + eps
    ratio = y / x
    n = len(summary)
    n_above = int(np.sum(ratio > 1.0))
    n_10x = int(np.sum(ratio > 10.0))
    n_100x = int(np.sum(ratio > 100.0))
    median_excess = float(summary["hf_excess_log10"].median())

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    categories = [
        (ratio <= 1.0, f"{model_label} <= DFT ({n - n_above})", "#B8B8B8"),
        ((ratio > 1.0) & (ratio <= 10.0), f"1-10x excess ({n_above - n_10x})", AD_COLOR),
        ((ratio > 10.0) & (ratio <= 100.0), f"10-100x excess ({n_10x - n_100x})", HIP_COLOR),
        (ratio > 100.0, f">100x excess ({n_100x})", "#B75DAE"),
    ]
    for mask, label, color in categories:
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask],
            y[mask],
            label=label,
            color=color,
            s=38,
            edgecolor="k",
            linewidth=0.25,
            alpha=0.85,
        )

    lo = min(summary["hf_fraction_dft"].min(), summary["hf_fraction_eqv2"].min()) + eps
    hi = max(summary["hf_fraction_dft"].max(), summary["hf_fraction_eqv2"].max()) + eps
    lo = 10 ** np.floor(np.log10(lo))
    hi = 10 ** np.ceil(np.log10(hi))
    guide_x = np.logspace(np.log10(lo), np.log10(hi), 200)
    for mult, label, linestyle in [(1.0, "parity", "--"), (10.0, "10x", ":"), (100.0, "100x", "-.")]:
        guide_y = mult * guide_x
        mask = guide_y <= hi
        ax.plot(guide_x[mask], guide_y[mask], color=GUIDE_COLOR, linestyle=linestyle, linewidth=1.2)
        if np.any(mask):
            ax.text(
                guide_x[mask][-1],
                guide_y[mask][-1],
                f" {label}",
                color=GUIDE_COLOR,
                fontsize=9,
                va="center",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("DFT HF fraction")
    ax.set_ylabel(f"{model_label} HF fraction")
    # ax.set_title(f"{model_label} high-frequency force content")
    ax.text(
        0.04,
        0.96,
        (
            f"median excess = {median_excess:+.2f} log10\n"
            f"{n_above}/{n} above parity\n"
            f"{n_10x}/{n} >10x, {n_100x}/{n} >100x"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, boxstyle="round,pad=0.35"),
    )
    finish_axis(ax, legend=True)
    fig.tight_layout(pad=0.01)
    path = out_dir / f"{model_slug(model_label)}_vs_dft_hf_fraction.png"
    fig.savefig(path, dpi=180)
    print(f"Saved {path}")
    plt.close(fig)
    return path


def save_hf_excess_hist(summary: pd.DataFrame, out_dir: Path, model_label: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.histplot(summary, x="hf_excess_log10", bins=30, ax=axes[0], color=AD_COLOR)
    axes[0].axvline(0.0, color=GUIDE_COLOR, linestyle="--", linewidth=1.2)
    axes[0].set_xlabel(rf"$\log_{{10}}(\mathrm{{HF}}_{{{model_label}}}+\epsilon)-\log_{{10}}(\mathrm{{HF}}_{{DFT}}+\epsilon)$")
    axes[0].set_title("HF excess distribution")
    finish_axis(axes[0])

    sns.scatterplot(
        data=summary,
        x="hf_excess_log10",
        y="projected_force_mae_ev_ang",
        hue="direction_id",
        palette=[AD_COLOR, HIP_COLOR],
        ax=axes[1],
        s=42,
        edgecolor="k",
        linewidth=0.25,
    )
    axes[1].set_xlabel("HF excess [log10]")
    axes[1].set_ylabel(r"projected force MAE [eV/$\AA$]")
    axes[1].set_title("Excess roughness vs line force error")
    finish_axis(axes[1], legend=True)

    fig.tight_layout(pad=0.01)
    path = out_dir / f"{model_slug(model_label)}_hf_excess_summary.png"
    fig.savefig(path, dpi=180)
    print(f"Saved {path}")
    plt.close(fig)
    return path


def maybe_join_hessian_metrics(summary: pd.DataFrame, metrics_csv: Path | None, out_dir: Path) -> tuple[pd.DataFrame, Path | None]:
    if metrics_csv is None or not metrics_csv.exists():
        return summary, None
    metrics = pd.read_csv(metrics_csv)
    if "dataset_idx" not in metrics:
        return summary, None
    keep = ["dataset_idx"] + [
        col
        for col in ["hessian_error", "hessian_fro_norm", "forces_error", "hessian_model_fro_norm", "hessian_true_fro_norm"]
        if col in metrics
    ]
    merged = summary.merge(metrics[keep].drop_duplicates("dataset_idx"), on="dataset_idx", how="left")
    corr_rows = []
    for target in [col for col in keep if col != "dataset_idx"]:
        valid = merged[["hf_excess_log10", "hf_fraction_eqv2", "hf_fraction_dft", target]].dropna()
        if len(valid) < 3:
            continue
        corr_rows.append(
            {
                "target": target,
                "n": int(len(valid)),
                "spearman_hf_excess": float(valid["hf_excess_log10"].corr(valid[target], method="spearman")),
                "spearman_hf_eqv2": float(valid["hf_fraction_eqv2"].corr(valid[target], method="spearman")),
                "spearman_hf_dft": float(valid["hf_fraction_dft"].corr(valid[target], method="spearman")),
            }
        )
    corr_path = out_dir / "hf_vs_hessian_metric_correlations.csv"
    pd.DataFrame(corr_rows).to_csv(corr_path, index=False)
    return merged, corr_path


def write_report(
    summary: pd.DataFrame,
    out_dir: Path,
    cutoff: float,
    written: list[Path],
    corr_path: Path | None,
    model_label: str,
) -> Path:
    median_excess = float(summary["hf_excess_log10"].median())
    p90_excess = float(summary["hf_excess_log10"].quantile(0.90))
    frac_eq_gt_dft = float(np.mean(summary["hf_fraction_eqv2"] > summary["hf_fraction_dft"]))
    text = f"""# T1x-Val Force Spectra

Analyzed `{len(summary)}` geometry/direction lines for `{model_label}` with high-frequency cutoff `{cutoff:g}` cycles/Angstrom.

## Summary

- Median DFT HF fraction: `{summary["hf_fraction_dft"].median():.4g}`
- Median {model_label} HF fraction: `{summary["hf_fraction_eqv2"].median():.4g}`
- Median HF excess log10({model_label}/DFT): `{median_excess:.4g}`
- P90 HF excess log10({model_label}/DFT): `{p90_excess:.4g}`
- Fraction of lines with {model_label} HF fraction > DFT HF fraction: `{frac_eq_gt_dft:.1%}`
- Median all-component force MAE: `{summary["force_mae_ev_ang"].median():.4g}` eV/Angstrom
- Median projected-force MAE: `{summary["projected_force_mae_ev_ang"].median():.4g}` eV/Angstrom
- Median DFT `d·F` vs `-dE/dlambda` MAE: `{summary["dft_energy_force_consistency_mae"].median():.4g}` eV/Angstrom
- Median {model_label} `d·F` vs `-dE/dlambda` MAE: `{summary["eqv2_energy_force_consistency_mae"].median():.4g}` eV/Angstrom

## Files

"""
    for path in written:
        text += f"- `{path}`\n"
    if corr_path is not None:
        text += f"- `{corr_path}`\n"
    report_path = out_dir / "force_spectra_report.md"
    report_path.write_text(text)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=DEFAULT_SCAN_DIR,
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--model-force-dir",
        type=Path,
        default=None,
        help="Directory with model force outputs. Defaults to SCAN_DIR/eqv2_force_outputs.",
    )
    parser.add_argument("--model-label", default="EqV2")
    parser.add_argument("--cutoff", type=float, default=20.0)
    parser.add_argument("--detrend-degree", type=int, default=3)
    parser.add_argument(
        "--hessian-metrics-csv",
        type=Path,
        default=project_root() / "results_evalhorm" / "eqv2_ts1x-val_autograd_metrics.csv",
    )
    args = parser.parse_args()

    scan_dir = args.scan_dir
    out_dir = args.out_dir or scan_dir / "force_spectra_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(scan_dir / "scan_manifest.csv")
    line_meta = (
        manifest.groupby(["geom_rank", "dataset_idx", "direction_id"], as_index=False)
        .agg(
            n_atoms=("n_atoms", "first"),
            n_points=("point_id", "count"),
            direction_path=("direction_path", "first"),
        )
        .sort_values(["geom_rank", "direction_id"])
    )

    dft_dir = scan_dir / "dft_force_outputs" / "line_npz"
    model_force_dir = args.model_force_dir or scan_dir / "eqv2_force_outputs"
    eqv2_dir = model_force_dir / "line_npz"
    rows: list[dict[str, object]] = []
    dft_mags: list[np.ndarray] = []
    eqv2_mags: list[np.ndarray] = []
    freq_ref: np.ndarray | None = None
    sanity_payload = None

    for row in line_meta.itertuples(index=False):
        dft_path = line_npz_path(dft_dir, int(row.geom_rank), int(row.dataset_idx), int(row.direction_id))
        eqv2_path = line_npz_path(eqv2_dir, int(row.geom_rank), int(row.dataset_idx), int(row.direction_id))
        if not dft_path.exists():
            raise FileNotFoundError(dft_path)
        if not eqv2_path.exists():
            raise FileNotFoundError(eqv2_path)
        direction_path = resolve_existing(row.direction_path, scan_dir)
        direction = np.load(direction_path)

        dft = np.load(dft_path)
        eqv2 = np.load(eqv2_path)
        lam = np.asarray(dft["lambda_ang"], dtype=np.float64)
        if not np.allclose(lam, np.asarray(eqv2["lambda_ang"], dtype=np.float64)):
            raise ValueError(f"lambda grid mismatch for {dft_path.name}")
        dlam = float(lam[1] - lam[0])

        f_dft = np.asarray(dft["forces_ev_ang"], dtype=np.float64)
        f_eqv2 = np.asarray(eqv2["forces_ev_ang"], dtype=np.float64)
        g_dft = projected_force(f_dft, direction)
        g_eqv2 = projected_force(f_eqv2, direction)
        dft_minus_dedlam = -np.gradient(np.asarray(dft["energy_hartree"], dtype=np.float64) * HARTREE_TO_EV, dlam)
        eqv2_minus_dedlam = -np.gradient(np.asarray(eqv2["energy_ev"], dtype=np.float64), dlam)
        freqs, mag_dft, _ = detrended_spectrum(g_dft, dlam, args.detrend_degree)
        freqs_eq, mag_eqv2, _ = detrended_spectrum(g_eqv2, dlam, args.detrend_degree)
        if not np.allclose(freqs, freqs_eq):
            raise ValueError(f"frequency grid mismatch for {dft_path.name}")
        if freq_ref is None:
            freq_ref = freqs
        elif not np.allclose(freq_ref, freqs):
            raise ValueError("frequency grid changed between lines")

        hf_dft = high_frequency_fraction(freqs, mag_dft, args.cutoff)
        hf_eqv2 = high_frequency_fraction(freqs, mag_eqv2, args.cutoff)
        eps = 1e-16
        diff = f_eqv2 - f_dft
        rows.append(
            {
                "geom_rank": int(row.geom_rank),
                "dataset_idx": int(row.dataset_idx),
                "direction_id": int(row.direction_id),
                "n_atoms": int(row.n_atoms),
                "n_points": int(row.n_points),
                "hf_fraction_dft": hf_dft,
                "hf_fraction_eqv2": hf_eqv2,
                "hf_excess_log10": float(np.log10(hf_eqv2 + eps) - np.log10(hf_dft + eps)),
                "projected_force_mae_ev_ang": float(np.mean(np.abs(g_eqv2 - g_dft))),
                "projected_force_rmse_ev_ang": float(np.sqrt(np.mean((g_eqv2 - g_dft) ** 2))),
                "force_mae_ev_ang": float(np.mean(np.abs(diff))),
                "force_rmse_ev_ang": float(np.sqrt(np.mean(diff**2))),
                "dft_energy_force_consistency_mae": float(np.mean(np.abs(g_dft - dft_minus_dedlam))),
                "eqv2_energy_force_consistency_mae": float(np.mean(np.abs(g_eqv2 - eqv2_minus_dedlam))),
                "dft_line_npz_path": str(dft_path.resolve()),
                "eqv2_line_npz_path": str(eqv2_path.resolve()),
            }
        )
        dft_mags.append(mag_dft)
        eqv2_mags.append(mag_eqv2)
        if sanity_payload is None:
            sanity_payload = (pd.Series(rows[-1]), lam, g_dft, g_eqv2, freqs, mag_dft, mag_eqv2)

    summary = pd.DataFrame(rows)
    summary, corr_path = maybe_join_hessian_metrics(summary, args.hessian_metrics_csv, out_dir)
    summary_path = out_dir / "force_spectra_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_parquet(out_dir / "force_spectra_summary.parquet", index=False)

    freq_arr = np.asarray(freq_ref, dtype=np.float64)
    dft_mag_arr = np.stack(dft_mags)
    eqv2_mag_arr = np.stack(eqv2_mags)
    np.savez_compressed(
        out_dir / "force_spectra_arrays.npz",
        freqs=freq_arr,
        dft_mag=dft_mag_arr,
        eqv2_mag=eqv2_mag_arr,
        geom_rank=summary["geom_rank"].to_numpy(dtype=np.int64),
        dataset_idx=summary["dataset_idx"].to_numpy(dtype=np.int64),
        direction_id=summary["direction_id"].to_numpy(dtype=np.int64),
    )

    written = [
        summary_path,
        out_dir / "force_spectra_arrays.npz",
        save_sanity_plot(*sanity_payload, cutoff=args.cutoff, out_dir=out_dir, model_label=args.model_label),
        save_median_spectra(freq_arr, dft_mag_arr, eqv2_mag_arr, args.cutoff, out_dir, args.model_label),
        save_hf_scatter(summary, out_dir, args.model_label),
        save_hf_excess_hist(summary, out_dir, args.model_label),
    ]
    report_path = write_report(summary, out_dir, args.cutoff, written, corr_path, args.model_label)

    print(f"Wrote summary rows: {len(summary)}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(
        "Median HF fractions: "
        f"DFT={summary['hf_fraction_dft'].median():.4g}, "
        f"{args.model_label}={summary['hf_fraction_eqv2'].median():.4g}, "
        f"excess={summary['hf_excess_log10'].median():+.3f} log10"
    )


if __name__ == "__main__":
    main()
