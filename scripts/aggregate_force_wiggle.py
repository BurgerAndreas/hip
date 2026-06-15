#!/usr/bin/env python
"""Aggregate force-wiggle diagnostic chunks into tables, plots, and a report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


KEY_METRICS = [
    "line_fd_force_curvature_std",
    "line_fd_energy_curvature_std",
    "line_force_energy_curvature_mae",
    "eps_slope_rel_range",
    "eps_slope_median_abs_autograd_error",
    "force_parallel_residual_rms_norm",
    "force_parallel_second_diff_rms",
    "fft_total_power",
    "fft_high_freq_power_fraction",
    "fft_spectral_centroid",
    "edge_count_range",
    "abs_autograd_minus_true",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_chunks(input_dir: Path, filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(input_dir.glob(f"chunk_*/{filename}")):
        frame = pd.read_csv(path)
        frame.insert(0, "chunk", path.parent.name)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No chunk files matching {input_dir}/chunk_*/{filename}")
    return pd.concat(frames, ignore_index=True)


def numeric_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in df:
            continue
        values = df[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if values.size == 0:
            continue
        rows.append(
            {
                "metric": metric,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def build_conclusions(metrics_df: pd.DataFrame, slopes_df: pd.DataFrame) -> dict[str, Any]:
    force_std = metrics_df["line_fd_force_curvature_std"].to_numpy(float)
    energy_std = metrics_df["line_fd_energy_curvature_std"].to_numpy(float)
    eps_rel_range = metrics_df["eps_slope_rel_range"].to_numpy(float)
    residual_norm = metrics_df["force_parallel_residual_rms_norm"].to_numpy(float)
    edge_range = metrics_df["edge_count_range"].to_numpy(float)

    slopes_by_eps = (
        slopes_df.groupby("eps", as_index=False)
        .agg(
            fd_force_slope_median=("fd_force_slope", "median"),
            fd_force_slope_std=("fd_force_slope", "std"),
            autograd_vhv_median=("autograd_vhv", "median"),
            abs_autograd_minus_fd_force_median=("abs_autograd_minus_fd_force", "median"),
            fd_energy_curvature_median=("fd_energy_curvature", "median"),
        )
        .sort_values("eps")
    )

    return {
        "n_lines": int(len(metrics_df)),
        "n_unique_structures": int(metrics_df["dataset_idx"].nunique()),
        "n_slope_rows": int(len(slopes_df)),
        "force_curvature_std_greater_than_energy_fraction": float(np.mean(force_std > energy_std)),
        "median_force_curvature_std": float(np.median(force_std)),
        "median_energy_curvature_std": float(np.median(energy_std)),
        "median_force_energy_curvature_mae": float(
            np.median(metrics_df["line_force_energy_curvature_mae"].to_numpy(float))
        ),
        "median_eps_slope_rel_range": float(np.median(eps_rel_range)),
        "p95_eps_slope_rel_range": float(np.quantile(eps_rel_range, 0.95)),
        "median_force_residual_rms_norm": float(np.median(residual_norm)),
        "p95_force_residual_rms_norm": float(np.quantile(residual_norm, 0.95)),
        "median_high_freq_power_fraction": float(
            np.median(metrics_df["fft_high_freq_power_fraction"].to_numpy(float))
        ),
        "edge_count_change_fraction": float(np.mean(edge_range > 0)),
        "max_edge_count_range": float(np.max(edge_range)),
        "slopes_by_eps": slopes_by_eps.to_dict(orient="records"),
    }


def save_boxplot(
    df: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    ylabel: str,
    title: str,
    path: Path,
    logy: bool = True,
) -> None:
    data = [df[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float) for col in columns]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    for i, values in enumerate(data):
        ax.scatter(np.full(values.size, i + 1), values, alpha=0.12, s=8, color="black")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_slope_eps_plot(slopes_df: pd.DataFrame, path: Path) -> None:
    grouped = (
        slopes_df.groupby("eps", as_index=False)
        .agg(
            fd_force_median=("fd_force_slope", "median"),
            fd_force_p25=("fd_force_slope", lambda x: np.quantile(x, 0.25)),
            fd_force_p75=("fd_force_slope", lambda x: np.quantile(x, 0.75)),
            auto_median=("autograd_vhv", "median"),
            energy_median=("fd_energy_curvature", "median"),
            abs_err_median=("abs_autograd_minus_fd_force", "median"),
        )
        .sort_values("eps")
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(grouped["eps"], grouped["fd_force_median"], marker="o", label="FD force slope")
    axes[0].plot(grouped["eps"], grouped["auto_median"], marker="o", label="autograd vHv")
    axes[0].plot(grouped["eps"], grouped["energy_median"], marker="o", label="FD energy curvature")
    axes[0].fill_between(grouped["eps"], grouped["fd_force_p25"], grouped["fd_force_p75"], alpha=0.18)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("epsilon / Angstrom")
    axes[0].set_ylabel("directional curvature")
    axes[0].set_title("Local force-slope epsilon sweep")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(grouped["eps"], grouped["abs_err_median"], marker="o")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epsilon / Angstrom")
    axes[1].set_ylabel("median |autograd - FD force slope|")
    axes[1].set_title("Autograd-vs-slope error")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_spectrum_plot(metrics_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(metrics_df["force_parallel_residual_rms_norm"], bins=40)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("detrended F.v residual RMS / std(F.v)")
    axes[0].set_ylabel("line count")
    axes[0].set_title("Force residual amplitude")
    axes[1].hist(metrics_df["fft_high_freq_power_fraction"], bins=40)
    axes[1].set_xlabel("high-frequency power fraction")
    axes[1].set_title("Detrended force spectrum")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_curvature_scatter(metrics_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].scatter(
        metrics_df["line_fd_energy_curvature_std"],
        metrics_df["line_fd_force_curvature_std"],
        alpha=0.5,
        s=14,
    )
    lim = max(
        float(metrics_df["line_fd_energy_curvature_std"].max()),
        float(metrics_df["line_fd_force_curvature_std"].max()),
    )
    axes[0].plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("std d2E/dt2")
    axes[0].set_ylabel("std -d(F.v)/dt")
    axes[0].set_title("Curvature variability along lines")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(
        metrics_df["force_parallel_residual_rms_norm"],
        metrics_df["eps_slope_rel_range"],
        alpha=0.5,
        s=14,
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("force residual RMS norm")
    axes[1].set_ylabel("epsilon slope relative range")
    axes[1].set_title("Wiggle amplitude vs local slope instability")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    path: Path,
    conclusions: dict[str, Any],
    metric_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    lookup = metric_summary.set_index("metric")

    def median(metric: str) -> float:
        return float(lookup.loc[metric, "median"])

    text = f"""# Force Wiggle Diagnostic

Aggregated `{conclusions["n_lines"]}` displacement lines from `{conclusions["n_unique_structures"]}` structures.

## Force Line Scans

- Median std of force-derived curvature `-d(F.v)/dt`: `{median("line_fd_force_curvature_std"):.4g}`
- Median std of energy-derived curvature `d2E/dt2`: `{median("line_fd_energy_curvature_std"):.4g}`
- Fraction where force curvature is more variable than energy curvature: `{conclusions["force_curvature_std_greater_than_energy_fraction"]:.1%}`
- Median `|force curvature - energy curvature|`: `{conclusions["median_force_energy_curvature_mae"]:.4g}`

## Autograd Hessian vs Local Force Slope

- Median epsilon-sweep relative range of force slope: `{conclusions["median_eps_slope_rel_range"]:.4g}`
- P95 epsilon-sweep relative range of force slope: `{conclusions["p95_eps_slope_rel_range"]:.4g}`
- Median `|autograd vHv - FD force slope|`: `{median("eps_slope_median_abs_autograd_error"):.4g}`

## Directional Force Spectrum

- Median detrended force residual RMS / std(F.v): `{conclusions["median_force_residual_rms_norm"]:.4g}`
- P95 detrended force residual RMS / std(F.v): `{conclusions["p95_force_residual_rms_norm"]:.4g}`
- Median high-frequency power fraction of detrended force: `{conclusions["median_high_freq_power_fraction"]:.4g}`

## Neighbor Lists

Edge-count changes occurred in `{conclusions["edge_count_change_fraction"]:.1%}` of lines. Max edge-count range was `{conclusions["max_edge_count_range"]:.0f}`.

Combined tables and plots are in `{output_dir}`.
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=project_root() / "runs" / "force_wiggle_gpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "runs" / "force_wiggle_gpu_aggregate",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = read_chunks(args.input_dir, "force_line_metrics.csv")
    line_df = read_chunks(args.input_dir, "force_line_scan.csv")
    slopes_df = read_chunks(args.input_dir, "force_slope_sweep.csv")

    metrics_df.to_csv(args.output_dir / "force_line_metrics_all.csv", index=False)
    line_df.to_csv(args.output_dir / "force_line_scan_all.csv", index=False)
    slopes_df.to_csv(args.output_dir / "force_slope_sweep_all.csv", index=False)
    metrics_df.to_parquet(args.output_dir / "force_line_metrics_all.parquet", index=False)
    line_df.to_parquet(args.output_dir / "force_line_scan_all.parquet", index=False)
    slopes_df.to_parquet(args.output_dir / "force_slope_sweep_all.parquet", index=False)

    metric_summary = numeric_summary(metrics_df, KEY_METRICS)
    metric_summary.to_csv(args.output_dir / "metric_summary.csv", index=False)
    conclusions = build_conclusions(metrics_df, slopes_df)
    (args.output_dir / "aggregate_summary.json").write_text(json.dumps(conclusions, indent=2))
    write_report(args.output_dir / "report.md", conclusions, metric_summary, args.output_dir)

    save_boxplot(
        metrics_df,
        ["line_fd_force_curvature_std", "line_fd_energy_curvature_std"],
        ["force-derived\ncurvature", "energy-derived\ncurvature"],
        "std along line",
        "Curvature variability along force line scans",
        args.output_dir / "curvature_variability.png",
    )
    save_boxplot(
        metrics_df,
        ["eps_slope_rel_range", "force_parallel_residual_rms_norm"],
        ["epsilon slope\nrelative range", "force residual\nRMS norm"],
        "dimensionless",
        "Force wiggle/stability metrics",
        args.output_dir / "force_wiggle_metrics.png",
    )
    save_slope_eps_plot(slopes_df, args.output_dir / "slope_epsilon_sensitivity.png")
    save_spectrum_plot(metrics_df, args.output_dir / "force_spectrum.png")
    save_curvature_scatter(metrics_df, args.output_dir / "curvature_scatter.png")

    print(f"Wrote aggregate outputs to {args.output_dir}")
    print(f"Report: {args.output_dir / 'report.md'}")
    print(f"Summary: {args.output_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
