#!/usr/bin/env python
"""Aggregate Hessian smoothness GPU chunk outputs into summary tables and plots."""
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
    "eqv2_auto_vs_fd_force_mae",
    "eqv2_auto_vs_fd_energy_mae",
    "eqv2_fd_force_vs_fd_energy_mae",
    "hip_pred_vs_eqv2_fd_force_mae",
    "eqv2_auto_vs_true_abs",
    "hip_pred_vs_true_abs",
    "eqv2_auto_roughness",
    "hip_pred_roughness",
    "eqv2_edge_count_range",
    "hip_edge_count_range",
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
        values = df[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        rows.append(
            {
                "metric": metric,
                "n": int(finite.size),
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "p90": float(np.quantile(finite, 0.90)),
                "p95": float(np.quantile(finite, 0.95)),
                "max": float(np.max(finite)),
            }
        )
    return pd.DataFrame(rows)


def finite_fraction(df: pd.DataFrame, column: str, threshold: float) -> float:
    values = df[column].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite <= threshold))


def build_conclusions(metrics_df: pd.DataFrame, fd_df: pd.DataFrame) -> dict[str, Any]:
    auto_force = metrics_df["eqv2_auto_vs_fd_force_mae"].to_numpy(float)
    auto_energy = metrics_df["eqv2_auto_vs_fd_energy_mae"].to_numpy(float)
    force_energy = metrics_df["eqv2_fd_force_vs_fd_energy_mae"].to_numpy(float)
    eqv2_true = metrics_df["eqv2_auto_vs_true_abs"].to_numpy(float)
    hip_true = metrics_df["hip_pred_vs_true_abs"].to_numpy(float)
    eqv2_edges = metrics_df["eqv2_edge_count_range"].to_numpy(float)
    hip_edges = metrics_df["hip_edge_count_range"].to_numpy(float)

    finite_auto_force = auto_force[np.isfinite(auto_force)]
    finite_auto_energy = auto_energy[np.isfinite(auto_energy)]
    finite_force_energy = force_energy[np.isfinite(force_energy)]
    finite_eqv2_true = eqv2_true[np.isfinite(eqv2_true)]
    finite_hip_true = hip_true[np.isfinite(hip_true)]

    fd_by_eps = (
        fd_df.groupby("eps", as_index=False)
        .agg(
            fd_energy_curvature_mean=("fd_energy_curvature", "mean"),
            fd_force_curvature_mean=("fd_force_curvature", "mean"),
            eqv2_auto_vhv_mean=("eqv2_auto_vhv", "mean"),
            hip_pred_vhv_mean=("hip_pred_vhv", "mean"),
            fd_energy_curvature_std=("fd_energy_curvature", "std"),
            fd_force_curvature_std=("fd_force_curvature", "std"),
        )
        .sort_values("eps")
    )

    return {
        "n_lines": int(len(metrics_df)),
        "n_unique_structures": int(metrics_df["dataset_idx"].nunique()),
        "n_fd_rows": int(len(fd_df)),
        "eqv2_autograd_matches_force_fd_better_than_energy_fd_fraction": float(
            np.mean(finite_auto_force < finite_auto_energy)
        ),
        "eqv2_force_fd_energy_fd_median_gap": float(np.median(finite_force_energy)),
        "hip_better_than_eqv2_vs_reference_fraction": float(
            np.mean(finite_hip_true < finite_eqv2_true)
        ),
        "hip_vs_eqv2_reference_median_error_ratio": float(
            np.median(finite_hip_true / (finite_eqv2_true + 1e-12))
        ),
        "eqv2_any_neighbor_changes_fraction": float(np.mean(eqv2_edges > 0)),
        "hip_any_neighbor_changes_fraction": float(np.mean(hip_edges > 0)),
        "eqv2_edge_count_range_max": float(np.nanmax(eqv2_edges)),
        "hip_edge_count_range_max": float(np.nanmax(hip_edges)),
        "eqv2_min_cutoff_margin_below_0p02_fraction": finite_fraction(
            metrics_df, "eqv2_min_cutoff_margin_min", 0.02
        ),
        "hip_min_cutoff_margin_below_0p02_fraction": finite_fraction(
            metrics_df, "hip_min_cutoff_margin_min", 0.02
        ),
        "fd_by_eps": fd_by_eps.to_dict(orient="records"),
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
    ax.scatter(
        np.concatenate([np.full(values.size, i + 1) for i, values in enumerate(data)]),
        np.concatenate(data),
        alpha=0.12,
        s=8,
        color="black",
    )
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_fd_eps_plot(fd_df: pd.DataFrame, path: Path) -> None:
    grouped = (
        fd_df.groupby("eps", as_index=False)
        .agg(
            fd_energy_median=("fd_energy_curvature", "median"),
            fd_force_median=("fd_force_curvature", "median"),
            auto_median=("eqv2_auto_vhv", "median"),
            hip_median=("hip_pred_vhv", "median"),
            fd_energy_p25=("fd_energy_curvature", lambda x: np.quantile(x, 0.25)),
            fd_energy_p75=("fd_energy_curvature", lambda x: np.quantile(x, 0.75)),
            fd_force_p25=("fd_force_curvature", lambda x: np.quantile(x, 0.25)),
            fd_force_p75=("fd_force_curvature", lambda x: np.quantile(x, 0.75)),
        )
        .sort_values("eps")
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(grouped["eps"], grouped["fd_energy_median"], marker="o", label="FD energy")
    ax.plot(grouped["eps"], grouped["fd_force_median"], marker="o", label="FD force")
    ax.plot(grouped["eps"], grouped["auto_median"], marker="o", label="AD")
    ax.plot(grouped["eps"], grouped["hip_median"], marker="o", label="HIP predicted")
    ax.fill_between(
        grouped["eps"],
        grouped["fd_energy_p25"],
        grouped["fd_energy_p75"],
        alpha=0.18,
    )
    ax.fill_between(
        grouped["eps"],
        grouped["fd_force_p25"],
        grouped["fd_force_p75"],
        alpha=0.18,
    )
    ax.set_xscale("log")
    ax.set_xlabel("finite-difference epsilon / Angstrom")
    ax.set_ylabel("directional curvature")
    ax.set_title("Median directional curvature vs finite-difference step")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_neighbor_plot(metrics_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].hist(metrics_df["eqv2_min_cutoff_margin_min"], bins=30, alpha=0.8, label="EQV2")
    axes[0].hist(metrics_df["hip_min_cutoff_margin_min"], bins=30, alpha=0.6, label="HIP")
    axes[0].axvline(0.02, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("minimum cutoff margin / Angstrom")
    axes[0].set_ylabel("line count")
    axes[0].set_title("Closest approach to cutoff")
    axes[0].legend()

    axes[1].hist(metrics_df["eqv2_edge_count_range"], bins=20, alpha=0.8, label="EQV2")
    axes[1].hist(metrics_df["hip_edge_count_range"], bins=20, alpha=0.6, label="HIP")
    axes[1].set_xlabel("edge count range along line")
    axes[1].set_title("Neighbor-list changes")
    axes[1].legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_markdown_report(
    path: Path,
    conclusions: dict[str, Any],
    metric_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    lookup = metric_summary.set_index("metric")

    def median(metric: str) -> float:
        return float(lookup.loc[metric, "median"])

    text = f"""# Hessian Smoothness GPU Diagnostic

Aggregated `{conclusions["n_lines"]}` displacement lines from `{conclusions["n_unique_structures"]}` structures.

## Main Result

AD directional curvature matches finite differences of the EQV2 force field better than finite differences of the EQV2 energy in `{conclusions["eqv2_autograd_matches_force_fd_better_than_energy_fd_fraction"]:.1%}` of lines.

- Median `|AD - FD force|`: `{median("eqv2_auto_vs_fd_force_mae"):.4g}`
- Median `|AD - FD energy|`: `{median("eqv2_auto_vs_fd_energy_mae"):.4g}`
- Median `|FD force - FD energy|`: `{median("eqv2_fd_force_vs_fd_energy_mae"):.4g}`

This supports the interpretation that the direct EQV2 force head is not conservative with respect to the EQV2 energy head, rather than the issue being dominated by cutoff nonsmoothness.

## Reference Hessian

HIP is closer to the reference directional curvature in `{conclusions["hip_better_than_eqv2_vs_reference_fraction"]:.1%}` of lines.

- Median `|AD - reference|`: `{median("eqv2_auto_vs_true_abs"):.4g}`
- Median `|HIP predicted - reference|`: `{median("hip_pred_vs_true_abs"):.4g}`
- Median HIP/EQV2 reference-error ratio: `{conclusions["hip_vs_eqv2_reference_median_error_ratio"]:.4g}`

## Neighbor Lists

EQV2 edge-count changes occurred in `{conclusions["eqv2_any_neighbor_changes_fraction"]:.1%}` of lines; HIP edge-count changes occurred in `{conclusions["hip_any_neighbor_changes_fraction"]:.1%}` of lines.

Plots and combined tables are in `{output_dir}`.
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root() / "runs" / "hessian_smoothness_gpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "runs" / "hessian_smoothness_gpu_aggregate",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = read_chunks(args.input_dir, "line_metrics.csv")
    line_df = read_chunks(args.input_dir, "line_scan.csv")
    fd_df = read_chunks(args.input_dir, "fd_epsilon_sweep.csv")

    metrics_df.to_csv(args.output_dir / "line_metrics_all.csv", index=False)
    line_df.to_csv(args.output_dir / "line_scan_all.csv", index=False)
    fd_df.to_csv(args.output_dir / "fd_epsilon_sweep_all.csv", index=False)
    metrics_df.to_parquet(args.output_dir / "line_metrics_all.parquet", index=False)
    line_df.to_parquet(args.output_dir / "line_scan_all.parquet", index=False)
    fd_df.to_parquet(args.output_dir / "fd_epsilon_sweep_all.parquet", index=False)

    metric_summary = numeric_summary(metrics_df, KEY_METRICS)
    metric_summary.to_csv(args.output_dir / "metric_summary.csv", index=False)

    conclusions = build_conclusions(metrics_df, fd_df)
    (args.output_dir / "aggregate_summary.json").write_text(json.dumps(conclusions, indent=2))
    write_markdown_report(
        args.output_dir / "report.md",
        conclusions,
        metric_summary,
        args.output_dir,
    )

    save_boxplot(
        metrics_df,
        [
            "eqv2_auto_vs_fd_force_mae",
            "eqv2_auto_vs_fd_energy_mae",
            "eqv2_fd_force_vs_fd_energy_mae",
        ],
        ["auto vs\nFD force", "auto vs\nFD energy", "FD force vs\nFD energy"],
        "absolute directional-curvature error",
        "EQV2 force/energy consistency",
        args.output_dir / "eqv2_force_energy_consistency.png",
    )
    save_boxplot(
        metrics_df,
        ["eqv2_auto_vs_true_abs", "hip_pred_vs_true_abs"],
        ["AD", "HIP predicted"],
        "absolute directional-curvature error vs reference",
        "Reference Hessian directional-curvature error",
        args.output_dir / "reference_error.png",
    )
    save_boxplot(
        metrics_df,
        ["eqv2_auto_roughness", "hip_pred_roughness"],
        ["AD", "HIP predicted"],
        "roughness",
        "Curvature roughness along displacement lines",
        args.output_dir / "curvature_roughness.png",
    )
    save_fd_eps_plot(fd_df, args.output_dir / "fd_epsilon_sensitivity.png")
    save_neighbor_plot(metrics_df, args.output_dir / "neighbor_list_diagnostics.png")

    print(f"Wrote aggregate outputs to {args.output_dir}")
    print(f"Report: {args.output_dir / 'report.md'}")
    print(f"Summary: {args.output_dir / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
