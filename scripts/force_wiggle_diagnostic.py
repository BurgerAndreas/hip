#!/usr/bin/env python
"""Diagnose whether direct MLIP forces are wiggly along tiny geometry lines.

This is deliberately force-focused. For each sampled structure and random
direction v, the script:

1. Scans R(t) = R0 + t v and records E(t), F(t).v, force norms, and neighbor
   counts.
2. Compares center autograd v^T H v against finite-difference slopes of F.v
   over several eps values.
3. Computes an FFT spectrum of the detrended directional force trace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import radius_graph

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.ff_lmdb import LmdbDataset


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def model_metadata(checkpoint: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = ckpt.get("hyper_parameters", {}).get("model_config", {})
    keys = ["name", "direct_forces", "enable_hessian_head", "max_radius", "cutoff_hessian"]
    return {key: model_config.get(key) for key in keys}


def tensor_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def normalized_random_direction(
    rng: np.random.Generator,
    n_atoms: int,
    remove_translation: bool = True,
) -> np.ndarray:
    direction = rng.normal(size=(n_atoms, 3))
    if remove_translation:
        direction = direction - direction.mean(axis=0, keepdims=True)
    norm = float(np.linalg.norm(direction.reshape(-1)))
    if norm < 1e-12:
        return normalized_random_direction(rng, n_atoms, remove_translation)
    return direction / norm


def edge_count(coords: np.ndarray, cutoff: float, device: str) -> int:
    pos = torch.tensor(coords, dtype=torch.float32, device=device)
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
    with torch.no_grad():
        edges = radius_graph(pos, r=float(cutoff), batch=batch)
    return int(edges.shape[1])


def pairwise_cutoff_margin(coords: np.ndarray, cutoff: float) -> tuple[float, int]:
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
    if not np.any(mask):
        return float("nan"), 0
    margins = np.abs(distances[mask] - cutoff)
    return float(margins.min()), int(np.sum(margins < 0.02))


def directional_curvature(hessian: np.ndarray, direction: np.ndarray) -> float:
    flat = direction.reshape(-1)
    return float(flat @ hessian @ flat)


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def detrend(values: np.ndarray, x: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    degree = min(degree, max(0, values.size - 1))
    coeffs = np.polyfit(x, values, degree)
    trend = np.polyval(coeffs, x)
    return values - trend, trend


def fft_metrics(residual: np.ndarray, dt: float) -> dict[str, float]:
    centered = residual - np.mean(residual)
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(centered.size, d=dt)
    power = np.abs(spectrum) ** 2
    if power.size <= 1 or float(np.sum(power[1:])) <= 0.0:
        return {
            "fft_total_power": 0.0,
            "fft_high_freq_power_fraction": 0.0,
            "fft_spectral_centroid": 0.0,
            "fft_peak_frequency": 0.0,
        }

    nonzero_power = power[1:]
    nonzero_freqs = freqs[1:]
    high_start = max(1, int(np.ceil((power.size - 1) * 0.5)))
    high_power = float(np.sum(power[high_start:]))
    total_power = float(np.sum(nonzero_power))
    return {
        "fft_total_power": total_power,
        "fft_high_freq_power_fraction": high_power / total_power,
        "fft_spectral_centroid": float(np.sum(nonzero_freqs * nonzero_power) / total_power),
        "fft_peak_frequency": float(nonzero_freqs[int(np.argmax(nonzero_power))]),
    }


def roughness_metrics(values: np.ndarray, residual: np.ndarray, dt: float) -> dict[str, float]:
    diffs = np.diff(values)
    second = values[2:] - 2.0 * values[1:-1] + values[:-2]
    value_scale = float(np.std(values) + 1e-12)
    return {
        "force_parallel_std": float(np.std(values)),
        "force_parallel_total_variation": float(np.sum(np.abs(diffs))),
        "force_parallel_max_jump_per_ang": float(np.max(np.abs(diffs)) / dt),
        "force_parallel_second_diff_rms": float(np.sqrt(np.mean(second**2))) if second.size else 0.0,
        "force_parallel_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "force_parallel_residual_max_abs": float(np.max(np.abs(residual))),
        "force_parallel_residual_rms_norm": float(np.sqrt(np.mean(residual**2)) / value_scale),
    }


class ForceProbe:
    def __init__(self, checkpoint: Path, device: str):
        self.device = device
        self.calc = EquiformerTorchCalculator(
            checkpoint_path=str(checkpoint),
            hessian_method="autograd",
            device=device,
        )

    def energy_forces(self, coords: np.ndarray, atomic_nums: np.ndarray) -> tuple[float, np.ndarray]:
        out = self.calc.predict(
            coords=torch.tensor(coords, dtype=torch.float32, device=self.device),
            atomic_nums=torch.tensor(atomic_nums, dtype=torch.long, device=self.device),
            do_hessian=False,
        )
        return tensor_to_float(out["energy"]), out["forces"].reshape(-1, 3).detach().cpu().numpy()

    def autograd_vhv(self, coords: np.ndarray, atomic_nums: np.ndarray, direction: np.ndarray) -> float:
        n_atoms = int(atomic_nums.size)
        out = self.calc.predict(
            coords=torch.tensor(coords, dtype=torch.float32, device=self.device),
            atomic_nums=torch.tensor(atomic_nums, dtype=torch.long, device=self.device),
            hessian_method="autograd",
            do_hessian=True,
        )
        hessian = out["hessian"].reshape(3 * n_atoms, 3 * n_atoms).detach().cpu().numpy()
        return directional_curvature(symmetrize(hessian), direction)


def scan_line(
    probe: ForceProbe,
    coords: np.ndarray,
    atomic_nums: np.ndarray,
    direction: np.ndarray,
    t_values: np.ndarray,
    cutoff: float,
    device: str,
) -> pd.DataFrame:
    flat_direction = direction.reshape(-1)
    rows: list[dict[str, float]] = []
    for t in t_values:
        displaced = coords + float(t) * direction
        energy, forces = probe.energy_forces(displaced, atomic_nums)
        forces_flat = forces.reshape(-1)
        force_parallel = float(forces_flat @ flat_direction)
        force_parallel_vec = force_parallel * flat_direction
        force_perp = forces_flat - force_parallel_vec
        margin, near_count = pairwise_cutoff_margin(displaced, cutoff)
        rows.append(
            {
                "t": float(t),
                "energy": energy,
                "force_parallel": force_parallel,
                "force_norm": float(np.linalg.norm(forces_flat)),
                "force_perp_norm": float(np.linalg.norm(force_perp)),
                "max_atom_force_norm": float(np.linalg.norm(forces, axis=1).max()),
                "edge_count": edge_count(displaced, cutoff, device),
                "min_cutoff_margin": margin,
                "pairs_within_0p02_cutoff": near_count,
            }
        )
    return pd.DataFrame(rows)


def force_slope_sweep(
    probe: ForceProbe,
    coords: np.ndarray,
    atomic_nums: np.ndarray,
    direction: np.ndarray,
    eps_values: list[float],
    autograd_vhv: float,
    center_energy: float,
) -> list[dict[str, float]]:
    flat_direction = direction.reshape(-1)
    rows: list[dict[str, float]] = []
    for eps in eps_values:
        e_plus, f_plus = probe.energy_forces(coords + eps * direction, atomic_nums)
        e_minus, f_minus = probe.energy_forces(coords - eps * direction, atomic_nums)
        f_plus_proj = float(f_plus.reshape(-1) @ flat_direction)
        f_minus_proj = float(f_minus.reshape(-1) @ flat_direction)
        fd_force_slope = float(-(f_plus_proj - f_minus_proj) / (2.0 * eps))
        rows.append(
            {
                "eps": float(eps),
                "autograd_vhv": autograd_vhv,
                "fd_force_slope": fd_force_slope,
                "fd_energy_curvature": float((e_plus - 2.0 * center_energy + e_minus) / eps**2),
                "abs_autograd_minus_fd_force": abs(autograd_vhv - fd_force_slope),
            }
        )
    return rows


def finite_difference_curvature_from_line(line_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = line_df["t"].to_numpy(float)
    f = line_df["force_parallel"].to_numpy(float)
    e = line_df["energy"].to_numpy(float)
    dt = float(t[1] - t[0])
    force_curv = -(f[2:] - f[:-2]) / (2.0 * dt)
    energy_curv = (e[2:] - 2.0 * e[1:-1] + e[:-2]) / dt**2
    return force_curv, energy_curv


def line_metrics(
    line_df: pd.DataFrame,
    slope_rows: list[dict[str, float]],
    autograd_vhv: float,
    detrend_degree: int,
) -> dict[str, float]:
    t = line_df["t"].to_numpy(float)
    values = line_df["force_parallel"].to_numpy(float)
    dt = float(t[1] - t[0])
    residual, _trend = detrend(values, t, detrend_degree)
    force_curv, energy_curv = finite_difference_curvature_from_line(line_df)
    slopes = np.asarray([row["fd_force_slope"] for row in slope_rows], dtype=float)
    metrics = {
        "autograd_vhv": autograd_vhv,
        "line_fd_force_curvature_mean": float(np.mean(force_curv)),
        "line_fd_force_curvature_std": float(np.std(force_curv)),
        "line_fd_energy_curvature_mean": float(np.mean(energy_curv)),
        "line_fd_energy_curvature_std": float(np.std(energy_curv)),
        "line_force_energy_curvature_mae": float(np.mean(np.abs(force_curv - energy_curv))),
        "eps_slope_range": float(np.max(slopes) - np.min(slopes)),
        "eps_slope_std": float(np.std(slopes)),
        "eps_slope_rel_range": float((np.max(slopes) - np.min(slopes)) / (abs(np.median(slopes)) + 1e-12)),
        "eps_slope_median_abs_autograd_error": float(
            np.median([row["abs_autograd_minus_fd_force"] for row in slope_rows])
        ),
        "edge_count_range": float(line_df["edge_count"].max() - line_df["edge_count"].min()),
        "min_cutoff_margin_min": float(line_df["min_cutoff_margin"].min()),
    }
    metrics.update(roughness_metrics(values, residual, dt))
    metrics.update(fft_metrics(residual, dt))
    return metrics


def add_reference_metric(
    metrics: dict[str, float],
    data: Any,
    direction: np.ndarray,
    autograd_vhv: float,
) -> None:
    if not hasattr(data, "hessian"):
        return
    n_atoms = direction.shape[0]
    hessian = data.hessian.reshape(3 * n_atoms, 3 * n_atoms).detach().cpu().numpy()
    true_vhv = directional_curvature(symmetrize(hessian), direction)
    metrics["true_vhv"] = true_vhv
    metrics["abs_autograd_minus_true"] = abs(autograd_vhv - true_vhv)


def plot_line(line_df: pd.DataFrame, metrics: dict[str, float], path: Path) -> None:
    t = line_df["t"].to_numpy(float)
    force = line_df["force_parallel"].to_numpy(float)
    residual, trend = detrend(force, t, 3)
    force_curv, energy_curv = finite_difference_curvature_from_line(line_df)
    t_inner = t[1:-1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(t, line_df["energy"], marker="o")
    axes[0, 0].set_title("Energy")
    axes[0, 1].plot(t, force, marker="o", label="F.v")
    axes[0, 1].plot(t, trend, label="poly trend")
    axes[0, 1].set_title("Directional force")
    axes[0, 1].legend()
    axes[1, 0].plot(t, residual, marker="o")
    axes[1, 0].set_title(
        f"Detrended force, high-freq={metrics['fft_high_freq_power_fraction']:.3f}"
    )
    axes[1, 1].plot(t_inner, force_curv, marker="o", label="-d(F.v)/dt")
    axes[1, 1].plot(t_inner, energy_curv, marker="o", label="d2E/dt2")
    axes[1, 1].axhline(metrics["autograd_vhv"], color="black", linestyle="--", label="autograd")
    axes[1, 1].set_title("Directional curvature")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.set_xlabel("t / Angstrom")
        ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate_summary(metrics_df: pd.DataFrame) -> dict[str, Any]:
    numeric = metrics_df.select_dtypes(include=[np.number])
    summary: dict[str, Any] = {"n_lines": int(len(metrics_df))}
    for col in numeric.columns:
        if col in {"structure_idx", "dataset_idx", "direction_idx", "n_atoms"}:
            continue
        values = numeric[col].to_numpy(float)
        finite = values[np.isfinite(values)]
        if finite.size:
            summary[col] = {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "p90": float(np.quantile(finite, 0.90)),
                "max": float(np.max(finite)),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "ckpt" / "eqv2.ckpt")
    parser.add_argument("--dataset", type=Path, default=project_root() / "data" / "sample_100.lmdb")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "runs" / "force_wiggle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-indices", type=parse_int_list, default=None)
    parser.add_argument("--max-structures", type=int, default=3)
    parser.add_argument("--directions-per-structure", type=int, default=2)
    parser.add_argument("--n-t", type=int, default=41)
    parser.add_argument("--t-max", type=float, default=0.02)
    parser.add_argument("--fd-eps", type=parse_float_list, default=parse_float_list("0.0001,0.0003,0.001,0.003,0.01"))
    parser.add_argument("--detrend-degree", type=int, default=3)
    parser.add_argument("--seed", type=int, default=141)
    parser.add_argument("--max-plots", type=int, default=6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metadata = model_metadata(args.checkpoint)
    probe = ForceProbe(args.checkpoint, args.device)
    cutoff = float(metadata.get("max_radius") or probe.calc.potential.cutoff)
    dataset = LmdbDataset(args.dataset)
    rng = np.random.default_rng(args.seed)

    if args.dataset_indices is None:
        n_structures = min(args.max_structures, len(dataset))
        structure_indices = np.linspace(0, len(dataset) - 1, n_structures, dtype=int).tolist()
    else:
        structure_indices = args.dataset_indices

    t_values = np.linspace(-args.t_max, args.t_max, args.n_t, dtype=float)
    line_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    slope_rows_all: list[dict[str, Any]] = []
    n_plots = 0

    for structure_idx, dataset_idx in enumerate(structure_indices):
        data = dataset[int(dataset_idx)]
        coords = data.pos.detach().cpu().numpy().astype(np.float64)
        atomic_nums = data.z.detach().cpu().numpy().astype(np.int64)
        for direction_idx in range(args.directions_per_structure):
            direction = normalized_random_direction(rng, coords.shape[0])
            line_df = scan_line(probe, coords, atomic_nums, direction, t_values, cutoff, args.device)
            autograd_vhv = probe.autograd_vhv(coords, atomic_nums, direction)
            center_energy = float(line_df.iloc[(line_df["t"].abs()).argmin()]["energy"])
            slope_rows = force_slope_sweep(
                probe,
                coords,
                atomic_nums,
                direction,
                args.fd_eps,
                autograd_vhv,
                center_energy,
            )
            metrics = line_metrics(line_df, slope_rows, autograd_vhv, args.detrend_degree)
            add_reference_metric(metrics, data, direction, autograd_vhv)
            common = {
                "structure_idx": structure_idx,
                "dataset_idx": int(dataset_idx),
                "direction_idx": direction_idx,
                "n_atoms": int(coords.shape[0]),
            }
            metrics.update(common)
            metric_rows.append(metrics)

            for row in slope_rows:
                row.update(common)
                slope_rows_all.append(row)

            line_df.insert(0, "direction_idx", direction_idx)
            line_df.insert(0, "dataset_idx", int(dataset_idx))
            line_df.insert(0, "structure_idx", structure_idx)
            line_df.insert(0, "n_atoms", int(coords.shape[0]))
            line_frames.append(line_df)

            if n_plots < args.max_plots:
                plot_line(
                    line_df,
                    metrics,
                    plot_dir / f"structure_{structure_idx:04d}_dataset_{dataset_idx}_direction_{direction_idx:02d}.png",
                )
                n_plots += 1

            print(
                f"Finished structure={structure_idx} dataset_idx={dataset_idx} "
                f"direction={direction_idx}"
            )

    line_all = pd.concat(line_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    slopes_df = pd.DataFrame(slope_rows_all)
    summary = aggregate_summary(metrics_df)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "dataset": str(args.dataset),
            "device": args.device,
            "model_config": metadata,
            "t_values": t_values.tolist(),
            "fd_eps": args.fd_eps,
        }
    )

    line_all.to_csv(args.output_dir / "force_line_scan.csv", index=False)
    metrics_df.to_csv(args.output_dir / "force_line_metrics.csv", index=False)
    slopes_df.to_csv(args.output_dir / "force_slope_sweep.csv", index=False)
    line_all.to_parquet(args.output_dir / "force_line_scan.parquet", index=False)
    metrics_df.to_parquet(args.output_dir / "force_line_metrics.parquet", index=False)
    slopes_df.to_parquet(args.output_dir / "force_slope_sweep.parquet", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote line scan: {args.output_dir / 'force_line_scan.csv'}")
    print(f"Wrote metrics: {args.output_dir / 'force_line_metrics.csv'}")
    print(f"Wrote slope sweep: {args.output_dir / 'force_slope_sweep.csv'}")
    print(f"Wrote summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
