#!/usr/bin/env python
"""Probe whether AD Hessians expose nonsmooth MLIP force fields.

The diagnostic samples short one-dimensional displacement lines through real
HORM geometries and compares:

- AD force-Jacobian curvature, v^T H_auto v
- finite-difference curvature from EQV2 forces and energies
- HIP direct Hessian-head curvature, v^T H_pred v, on the same geometries
- graph-neighbor count changes along each line

The output is intentionally tabular so the scan can be rerun with different
step sizes or checkpoints and compared without changing plotting code.
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


def default_eqv2_checkpoint() -> Path:
    requested = project_root() / "ckpt" / "ckpt_eqv2.ckpt"
    fallback = project_root() / "ckpt" / "eqv2.ckpt"
    return requested if requested.exists() else fallback


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def tensor_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def as_numpy_hessian(value: torch.Tensor, n_atoms: int) -> np.ndarray:
    return value.reshape(3 * n_atoms, 3 * n_atoms).detach().cpu().numpy()


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def directional_curvature(hessian: np.ndarray, direction: np.ndarray) -> float:
    flat = direction.reshape(-1)
    return float(flat @ hessian @ flat)


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


def pairwise_cutoff_margin(coords: np.ndarray, cutoff: float) -> tuple[float, int]:
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
    if not np.any(mask):
        return float("nan"), 0
    margins = np.abs(distances[mask] - cutoff)
    return float(margins.min()), int(np.sum(margins < 0.02))


def edge_count(coords: np.ndarray, cutoff: float, device: str) -> int:
    pos = torch.tensor(coords, dtype=torch.float32, device=device)
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)
    with torch.no_grad():
        edges = radius_graph(pos, r=float(cutoff), batch=batch)
    return int(edges.shape[1])


def model_metadata(checkpoint: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = ckpt.get("hyper_parameters", {}).get("model_config", {})
    keys = [
        "name",
        "direct_forces",
        "enable_hessian_head",
        "max_radius",
        "cutoff_hessian",
        "fully_connected_hessian",
    ]
    return {key: model_config.get(key) for key in keys}


class ModelPair:
    def __init__(self, hip_checkpoint: Path, eqv2_checkpoint: Path, device: str):
        self.device = device
        self.hip = EquiformerTorchCalculator(
            checkpoint_path=str(hip_checkpoint),
            hessian_method="predict",
            device=device,
        )
        self.eqv2 = EquiformerTorchCalculator(
            checkpoint_path=str(eqv2_checkpoint),
            hessian_method="autograd",
            device=device,
        )

    def eqv2_energy_forces(
        self,
        coords: np.ndarray,
        atomic_nums: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        out = self.eqv2.predict(
            coords=torch.tensor(coords, dtype=torch.float32, device=self.device),
            atomic_nums=torch.tensor(atomic_nums, dtype=torch.long, device=self.device),
            do_hessian=False,
        )
        return (
            tensor_to_float(out["energy"]),
            out["forces"].reshape(-1, 3).detach().cpu().numpy(),
        )

    def eqv2_autograd_hessian(
        self,
        coords: np.ndarray,
        atomic_nums: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        n_atoms = int(atomic_nums.size)
        out = self.eqv2.predict(
            coords=torch.tensor(coords, dtype=torch.float32, device=self.device),
            atomic_nums=torch.tensor(atomic_nums, dtype=torch.long, device=self.device),
            hessian_method="autograd",
            do_hessian=True,
        )
        return (
            tensor_to_float(out["energy"]),
            out["forces"].reshape(n_atoms, 3).detach().cpu().numpy(),
            symmetrize(as_numpy_hessian(out["hessian"], n_atoms)),
        )

    def hip_pred_hessian(
        self,
        coords: np.ndarray,
        atomic_nums: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        n_atoms = int(atomic_nums.size)
        out = self.hip.predict(
            coords=torch.tensor(coords, dtype=torch.float32, device=self.device),
            atomic_nums=torch.tensor(atomic_nums, dtype=torch.long, device=self.device),
            hessian_method="predict",
            do_hessian=True,
        )
        return (
            tensor_to_float(out["energy"]),
            out["forces"].reshape(n_atoms, 3).detach().cpu().numpy(),
            symmetrize(as_numpy_hessian(out["hessian"], n_atoms)),
        )


def finite_difference_at_center(
    models: ModelPair,
    coords: np.ndarray,
    atomic_nums: np.ndarray,
    direction: np.ndarray,
    eps_values: list[float],
    eqv2_energy0: float,
    hip_energy0: float,
    eqv2_force_proj0: float,
    eqv2_auto_vhv: float,
    hip_pred_vhv: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    for eps in eps_values:
        e_plus, f_plus = models.eqv2_energy_forces(coords + eps * direction, atomic_nums)
        e_minus, f_minus = models.eqv2_energy_forces(coords - eps * direction, atomic_nums)
        f_plus_proj = float(f_plus.reshape(-1) @ direction.reshape(-1))
        f_minus_proj = float(f_minus.reshape(-1) @ direction.reshape(-1))
        rows.append(
            {
                "eps": float(eps),
                "eqv2_energy0": eqv2_energy0,
                "hip_energy0": hip_energy0,
                "eqv2_force_proj0": eqv2_force_proj0,
                "eqv2_auto_vhv": eqv2_auto_vhv,
                "hip_pred_vhv": hip_pred_vhv,
                "fd_energy_curvature": float((e_plus - 2.0 * eqv2_energy0 + e_minus) / eps**2),
                "fd_force_curvature": float(-(f_plus_proj - f_minus_proj) / (2.0 * eps)),
            }
        )
    return rows


def central_difference(values: np.ndarray, dt: float, kind: str) -> np.ndarray:
    if kind == "second":
        return (values[2:] - 2.0 * values[1:-1] + values[:-2]) / dt**2
    if kind == "first_negative":
        return -(values[2:] - values[:-2]) / (2.0 * dt)
    raise ValueError(kind)


def roughness(values: np.ndarray) -> float:
    if values.size < 3:
        return float("nan")
    numerator = np.mean(np.abs(values[2:] - 2.0 * values[1:-1] + values[:-2]))
    denominator = np.mean(np.abs(values)) + 1e-12
    return float(numerator / denominator)


def total_variation(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.sum(np.abs(np.diff(values))))


def analyze_line(
    models: ModelPair,
    coords: np.ndarray,
    atomic_nums: np.ndarray,
    direction: np.ndarray,
    t_values: np.ndarray,
    eqv2_cutoff: float,
    hip_cutoff: float,
    device: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float]] = []
    n_atoms = int(atomic_nums.size)
    for t in t_values:
        displaced = coords + float(t) * direction
        eqv2_e, eqv2_f, h_auto = models.eqv2_autograd_hessian(displaced, atomic_nums)
        hip_e, hip_f, h_hip = models.hip_pred_hessian(displaced, atomic_nums)
        margin_eqv2, near_eqv2 = pairwise_cutoff_margin(displaced, eqv2_cutoff)
        margin_hip, near_hip = pairwise_cutoff_margin(displaced, hip_cutoff)
        rows.append(
            {
                "t": float(t),
                "eqv2_energy": eqv2_e,
                "hip_energy": hip_e,
                "eqv2_force_proj": float(eqv2_f.reshape(-1) @ direction.reshape(-1)),
                "hip_force_proj": float(hip_f.reshape(-1) @ direction.reshape(-1)),
                "eqv2_auto_vhv": directional_curvature(h_auto, direction),
                "hip_pred_vhv": directional_curvature(h_hip, direction),
                "eqv2_edge_count": edge_count(displaced, eqv2_cutoff, device),
                "hip_edge_count": edge_count(displaced, hip_cutoff, device),
                "eqv2_min_cutoff_margin": margin_eqv2,
                "hip_min_cutoff_margin": margin_hip,
                "eqv2_pairs_within_0p02_cutoff": near_eqv2,
                "hip_pairs_within_0p02_cutoff": near_hip,
                "n_atoms": n_atoms,
            }
        )

    df = pd.DataFrame(rows)
    if len(t_values) >= 3:
        dt = float(t_values[1] - t_values[0])
        inner = df.iloc[1:-1].copy()
        fd_energy = central_difference(df["eqv2_energy"].to_numpy(float), dt, "second")
        fd_force = central_difference(df["eqv2_force_proj"].to_numpy(float), dt, "first_negative")
        auto_inner = inner["eqv2_auto_vhv"].to_numpy(float)
        hip_inner = inner["hip_pred_vhv"].to_numpy(float)
        metrics = {
            "eqv2_auto_vs_fd_force_mae": float(np.mean(np.abs(auto_inner - fd_force))),
            "eqv2_auto_vs_fd_energy_mae": float(np.mean(np.abs(auto_inner - fd_energy))),
            "hip_pred_vs_eqv2_fd_force_mae": float(np.mean(np.abs(hip_inner - fd_force))),
            "eqv2_fd_force_vs_fd_energy_mae": float(np.mean(np.abs(fd_force - fd_energy))),
            "eqv2_auto_tv": total_variation(df["eqv2_auto_vhv"].to_numpy(float)),
            "hip_pred_tv": total_variation(df["hip_pred_vhv"].to_numpy(float)),
            "eqv2_fd_force_tv": total_variation(fd_force),
            "eqv2_auto_roughness": roughness(df["eqv2_auto_vhv"].to_numpy(float)),
            "hip_pred_roughness": roughness(df["hip_pred_vhv"].to_numpy(float)),
            "eqv2_fd_force_roughness": roughness(fd_force),
            "eqv2_edge_count_range": float(
                df["eqv2_edge_count"].max() - df["eqv2_edge_count"].min()
            ),
            "hip_edge_count_range": float(df["hip_edge_count"].max() - df["hip_edge_count"].min()),
            "eqv2_min_cutoff_margin_min": float(df["eqv2_min_cutoff_margin"].min()),
            "hip_min_cutoff_margin_min": float(df["hip_min_cutoff_margin"].min()),
        }
    else:
        metrics = {}
    return df, metrics


def add_reference_hessian_metrics(
    metrics: dict[str, float],
    data: Any,
    direction: np.ndarray,
    eqv2_auto_vhv: float,
    hip_pred_vhv: float,
) -> None:
    if not hasattr(data, "hessian"):
        return
    n_atoms = int(direction.shape[0])
    h_true = data.hessian.reshape(3 * n_atoms, 3 * n_atoms).detach().cpu().numpy()
    h_true = symmetrize(h_true)
    true_vhv = directional_curvature(h_true, direction)
    metrics["true_vhv"] = true_vhv
    metrics["eqv2_auto_vs_true_abs"] = abs(eqv2_auto_vhv - true_vhv)
    metrics["hip_pred_vs_true_abs"] = abs(hip_pred_vhv - true_vhv)


def plot_directions(line_df: pd.DataFrame, output_dir: Path, max_plots: int) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    keys = list(line_df[["structure_idx", "direction_idx"]].drop_duplicates().itertuples(index=False))
    for structure_idx, direction_idx in keys[:max_plots]:
        sub = line_df[
            (line_df["structure_idx"] == structure_idx)
            & (line_df["direction_idx"] == direction_idx)
        ].sort_values("t")
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        axes[0, 0].plot(sub["t"], sub["eqv2_energy"], marker="o")
        axes[0, 0].set_title("EQV2 energy")
        axes[0, 1].plot(sub["t"], sub["eqv2_force_proj"], marker="o")
        axes[0, 1].set_title("EQV2 force projection")
        axes[1, 0].plot(sub["t"], sub["eqv2_auto_vhv"], marker="o", label="AD")
        axes[1, 0].plot(sub["t"], sub["hip_pred_vhv"], marker="o", label="HIP predicted")
        axes[1, 0].set_title("Directional curvature")
        axes[1, 0].legend()
        axes[1, 1].plot(sub["t"], sub["eqv2_edge_count"], marker="o", label="EQV2")
        axes[1, 1].plot(sub["t"], sub["hip_edge_count"], marker="o", label="HIP")
        axes[1, 1].set_title("Neighbor count")
        axes[1, 1].legend()
        for ax in axes.ravel():
            ax.set_xlabel("t / Angstrom")
        fig.savefig(plot_dir / f"structure_{structure_idx:04d}_direction_{direction_idx:02d}.png", dpi=180)
        plt.close(fig)


def aggregate_summary(metrics_df: pd.DataFrame) -> dict[str, Any]:
    numeric = metrics_df.select_dtypes(include=[np.number])
    summary: dict[str, Any] = {
        "n_lines": int(len(metrics_df)),
        "columns": list(metrics_df.columns),
    }
    for col in numeric.columns:
        if col in {"structure_idx", "direction_idx", "dataset_idx", "n_atoms"}:
            continue
        values = numeric[col].to_numpy(float)
        finite = values[np.isfinite(values)]
        if finite.size:
            summary[col] = {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "max": float(np.max(finite)),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hip-checkpoint", type=Path, default=project_root() / "ckpt" / "hip_v2.ckpt")
    parser.add_argument("--eqv2-checkpoint", type=Path, default=default_eqv2_checkpoint())
    parser.add_argument("--dataset", type=Path, default=project_root() / "data" / "sample_100.lmdb")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "runs" / "hessian_smoothness")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-structures", type=int, default=3)
    parser.add_argument(
        "--dataset-indices",
        type=parse_int_list,
        default=None,
        help="Comma-separated LMDB indices to scan. Overrides --max-structures selection.",
    )
    parser.add_argument("--directions-per-structure", type=int, default=4)
    parser.add_argument("--n-t", type=int, default=9)
    parser.add_argument("--t-max", type=float, default=0.01)
    parser.add_argument("--fd-eps", type=parse_float_list, default=parse_float_list("0.0001,0.0003,0.001,0.003,0.01"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-plots", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dataset = LmdbDataset(args.dataset)

    hip_meta = model_metadata(args.hip_checkpoint)
    eqv2_meta = model_metadata(args.eqv2_checkpoint)
    models = ModelPair(args.hip_checkpoint, args.eqv2_checkpoint, args.device)
    eqv2_cutoff = float(eqv2_meta.get("max_radius") or models.eqv2.potential.cutoff)
    hip_cutoff = float(hip_meta.get("max_radius") or models.hip.potential.cutoff)

    t_values = np.linspace(-args.t_max, args.t_max, args.n_t, dtype=float)
    line_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    fd_rows: list[dict[str, Any]] = []

    if args.dataset_indices is None:
        n_structures = min(args.max_structures, len(dataset))
        structure_indices = np.linspace(0, len(dataset) - 1, n_structures, dtype=int).tolist()
    else:
        structure_indices = args.dataset_indices
    for structure_order, dataset_idx in enumerate(structure_indices):
        data = dataset[int(dataset_idx)]
        coords = data.pos.detach().cpu().numpy().astype(np.float64)
        atomic_nums = data.z.detach().cpu().numpy().astype(np.int64)
        for direction_idx in range(args.directions_per_structure):
            direction = normalized_random_direction(rng, coords.shape[0])
            line_df, metrics = analyze_line(
                models=models,
                coords=coords,
                atomic_nums=atomic_nums,
                direction=direction,
                t_values=t_values,
                eqv2_cutoff=eqv2_cutoff,
                hip_cutoff=hip_cutoff,
                device=args.device,
            )
            center_row = line_df.iloc[(line_df["t"].abs()).argmin()]
            metrics.update(
                {
                    "structure_idx": structure_order,
                    "dataset_idx": int(dataset_idx),
                    "direction_idx": direction_idx,
                    "n_atoms": int(coords.shape[0]),
                    "eqv2_center_energy": float(center_row["eqv2_energy"]),
                    "hip_center_energy": float(center_row["hip_energy"]),
                }
            )
            add_reference_hessian_metrics(
                metrics,
                data,
                direction,
                float(center_row["eqv2_auto_vhv"]),
                float(center_row["hip_pred_vhv"]),
            )
            metric_rows.append(metrics)

            line_df.insert(0, "direction_idx", direction_idx)
            line_df.insert(0, "dataset_idx", int(dataset_idx))
            line_df.insert(0, "structure_idx", structure_order)
            line_frames.append(line_df)

            for fd_row in finite_difference_at_center(
                models,
                coords,
                atomic_nums,
                direction,
                args.fd_eps,
                float(center_row["eqv2_energy"]),
                float(center_row["hip_energy"]),
                float(center_row["eqv2_force_proj"]),
                float(center_row["eqv2_auto_vhv"]),
                float(center_row["hip_pred_vhv"]),
            ):
                fd_row.update(
                    {
                        "structure_idx": structure_order,
                        "dataset_idx": int(dataset_idx),
                        "direction_idx": direction_idx,
                        "n_atoms": int(coords.shape[0]),
                    }
                )
                fd_rows.append(fd_row)

            print(
                f"Finished structure={structure_order} dataset_idx={dataset_idx} "
                f"direction={direction_idx}"
            )

    line_df = pd.concat(line_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    fd_df = pd.DataFrame(fd_rows)
    summary = aggregate_summary(metrics_df)
    summary["hip_checkpoint"] = str(args.hip_checkpoint)
    summary["eqv2_checkpoint"] = str(args.eqv2_checkpoint)
    summary["dataset"] = str(args.dataset)
    summary["device"] = args.device
    summary["t_values"] = t_values.tolist()
    summary["fd_eps"] = args.fd_eps
    summary["hip_model_config"] = hip_meta
    summary["eqv2_model_config"] = eqv2_meta

    line_df.to_csv(args.output_dir / "line_scan.csv", index=False)
    metrics_df.to_csv(args.output_dir / "line_metrics.csv", index=False)
    fd_df.to_csv(args.output_dir / "fd_epsilon_sweep.csv", index=False)
    line_df.to_parquet(args.output_dir / "line_scan.parquet", index=False)
    metrics_df.to_parquet(args.output_dir / "line_metrics.parquet", index=False)
    fd_df.to_parquet(args.output_dir / "fd_epsilon_sweep.parquet", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_directions(line_df, args.output_dir, args.max_plots)

    print(f"Wrote line scan: {args.output_dir / 'line_scan.csv'}")
    print(f"Wrote metrics: {args.output_dir / 'line_metrics.csv'}")
    print(f"Wrote FD epsilon sweep: {args.output_dir / 'fd_epsilon_sweep.csv'}")
    print(f"Wrote summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
