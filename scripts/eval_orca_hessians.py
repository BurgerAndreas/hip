#!/usr/bin/env python3
"""Evaluate HIP Hessian predictions on ORCA `.hess` files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch as TGBatch
from torch_geometric.data import Data as TGData
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hip.frequency_analysis import analyze_frequencies_np  # noqa: E402
from hip.training_module import PotentialModule  # noqa: E402
from hip.units import (  # noqa: E402
    bohr_to_angstrom,
    ev_angstrom_2_to_hartree_bohr_2,
)
from scripts.collect_orca_hessian_results import parse_atoms, parse_hessian  # noqa: E402


FIELDNAMES = [
    "sample_idx",
    "sample_name",
    "hess_path",
    "hessian_method",
    "natoms",
    "n_dof",
    "status",
    "error",
    "time_ms",
    "memory_mb",
    "energy_pred_ev",
    "hessian_mae_ev_a2",
    "hessian_rmse_ev_a2",
    "hessian_max_abs_ev_a2",
    "hessian_rel_mae",
    "cart_eigval_mae_ev_a2",
    "cart_eigval_rmse_ev_a2",
    "true_asymmetry_mae_ev_a2",
    "pred_asymmetry_mae_ev_a2",
    "true_neg_num",
    "pred_neg_num",
    "neg_num_agree",
    "eigval_mae_eckart",
    "eigval1_mae_eckart",
    "eigval2_mae_eckart",
    "eckart_eigval_mae_hartree_bohr2",
    "eckart_lowest_eigval_mae_hartree_bohr2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hess-glob",
        default="/scratch/aburger/hip/orca_hessians/jobs/*/*.hess",
        help="Glob for ORCA .hess files.",
    )
    parser.add_argument(
        "--ckpt-path",
        "-c",
        default="ckpt/hip_v2.ckpt",
        help="HIP Lightning checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        default="/scratch/aburger/hip/orca_hessians/hip_eval/hip_v2_large_hessians",
        help="Directory for CSV and summary outputs.",
    )
    parser.add_argument(
        "--max-samples",
        "-m",
        type=int,
        default=None,
        help="Maximum number of Hessian files to evaluate.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based shard index for chunked or Slurm array evaluation.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Number of Hessian files per shard.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Override message passing cutoff in Angstrom.",
    )
    parser.add_argument(
        "--cutoff-hessian",
        type=float,
        default=None,
        help="Override Hessian edge cutoff in Angstrom.",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=None,
        help="Override maximum message-passing graph neighbors.",
    )
    parser.add_argument(
        "--hessian-method",
        choices=["predict", "autograd"],
        default="predict",
        help="Use the model Hessian head or compute the force Jacobian with autograd.",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Ignore an existing metrics CSV and recompute all samples.",
    )
    return parser.parse_args()


def select_shard(
    hess_paths: list[Path], shard_index: int | None, shard_size: int | None
) -> list[Path]:
    if shard_index is None and shard_size is None:
        return hess_paths
    if shard_index is None or shard_size is None:
        raise ValueError("--shard-index and --shard-size must be provided together")
    if shard_index < 0:
        raise ValueError("--shard-index must be non-negative")
    if shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    start = shard_index * shard_size
    stop = start + shard_size
    return hess_paths[start:stop]


def completed_paths(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    with results_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["hess_path"]
            for row in reader
            if row.get("status") == "ok" and row.get("hess_path")
        }


def load_orca_hessian(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    hessian_hartree_bohr2 = parse_hessian(lines)
    atomic_numbers, coords_bohr, symbols = parse_atoms(lines)
    expected_shape = (3 * len(atomic_numbers), 3 * len(atomic_numbers))
    if hessian_hartree_bohr2.shape != expected_shape:
        raise ValueError(
            f"Hessian has shape {hessian_hartree_bohr2.shape}, expected {expected_shape}"
        )
    return atomic_numbers, coords_bohr, symbols, hessian_hartree_bohr2


def make_batch(
    atomic_numbers: np.ndarray, coords_angstrom: np.ndarray, device: torch.device
) -> TGBatch:
    data = TGData(
        pos=torch.as_tensor(coords_angstrom, dtype=torch.float32),
        z=torch.as_tensor(atomic_numbers, dtype=torch.int64),
        charges=torch.as_tensor(atomic_numbers, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_numbers)], dtype=torch.int64),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )
    return TGBatch.from_data_list([data]).to(device)


def _get_derivatives(
    x: torch.Tensor,
    y: torch.Tensor,
    retain_graph: bool | None = None,
    create_graph: bool = False,
) -> torch.Tensor:
    return torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph
    )[0]


def compute_hessian_autograd(
    coords: torch.Tensor, energy: torch.Tensor, forces: torch.Tensor | None = None
) -> torch.Tensor:
    if forces is None:
        forces = -_get_derivatives(coords, energy, create_graph=True)

    hessian_rows = []
    for force_component in forces.reshape(-1):
        hessian_rows.append(_get_derivatives(coords, -force_component, retain_graph=True))
    return torch.stack(hessian_rows).reshape(forces.numel(), -1)


def predict_dense_hessian(
    model: torch.nn.Module, batch: TGBatch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            energy, _forces, outputs = model.forward(
                batch,
                otf_graph=True,
                hessian=True,
                add_props=True,
                return_dense_hessian=True,
            )
        end_event.record()
        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event)
        memory_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        start = time.perf_counter()
        with torch.no_grad():
            energy, _forces, outputs = model.forward(
                batch,
                otf_graph=True,
                hessian=True,
                add_props=True,
                return_dense_hessian=True,
            )
        time_ms = (time.perf_counter() - start) * 1000.0
        memory_mb = 0.0

    n_dof = batch.pos.numel()
    hessian = outputs["hessian"].reshape(n_dof, n_dof)
    return energy, hessian, time_ms, memory_mb


def autograd_dense_hessian(
    model: torch.nn.Module, batch: TGBatch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    batch.pos.requires_grad_(True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        energy, forces, _outputs = model.forward(
            batch,
            otf_graph=True,
            hessian=False,
        )
        hessian = compute_hessian_autograd(batch.pos, energy, forces)
        end_event.record()
        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event)
        memory_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        start = time.perf_counter()
        energy, forces, _outputs = model.forward(
            batch,
            otf_graph=True,
            hessian=False,
        )
        hessian = compute_hessian_autograd(batch.pos, energy, forces)
        time_ms = (time.perf_counter() - start) * 1000.0
        memory_mb = 0.0
    return energy, hessian, time_ms, memory_mb


def compute_dense_hessian(
    model: torch.nn.Module,
    batch: TGBatch,
    device: torch.device,
    hessian_method: str,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    if hessian_method == "predict":
        return predict_dense_hessian(model, batch, device)
    if hessian_method == "autograd":
        return autograd_dense_hessian(model, batch, device)
    raise ValueError(f"Unknown hessian method: {hessian_method}")


def load_module_for_inference(
    checkpoint_path: Path, device: torch.device
) -> PotentialModule:
    original_fix_paths = PotentialModule.fix_paths

    def keep_checkpoint_paths(
        self: PotentialModule, training_config: dict[str, Any]
    ) -> dict[str, Any]:
        # Evaluation never constructs train/val datasets, so do not require the
        # original training LMDB paths to exist on this filesystem.
        return dict(training_config)

    PotentialModule.fix_paths = keep_checkpoint_paths
    try:
        return PotentialModule.load_from_checkpoint(
            str(checkpoint_path),
            strict=False,
            map_location=device,
        )
    finally:
        PotentialModule.fix_paths = original_fix_paths


def sample_metrics(
    model: torch.nn.Module,
    path: Path,
    sample_idx: int,
    device: torch.device,
    hessian_method: str,
) -> dict[str, Any]:
    atomic_numbers, coords_bohr, symbols, true_hessian_hartree_bohr2 = load_orca_hessian(
        path
    )
    coords_angstrom = coords_bohr * bohr_to_angstrom
    true_hessian_ev_a2 = true_hessian_hartree_bohr2 / ev_angstrom_2_to_hartree_bohr_2
    batch = make_batch(atomic_numbers, coords_angstrom, device)
    energy, pred_hessian_ev_a2_t, time_ms, memory_mb = compute_dense_hessian(
        model, batch, device, hessian_method
    )

    pred_hessian_ev_a2 = pred_hessian_ev_a2_t.detach().cpu().numpy()

    diff = pred_hessian_ev_a2 - true_hessian_ev_a2
    true_eigvals = np.linalg.eigvalsh(true_hessian_ev_a2)
    pred_eigvals = np.linalg.eigvalsh(pred_hessian_ev_a2)

    # Match scripts/eval.py: pass raw dataset-style Hessians and coordinates
    # directly into analyze_frequencies_np, even though that helper documents
    # Hartree/Bohr^2 Hessians and Bohr coordinates.
    true_freqs = analyze_frequencies_np(
        hessian=true_hessian_ev_a2,
        cart_coords=coords_angstrom,
        atomsymbols=symbols,
    )
    pred_freqs = analyze_frequencies_np(
        hessian=pred_hessian_ev_a2,
        cart_coords=coords_angstrom,
        atomsymbols=symbols,
    )
    eckart_diff = pred_freqs["eigvals"] - true_freqs["eigvals"]

    return {
        "sample_idx": sample_idx,
        "sample_name": path.parent.name,
        "hess_path": str(path),
        "hessian_method": hessian_method,
        "natoms": len(atomic_numbers),
        "n_dof": 3 * len(atomic_numbers),
        "status": "ok",
        "error": "",
        "time_ms": time_ms,
        "memory_mb": memory_mb,
        "energy_pred_ev": energy.squeeze().detach().cpu().item(),
        "hessian_mae_ev_a2": float(np.mean(np.abs(diff))),
        "hessian_rmse_ev_a2": float(np.sqrt(np.mean(diff**2))),
        "hessian_max_abs_ev_a2": float(np.max(np.abs(diff))),
        "hessian_rel_mae": float(
            np.mean(np.abs(diff)) / (np.mean(np.abs(true_hessian_ev_a2)) + 1e-12)
        ),
        "cart_eigval_mae_ev_a2": float(np.mean(np.abs(pred_eigvals - true_eigvals))),
        "cart_eigval_rmse_ev_a2": float(
            np.sqrt(np.mean((pred_eigvals - true_eigvals) ** 2))
        ),
        "true_asymmetry_mae_ev_a2": float(
            np.mean(np.abs(true_hessian_ev_a2 - true_hessian_ev_a2.T))
        ),
        "pred_asymmetry_mae_ev_a2": float(
            np.mean(np.abs(pred_hessian_ev_a2 - pred_hessian_ev_a2.T))
        ),
        "true_neg_num": int(true_freqs["neg_num"]),
        "pred_neg_num": int(pred_freqs["neg_num"]),
        "neg_num_agree": int(true_freqs["neg_num"] == pred_freqs["neg_num"]),
        "eigval_mae_eckart": float(np.mean(np.abs(eckart_diff))),
        "eigval1_mae_eckart": float(abs(eckart_diff[0])),
        "eigval2_mae_eckart": float(abs(eckart_diff[1])),
        "eckart_eigval_mae_hartree_bohr2": float(np.mean(np.abs(eckart_diff))),
        "eckart_lowest_eigval_mae_hartree_bohr2": float(abs(eckart_diff[0])),
    }


def error_row(
    path: Path, sample_idx: int, hessian_method: str, error: BaseException
) -> dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "sample_idx": sample_idx,
            "sample_name": path.parent.name,
            "hess_path": str(path),
            "hessian_method": hessian_method,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }
    )
    return row


def write_summary(results_path: Path, summary_path: Path) -> None:
    rows: list[dict[str, str]] = []
    if results_path.exists():
        with results_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary: dict[str, Any] = {
        "results_path": str(results_path),
        "count_total_rows": len(rows),
        "count_ok": len(ok_rows),
        "count_error": sum(row.get("status") == "error" for row in rows),
    }
    for field in FIELDNAMES:
        values = []
        for row in ok_rows:
            try:
                values.append(float(row[field]))
            except (KeyError, TypeError, ValueError):
                continue
        if values:
            summary[f"mean_{field}"] = float(np.mean(values))
            summary[f"median_{field}"] = float(np.median(values))

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote summary to {summary_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"

    hess_paths = [Path(path) for path in sorted(glob.glob(args.hess_glob))]
    if args.max_samples is not None:
        hess_paths = hess_paths[: args.max_samples]
    total_hess_paths = len(hess_paths)
    hess_paths = select_shard(hess_paths, args.shard_index, args.shard_size)
    if not hess_paths:
        raise FileNotFoundError(
            f"No Hessian files selected from {total_hess_paths} matches for "
            f"shard_index={args.shard_index}, shard_size={args.shard_size}"
        )

    checkpoint_path = Path(args.ckpt_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.redo and results_path.exists():
        results_path.unlink()

    already_done = completed_paths(results_path)
    write_header = not results_path.exists()
    mode = "a" if results_path.exists() else "w"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {checkpoint_path} on {device}")
    module = load_module_for_inference(checkpoint_path, device)
    model = module.potential.to(device)
    model.eval()
    if args.cutoff is not None:
        model.cutoff = args.cutoff
    if args.cutoff_hessian is not None:
        model.cutoff_hessian = args.cutoff_hessian
    if args.max_neighbors is not None:
        model.max_neighbors = args.max_neighbors
    print(
        "Effective graph settings: "
        f"cutoff={model.cutoff}, "
        f"cutoff_hessian={model.cutoff_hessian}, "
        f"max_neighbors={model.max_neighbors}"
    )

    if args.shard_index is not None:
        print(
            f"Selected shard {args.shard_index} with shard_size={args.shard_size} "
            f"from {total_hess_paths} Hessian files"
        )
    print(f"Evaluating {len(hess_paths)} Hessian files with {args.hessian_method}")
    print(f"Writing per-sample metrics to {results_path}")
    with results_path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for sample_idx, path in enumerate(tqdm(hess_paths, desc="Evaluating")):
            if str(path) in already_done:
                continue
            try:
                row = sample_metrics(
                    model, path, sample_idx, device, args.hessian_method
                )
            except Exception as exc:
                traceback.print_exc()
                row = error_row(path, sample_idx, args.hessian_method, exc)
            writer.writerow(row)
            handle.flush()

    write_summary(results_path, summary_path)


if __name__ == "__main__":
    main()
