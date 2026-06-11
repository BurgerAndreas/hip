#!/usr/bin/env python3
"""Evaluate scratch HIP v3 on Hugging Face ORCA Hessian HDF5 files.

Download the public evaluation data and checkpoints with:

    uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download; snapshot_download('andreasburger/hip', repo_type='model', allow_patterns=['orca_wb97x_631gd_chno_30_100/**','ckpt/**'], local_dir='hf_hip')"

This script imports the model and utilities from the checkout it lives in
(`/scratch/aburger/hip`). It writes per-sample metrics and summary statistics
for the published HDF5 reference dataset.
"""

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

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch as TGBatch
from torch_geometric.data import Data as TGData
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hip.frequency_analysis import analyze_frequencies_np  # noqa: E402
from hip.training_module import PotentialModule  # noqa: E402
from hip.units import bohr_to_angstrom, ev_angstrom_2_to_hartree_bohr_2  # noqa: E402


FIELDNAMES = [
    "shard_index",
    "sample_idx",
    "sample_name",
    "h5_path",
    "checkpoint_path",
    "code_root",
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
    "true_asymmetry_mae_ev_a2",
    "pred_asymmetry_mae_ev_a2",
    "true_neg_num",
    "pred_neg_num",
    "neg_num_agree",
    "eckart_eigval_mae_ev_a2",
    "eckart_lowest_eigval_mae_ev_a2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5-glob",
        default="/scratch/aburger/hip/hf_hip/orca_wb97x_631gd_chno_30_100/h5/*.h5",
        help="Glob for Hugging Face ORCA reference HDF5 files.",
    )
    parser.add_argument(
        "--ckpt-path",
        default="/scratch/aburger/hip/ckpt/hip_v3.ckpt",
        help="Scratch HIP checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        default="/scratch/aburger/hip/orca_hessians/hip_eval/hip_v3_scratch_code_hf_h5",
        help="Directory for metrics.csv and summary.json.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=16.0)
    parser.add_argument("--cutoff-hessian", type=float, default=16.0)
    parser.add_argument("--max-neighbors", type=int, default=100000)
    parser.add_argument("--redo", action="store_true")
    return parser.parse_args()


def select_shard(
    paths: list[Path], shard_index: int | None, shard_size: int | None
) -> list[Path]:
    if shard_index is None and shard_size is None:
        return paths
    if shard_index is None or shard_size is None:
        raise ValueError("--shard-index and --shard-size must be provided together")
    if shard_index < 0:
        raise ValueError("--shard-index must be non-negative")
    if shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    start = shard_index * shard_size
    return paths[start : start + shard_size]


def completed_paths(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    with results_path.open(newline="") as handle:
        return {
            row["h5_path"]
            for row in csv.DictReader(handle)
            if row.get("status") == "ok" and row.get("h5_path")
        }


def resolve_checkpoint(path: str) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.exists():
        return checkpoint_path.resolve()
    if checkpoint_path.suffix == "":
        with_suffix = checkpoint_path.with_suffix(".ckpt")
        if with_suffix.exists():
            return with_suffix.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def load_hf_orca_hessian(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    with h5py.File(path, "r") as handle:
        atomic_numbers = np.asarray(handle["atomic_numbers"], dtype=np.int64)
        coords_bohr = np.asarray(handle["coordinates_bohr"], dtype=np.float64)
        hessian_hartree_bohr2 = np.asarray(
            handle["hessian_hartree_per_bohr2"], dtype=np.float64
        )
        symbols = [
            symbol.decode("utf-8") if isinstance(symbol, bytes) else str(symbol)
            for symbol in np.asarray(handle["symbols"])
        ]

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


def load_module_for_inference(checkpoint_path: Path, device: torch.device):
    original_fix_paths = PotentialModule.fix_paths

    def keep_checkpoint_paths(
        self: PotentialModule, training_config: dict[str, Any]
    ) -> dict[str, Any]:
        # Inference does not construct train/val datasets, so the original
        # checkpoint dataset paths do not need to exist on this filesystem.
        return dict(training_config)

    PotentialModule.fix_paths = keep_checkpoint_paths
    try:
        return PotentialModule.load_from_checkpoint(
            str(checkpoint_path), strict=False, map_location=device
        )
    finally:
        PotentialModule.fix_paths = original_fix_paths


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
            energy, _forces, outputs = model.forward(batch, otf_graph=True, hessian=True)
        end_event.record()
        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event)
        memory_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        start = time.perf_counter()
        with torch.no_grad():
            energy, _forces, outputs = model.forward(batch, otf_graph=True, hessian=True)
        time_ms = (time.perf_counter() - start) * 1000.0
        memory_mb = 0.0

    n_dof = batch.pos.numel()
    hessian = outputs["hessian"].reshape(n_dof, n_dof)
    return energy, hessian, time_ms, memory_mb


def sample_metrics(
    model: torch.nn.Module,
    path: Path,
    sample_idx: int,
    shard_index: int | None,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    # Load true Hessians and convert to eV/Angstrom^2
    atomic_numbers, coords_bohr, symbols, true_hessian_hartree_bohr2 = (
        load_hf_orca_hessian(path)
    )
    coords_angstrom = coords_bohr * bohr_to_angstrom
    true_hessian_ev_a2 = true_hessian_hartree_bohr2 / ev_angstrom_2_to_hartree_bohr_2

    # get predicted HIP Hessian
    batch = make_batch(atomic_numbers, coords_angstrom, device)
    energy, pred_hessian_ev_a2_t, time_ms, memory_mb = predict_dense_hessian(
        model, batch, device
    )
    pred_hessian_ev_a2 = pred_hessian_ev_a2_t.detach().cpu().numpy()
    diff = pred_hessian_ev_a2 - true_hessian_ev_a2

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
        "shard_index": shard_index if shard_index is not None else "",
        "sample_idx": sample_idx,
        "sample_name": path.stem,
        "h5_path": str(path),
        "checkpoint_path": str(checkpoint_path),
        "code_root": str(REPO_ROOT),
        "hessian_method": "predict",
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
        "true_asymmetry_mae_ev_a2": float(
            np.mean(np.abs(true_hessian_ev_a2 - true_hessian_ev_a2.T))
        ),
        "pred_asymmetry_mae_ev_a2": float(
            np.mean(np.abs(pred_hessian_ev_a2 - pred_hessian_ev_a2.T))
        ),
        "true_neg_num": int(true_freqs["neg_num"]),
        "pred_neg_num": int(pred_freqs["neg_num"]),
        "neg_num_agree": int(true_freqs["neg_num"] == pred_freqs["neg_num"]),
        "eckart_eigval_mae_ev_a2": float(np.mean(np.abs(eckart_diff))),
        "eckart_lowest_eigval_mae_ev_a2": float(abs(eckart_diff[0])),
    }


def error_row(
    path: Path,
    sample_idx: int,
    shard_index: int | None,
    checkpoint_path: Path,
    error: BaseException,
) -> dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "shard_index": shard_index if shard_index is not None else "",
            "sample_idx": sample_idx,
            "sample_name": path.stem,
            "h5_path": str(path),
            "checkpoint_path": str(checkpoint_path),
            "code_root": str(REPO_ROOT),
            "hessian_method": "predict",
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

    h5_paths = [Path(path) for path in sorted(glob.glob(args.h5_glob))]
    if args.max_samples is not None:
        h5_paths = h5_paths[: args.max_samples]
    total_h5_paths = len(h5_paths)
    h5_paths = select_shard(h5_paths, args.shard_index, args.shard_size)
    if not h5_paths:
        raise FileNotFoundError(
            f"No Hugging Face ORCA HDF5 files selected from {total_h5_paths} matches"
        )

    checkpoint_path = resolve_checkpoint(args.ckpt_path)

    if args.redo and results_path.exists():
        results_path.unlink()

    already_done = completed_paths(results_path)
    write_header = not results_path.exists()
    mode = "a" if results_path.exists() else "w"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"code_root={REPO_ROOT}")
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    if args.shard_index is not None:
        print(
            f"Selected shard {args.shard_index} with shard_size={args.shard_size} "
            f"from {total_h5_paths} Hugging Face ORCA HDF5 files"
        )

    module = load_module_for_inference(checkpoint_path, device)
    model = module.potential.to(device)
    model.eval()
    model.cutoff = args.cutoff
    model.cutoff_hessian = args.cutoff_hessian
    model.max_neighbors = args.max_neighbors

    print(f"Evaluating {len(h5_paths)} Hugging Face ORCA HDF5 files")
    print(f"Writing per-sample metrics to {results_path}")
    with results_path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for sample_idx, path in enumerate(tqdm(h5_paths, desc="Evaluating")):
            if str(path) in already_done:
                continue
            try:
                row = sample_metrics(
                    model, path, sample_idx, args.shard_index, checkpoint_path, device
                )
            except Exception as exc:
                traceback.print_exc()
                row = error_row(path, sample_idx, args.shard_index, checkpoint_path, exc)
            writer.writerow(row)
            handle.flush()

    write_summary(results_path, summary_path)


if __name__ == "__main__":
    main()
