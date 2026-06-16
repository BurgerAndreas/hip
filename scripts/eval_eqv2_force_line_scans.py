#!/usr/bin/env python
"""Evaluate EqV2 energy/forces on generated line-scan geometries.

This intentionally uses ``do_hessian=False``. The output is grouped by
``(geom_rank, direction_id)`` so each line scan has one compact NPZ file with
``lambda_ang``, ``energy_ev``, and ``forces_ev_ang``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.ff_lmdb import Z_TO_ATOM_SYMBOL

SYMBOL_TO_Z = {symbol: z for z, symbol in Z_TO_ATOM_SYMBOL.items()}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open() as handle:
        lines = [line.strip() for line in handle if line.strip()]
    n_atoms = int(lines[0])
    symbols: list[str] = []
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for i, line in enumerate(lines[2 : 2 + n_atoms]):
        fields = line.split()
        symbols.append(fields[0])
        coords[i] = [float(fields[1]), float(fields[2]), float(fields[3])]
    return symbols, coords


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=project_root() / "runs" / "t1x_val_force_spectra_100x2x51",
    )
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "ckpt" / "eqv2.ckpt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scan_dir = args.scan_dir
    output_dir = args.output_dir or scan_dir / "eqv2_force_outputs"
    npz_dir = output_dir / "line_npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(scan_dir / "scan_manifest.csv")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    calc = EquiformerTorchCalculator(
        checkpoint_path=str(args.checkpoint),
        hessian_method="autograd",
        device=device,
    )

    summary_rows: list[dict[str, object]] = []
    grouped = manifest.groupby(["geom_rank", "dataset_idx", "direction_id"], sort=True)
    for (geom_rank, dataset_idx, direction_id), frame in grouped:
        frame = frame.sort_values("point_id")
        line_path = npz_dir / f"g{int(geom_rank):04d}_idx{int(dataset_idx):06d}_d{int(direction_id)}.npz"
        if line_path.exists() and not args.overwrite:
            for row in frame.to_dict(orient="records"):
                summary_rows.append(
                    {
                        "scan_point_id": int(row["scan_point_id"]),
                        "geom_rank": int(geom_rank),
                        "dataset_idx": int(dataset_idx),
                        "direction_id": int(direction_id),
                        "point_id": int(row["point_id"]),
                        "lambda_ang": float(row["lambda_ang"]),
                        "line_npz_path": str(line_path.resolve()),
                        "skipped_existing": True,
                    }
                )
            continue

        energies: list[float] = []
        forces: list[np.ndarray] = []
        coords_rows: list[np.ndarray] = []
        atomic_nums: np.ndarray | None = None
        symbols0: list[str] | None = None

        for row in frame.to_dict(orient="records"):
            symbols, coords = read_xyz(Path(row["xyz_path"]))
            z = np.asarray([SYMBOL_TO_Z[symbol] for symbol in symbols], dtype=np.int64)
            if atomic_nums is None:
                atomic_nums = z
                symbols0 = symbols
            elif not np.array_equal(atomic_nums, z):
                raise ValueError(f"Atomic numbers changed within line {line_path}")

            with torch.no_grad():
                out = calc.predict(
                    coords=torch.tensor(coords, dtype=torch.float32, device=device),
                    atomic_nums=torch.tensor(z, dtype=torch.long, device=device),
                    do_hessian=False,
                )
            energy = float(out["energy"].detach().cpu().reshape(-1)[0].item())
            force = out["forces"].detach().cpu().numpy().reshape(coords.shape)
            energies.append(energy)
            forces.append(force.astype(np.float64))
            coords_rows.append(coords)
            summary_rows.append(
                {
                    "scan_point_id": int(row["scan_point_id"]),
                    "geom_rank": int(geom_rank),
                    "dataset_idx": int(dataset_idx),
                    "direction_id": int(direction_id),
                    "point_id": int(row["point_id"]),
                    "lambda_ang": float(row["lambda_ang"]),
                    "eqv2_energy_ev": energy,
                    "eqv2_force_norm_ev_ang": float(np.linalg.norm(force.reshape(-1))),
                    "eqv2_fmax_ev_ang": float(np.max(np.abs(force))),
                    "line_npz_path": str(line_path.resolve()),
                    "skipped_existing": False,
                }
            )

        np.savez_compressed(
            line_path,
            lambda_ang=frame["lambda_ang"].to_numpy(dtype=np.float64),
            point_id=frame["point_id"].to_numpy(dtype=np.int64),
            scan_point_id=frame["scan_point_id"].to_numpy(dtype=np.int64),
            atomic_numbers=np.asarray(atomic_nums, dtype=np.int64),
            symbols=np.asarray(symbols0),
            coords_ang=np.stack(coords_rows),
            energy_ev=np.asarray(energies, dtype=np.float64),
            forces_ev_ang=np.stack(forces),
        )
        print(f"Wrote {line_path}", flush=True)

    summary = pd.DataFrame(summary_rows).sort_values("scan_point_id")
    summary.to_csv(output_dir / "eqv2_force_summary.csv", index=False)
    summary.to_parquet(output_dir / "eqv2_force_summary.parquet", index=False)
    print(f"Wrote {len(summary)} EqV2 force rows to {output_dir}")


if __name__ == "__main__":
    main()
