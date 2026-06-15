#!/usr/bin/env python
"""Prepare and evaluate the 73-frame glycine MEP geodesic trajectory.

The input trajectory is produced by ``plotting/visualize_glycine_pt_xyzrender.py``.
This script writes ORCA inputs for every frame and, unless skipped, evaluates one
MLIP checkpoint to save energies, forces, and Hessians on the same geometries.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import pandas as pd
import torch

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.transition1x_dataset import Transition1xDataset


SPLIT = "test"
SAMPLE_ID = 5
O_ATOM = 3
N_ATOM = 4
H_ATOM = 9
DEFAULT_ORCA_ROUTE = "! wB97X-D3 6-31G(d) TightSCF Grid5 FinalGrid6 EnGrad Freq"
Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return safe.strip("_") or "model"


def symbols_from_z(atomic_nums: np.ndarray) -> list[str]:
    return [Z_TO_SYMBOL[int(z)] for z in atomic_nums.tolist()]


def read_trajectory_xyz(path: Path, n_atoms: int) -> np.ndarray:
    lines = path.read_text().splitlines()
    frames: list[np.ndarray] = []
    idx = 0
    while idx < len(lines):
        nat = int(lines[idx].strip())
        if nat != n_atoms:
            raise ValueError(f"{path} has {nat} atoms, expected {n_atoms}")
        atom_lines = lines[idx + 2 : idx + 2 + n_atoms]
        coords = np.asarray([[float(x) for x in line.split()[1:4]] for line in atom_lines], dtype=float)
        frames.append(coords)
        idx += n_atoms + 2
    return np.stack(frames)


def write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str) -> None:
    with path.open("w") as handle:
        handle.write(f"{len(symbols)}\n{comment}\n")
        for symbol, xyz in zip(symbols, coords, strict=True):
            handle.write(f"{symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


def write_orca_input(
    path: Path,
    symbols: list[str],
    coords: np.ndarray,
    route: str,
    charge: int,
    multiplicity: int,
) -> None:
    with path.open("w") as handle:
        handle.write(f"{route}\n\n%pal nprocs 16 end\n%maxcore 4000\n\n")
        handle.write(f"* xyz {charge} {multiplicity}\n")
        for symbol, xyz in zip(symbols, coords, strict=True):
            handle.write(f"  {symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")
        handle.write("*\n")


def bond_lengths(coords: np.ndarray) -> tuple[float, float]:
    q_nh = float(np.linalg.norm(coords[N_ATOM] - coords[H_ATOM]))
    q_oh = float(np.linalg.norm(coords[O_ATOM] - coords[H_ATOM]))
    return q_nh, q_oh


def prepare_inputs(
    out_dir: Path,
    coords_frames: np.ndarray,
    symbols: list[str],
    sample,
    route: str,
    charge: int,
    multiplicity: int,
) -> list[dict[str, object]]:
    xyz_dir = out_dir / "xyz"
    orca_dir = out_dir / "orca_inputs"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    orca_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "scan_manifest.csv"
    input_list_path = out_dir / "orca_input_list.txt"
    fieldnames = [
        "grid_id",
        "frame_id",
        "split",
        "sample_id",
        "formula",
        "rxn",
        "q_nh",
        "q_oh",
        "xi",
        "xyz_path",
        "orca_input_path",
    ]

    rows: list[dict[str, object]] = []
    with manifest_path.open("w", newline="") as manifest_handle, input_list_path.open("w") as list_handle:
        writer = csv.DictWriter(manifest_handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_id, coords in enumerate(coords_frames):
            q_nh, q_oh = bond_lengths(coords)
            xi = q_nh - q_oh
            name = f"mep_{frame_id:04d}_xi_{xi:.4f}_qNH_{q_nh:.3f}_qOH_{q_oh:.3f}"
            xyz_path = xyz_dir / f"{name}.xyz"
            inp_path = orca_dir / f"{name}.inp"
            comment = (
                f"split={SPLIT} sample_id={SAMPLE_ID} rxn={sample.rxn} "
                f"frame_id={frame_id} xi={xi:.8f} q_nh={q_nh:.8f} q_oh={q_oh:.8f}"
            )
            write_xyz(xyz_path, symbols, coords, comment)
            write_orca_input(inp_path, symbols, coords, route, charge, multiplicity)
            row = {
                "grid_id": frame_id,
                "frame_id": frame_id,
                "split": SPLIT,
                "sample_id": SAMPLE_ID,
                "formula": str(sample.formula),
                "rxn": str(sample.rxn),
                "q_nh": q_nh,
                "q_oh": q_oh,
                "xi": xi,
                "xyz_path": str(xyz_path.resolve()),
                "orca_input_path": str(inp_path.resolve()),
            }
            rows.append(row)
            writer.writerow(row)
            list_handle.write(f"{inp_path.resolve()}\n")
    return rows


def run_model_predictions(
    rows: list[dict[str, object]],
    coords_frames: np.ndarray,
    atomic_nums: np.ndarray,
    checkpoint: Path,
    hessian_method: str,
    model_label: str,
    device: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    calculator = EquiformerTorchCalculator(
        checkpoint_path=str(checkpoint),
        hessian_method=hessian_method,
        device=device,
    )
    atomic_nums_t = torch.tensor(atomic_nums, dtype=torch.long, device=device)
    n_atoms = int(atomic_nums.size)

    out_rows: list[dict[str, object]] = []
    energies: list[float] = []
    forces_rows: list[np.ndarray] = []
    hessian_rows: list[np.ndarray] = []

    for idx, (row, coords_np) in enumerate(zip(rows, coords_frames, strict=True), start=1):
        coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
        out = calculator.predict(coords=coords, atomic_nums=atomic_nums_t, do_hessian=True)
        energy = out.get("energy")
        energy_value = float(energy.detach().cpu().reshape(-1)[0].item()) if isinstance(energy, torch.Tensor) else float(energy)
        forces = out["forces"].reshape(n_atoms, 3).detach().cpu().numpy().astype(np.float64)
        hessian = out["hessian"].reshape(3 * n_atoms, 3 * n_atoms).detach().cpu().numpy().astype(np.float64)

        energies.append(energy_value)
        forces_rows.append(forces)
        hessian_rows.append(hessian)
        out_rows.append(
            row
            | {
                f"{model_label}_energy": energy_value,
                f"{model_label}_fmax": float(np.abs(forces).max()),
                f"{model_label}_hessian_asym_rel": float(
                    np.linalg.norm(hessian - hessian.T) / (np.linalg.norm(hessian) + 1e-30)
                ),
            }
        )
        if idx % 10 == 0 or idx == len(rows):
            print(f"[{model_label}] evaluated {idx}/{len(rows)} frames", flush=True)

    arrays = {
        "atomic_numbers": atomic_nums.astype(int),
        "frame_id": np.asarray([row["frame_id"] for row in rows], dtype=int),
        "q_nh": np.asarray([row["q_nh"] for row in rows], dtype=float),
        "q_oh": np.asarray([row["q_oh"] for row in rows], dtype=float),
        "xi": np.asarray([row["xi"] for row in rows], dtype=float),
        "coords_angstrom": coords_frames.astype(np.float64),
        "energies": np.asarray(energies, dtype=np.float64),
        "forces": np.stack(forces_rows),
        "hessians_cartesian": np.stack(hessian_rows),
    }
    return pd.DataFrame(out_rows), arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=project_root() / "data" / "transition1x.h5")
    parser.add_argument(
        "--trajectory-xyz",
        type=Path,
        default=project_root() / "runs" / "glycine_pt_xyzrender" / "xyz" / "reaction_path.xyz",
    )
    parser.add_argument("--output-dir", type=Path, default=project_root() / "runs" / "glycine_pt_mep_73")
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "ckpt" / "hip_v2.ckpt")
    parser.add_argument("--hessian-method", choices=["predict", "autograd"], default="predict")
    parser.add_argument("--model-label", default="hip_v2")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--orca-route", default=DEFAULT_ORCA_ROUTE)
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Transition1xDataset(str(args.h5), split=SPLIT, max_samples=SAMPLE_ID + 1)
    sample = dataset[SAMPLE_ID]
    atomic_nums = sample.z.detach().cpu().numpy().astype(int)
    symbols = symbols_from_z(atomic_nums)
    coords_frames = read_trajectory_xyz(args.trajectory_xyz, int(atomic_nums.size))
    rows = prepare_inputs(out_dir, coords_frames, symbols, sample, args.orca_route, args.charge, args.multiplicity)

    metadata = {
        "split": SPLIT,
        "sample_id": SAMPLE_ID,
        "formula": str(sample.formula),
        "rxn": str(sample.rxn),
        "trajectory_xyz": str(args.trajectory_xyz),
        "n_frames": int(coords_frames.shape[0]),
        "orca_route": args.orca_route,
        "charge": int(args.charge),
        "multiplicity": int(args.multiplicity),
    }

    if not args.skip_model:
        model_label = safe_label(args.model_label)
        output_prefix = safe_label(args.output_prefix or model_label)
        predictions, arrays = run_model_predictions(
            rows,
            coords_frames,
            atomic_nums,
            args.checkpoint,
            args.hessian_method,
            model_label,
            args.device,
        )
        predictions.to_parquet(out_dir / f"{output_prefix}_predictions.parquet", index=False)
        predictions.to_csv(out_dir / f"{output_prefix}_predictions.csv", index=False)
        np.savez_compressed(out_dir / f"{output_prefix}_arrays.npz", **arrays)
        metadata.update(
            {
                "checkpoint": str(args.checkpoint),
                "model_label": model_label,
                "hessian_method": args.hessian_method,
                "output_prefix": output_prefix,
            }
        )
        print(f"[{model_label}] wrote {out_dir / f'{output_prefix}_arrays.npz'}", flush=True)

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(rows)} MEP geometries to {out_dir}", flush=True)
    print(f"Manifest: {out_dir / 'scan_manifest.csv'}", flush=True)
    print(f"ORCA input list: {out_dir / 'orca_input_list.txt'}", flush=True)


if __name__ == "__main__":
    main()
