#!/usr/bin/env python
"""Evaluate an EquiformerV2 checkpoint on a glycine proton-transfer MEP run."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from hip.equiformer_torch_calculator import EquiformerTorchCalculator


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_xyz(path: Path) -> tuple[np.ndarray, int]:
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0].strip())
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for idx, line in enumerate(lines[2 : 2 + n_atoms]):
        fields = line.split()
        coords[idx] = [float(fields[1]), float(fields[2]), float(fields[3])]
    return coords, n_atoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mep-dir", type=Path, default=project_root() / "runs" / "glycine_pt_mep_145")
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "ckpt" / "eqv2.ckpt")
    parser.add_argument("--hessian-method", choices=["autograd", "predict"], default="autograd")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    mep_dir = args.mep_dir
    checkpoint = args.checkpoint
    stem = checkpoint.stem
    output = args.output or mep_dir / f"{stem}_autograd_arrays.npz"
    summary_csv = args.summary_csv or mep_dir / f"{stem}_autograd_predictions.csv"

    manifest = pd.read_csv(mep_dir / "scan_manifest.csv").sort_values("grid_id").reset_index(drop=True)
    if args.max_frames is not None:
        manifest = manifest.iloc[: args.max_frames].copy()

    calculator = EquiformerTorchCalculator(
        checkpoint_path=str(checkpoint),
        hessian_method=args.hessian_method,
        device=device,
    )

    atomic_numbers: np.ndarray | None = None
    for candidate in (
        mep_dir / "orca_vib_cache.npz",
        mep_dir / "hip_v2_arrays.npz",
        mep_dir / "leftnet_cf_arrays.npz",
    ):
        if not candidate.exists():
            continue
        with np.load(candidate) as data:
            if "atomic_numbers" in data.files:
                atomic_numbers = np.asarray(data["atomic_numbers"], dtype=int)
                break
    if atomic_numbers is None:
        raise FileNotFoundError(f"Could not find atomic_numbers in {mep_dir}")
    atomic_nums_t = torch.tensor(atomic_numbers, dtype=torch.long, device=device)
    n_atoms = int(atomic_numbers.size)

    energies: list[float] = []
    forces_rows: list[np.ndarray] = []
    hessian_rows: list[np.ndarray] = []
    coords_rows: list[np.ndarray] = []
    rows: list[dict[str, float | int | str]] = []
    start = time.perf_counter()

    for idx, row in enumerate(manifest.to_dict(orient="records"), start=1):
        coords_np, _ = read_xyz(Path(row["xyz_path"]))
        coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
        out = calculator.predict(coords=coords, atomic_nums=atomic_nums_t, do_hessian=True)
        energy = out.get("energy")
        energy_value = (
            float(energy.detach().cpu().reshape(-1)[0].item())
            if isinstance(energy, torch.Tensor)
            else float(energy)
        )
        forces = out["forces"].reshape(n_atoms, 3).detach().cpu().numpy().astype(np.float64)
        hessian = out["hessian"].reshape(3 * n_atoms, 3 * n_atoms).detach().cpu().numpy().astype(np.float64)

        q_nh = float(row["q_nh"])
        q_oh = float(row["q_oh"])
        xi = float(row["xi"]) if "xi" in row else q_nh - q_oh
        frame_id = int(row["frame_id"]) if "frame_id" in row else int(row["grid_id"])

        energies.append(energy_value)
        forces_rows.append(forces)
        hessian_rows.append(hessian)
        coords_rows.append(coords_np)
        rows.append(
            {
                "grid_id": int(row["grid_id"]),
                "frame_id": frame_id,
                "xi": xi,
                "q_nh": q_nh,
                "q_oh": q_oh,
                "energy_ev": energy_value,
                "force_norm_ev_ang": float(np.linalg.norm(forces.reshape(-1))),
                "fmax_ev_ang": float(np.max(np.abs(forces))),
                "hessian_asymmetry_mae_ev_ang2": float(np.mean(np.abs(hessian - hessian.T))),
            }
        )
        if idx % 10 == 0 or idx == len(manifest):
            elapsed = time.perf_counter() - start
            print(f"[{checkpoint.stem}] evaluated {idx}/{len(manifest)} frames in {elapsed:.1f}s", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        atomic_numbers=atomic_numbers,
        frame_id=(
            manifest["frame_id"].to_numpy(dtype=int)
            if "frame_id" in manifest.columns
            else manifest["grid_id"].to_numpy(dtype=int)
        ),
        q_nh=manifest["q_nh"].to_numpy(dtype=float),
        q_oh=manifest["q_oh"].to_numpy(dtype=float),
        xi=(
            manifest["xi"].to_numpy(dtype=float)
            if "xi" in manifest.columns
            else (manifest["q_nh"] - manifest["q_oh"]).to_numpy(dtype=float)
        ),
        coords_angstrom=np.stack(coords_rows),
        energies=np.asarray(energies, dtype=np.float64),
        forces=np.stack(forces_rows),
        hessians_cartesian=np.stack(hessian_rows),
    )
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"Wrote {output}", flush=True)
    print(f"Wrote {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
