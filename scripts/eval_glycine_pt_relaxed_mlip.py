#!/usr/bin/env python
"""Evaluate one MLIP checkpoint on the relaxed glycine PT scan geometries.

Reads the GFN2-xTB relaxed geometries produced by
``scripts/glycine_pt_scan_relaxed.py`` (from ``xtb_relaxed_arrays.npz`` +
``scan_manifest.csv``) and computes energy, forces, and the Cartesian Hessian at
every node with ``EquiformerTorchCalculator``. Output arrays mirror the layout
used by the rigid-scan pipeline (keys ``hessians_cartesian`` and ``forces``) so
downstream diagnostics can consume them directly. The geometries are identical
across methods, so HIP-predicted and autograd Hessians are compared on the same
points.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import pandas as pd
import torch

from hip.equiformer_torch_calculator import EquiformerTorchCalculator


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return safe.strip("_") or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=project_root() / "runs" / "glycine_pt_scan_relaxed")
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "ckpt" / "hip_v2.ckpt")
    parser.add_argument("--hessian-method", choices=["predict", "autograd"], default="predict")
    parser.add_argument("--model-label", default="hip_v2")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan_dir = args.scan_dir
    arrays_path = scan_dir / "xtb_relaxed_arrays.npz"
    manifest_path = scan_dir / "scan_manifest.csv"

    data = np.load(arrays_path)
    coords_frames = np.asarray(data["coords_angstrom"], dtype=np.float64)
    atomic_nums = np.asarray(data["atomic_numbers"], dtype=int)
    manifest = pd.read_csv(manifest_path).sort_values("grid_id").reset_index(drop=True)
    if len(manifest) != coords_frames.shape[0]:
        raise ValueError(
            f"manifest rows ({len(manifest)}) != coords frames ({coords_frames.shape[0]})"
        )

    n_atoms = int(atomic_nums.size)
    model_label = safe_label(args.model_label)
    output_prefix = safe_label(args.output_prefix or model_label)

    calculator = EquiformerTorchCalculator(
        checkpoint_path=str(args.checkpoint),
        hessian_method=args.hessian_method,
        device=args.device,
    )
    atomic_nums_t = torch.tensor(atomic_nums, dtype=torch.long, device=args.device)

    energies: list[float] = []
    forces_rows: list[np.ndarray] = []
    hessian_rows: list[np.ndarray] = []
    fmax_rows: list[float] = []
    asym_rows: list[float] = []

    n = coords_frames.shape[0]
    for idx, coords_np in enumerate(coords_frames, start=1):
        coords = torch.tensor(coords_np, dtype=torch.float32, device=args.device)
        out = calculator.predict(coords=coords, atomic_nums=atomic_nums_t, do_hessian=True)
        energy = out.get("energy")
        energy_value = (
            float(energy.detach().cpu().reshape(-1)[0].item())
            if isinstance(energy, torch.Tensor)
            else float(energy)
        )
        forces = out["forces"].reshape(n_atoms, 3).detach().cpu().numpy().astype(np.float64)
        hessian = out["hessian"].reshape(3 * n_atoms, 3 * n_atoms).detach().cpu().numpy().astype(np.float64)

        energies.append(energy_value)
        forces_rows.append(forces)
        hessian_rows.append(hessian)
        fmax_rows.append(float(np.abs(forces).max()))
        asym_rows.append(float(np.linalg.norm(hessian - hessian.T) / (np.linalg.norm(hessian) + 1e-30)))
        if idx % 25 == 0 or idx == n:
            print(f"[{model_label}] evaluated {idx}/{n}", flush=True)

    energies_arr = np.asarray(energies, dtype=np.float64)
    arrays = {
        "atomic_numbers": atomic_nums,
        "grid_id": manifest["grid_id"].to_numpy(dtype=int),
        "s": manifest["s"].to_numpy(dtype=float),
        "sigma": manifest["sigma"].to_numpy(dtype=float),
        "q_nh": manifest["q_nh_relaxed"].to_numpy(dtype=float),
        "q_oh": manifest["q_oh_relaxed"].to_numpy(dtype=float),
        "coords_angstrom": coords_frames,
        "energies": energies_arr,
        "forces": np.stack(forces_rows),
        "hessians_cartesian": np.stack(hessian_rows),
    }
    out_arrays = scan_dir / f"{output_prefix}_arrays.npz"
    np.savez_compressed(out_arrays, **arrays)

    predictions = manifest.copy()
    predictions[f"{model_label}_energy"] = energies_arr
    predictions[f"{model_label}_energy_relative_kcalmol"] = (
        energies_arr - energies_arr.min()
    ) * 23.060541945329334
    predictions[f"{model_label}_fmax"] = np.asarray(fmax_rows, dtype=float)
    predictions[f"{model_label}_hessian_asym_rel"] = np.asarray(asym_rows, dtype=float)
    predictions.to_csv(scan_dir / f"{output_prefix}_predictions.csv", index=False)

    print(f"[{model_label}] wrote {out_arrays}", flush=True)
    print(f"[{model_label}] wrote {scan_dir / f'{output_prefix}_predictions.csv'}", flush=True)


if __name__ == "__main__":
    main()
