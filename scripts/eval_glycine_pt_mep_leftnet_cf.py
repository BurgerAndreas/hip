#!/usr/bin/env python
"""Evaluate HORM LeftNet-CF on a glycine proton-transfer MEP run."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
SYMBOLS = ["H", "C", "N", "O", "F"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0].strip())
    symbols: list[str] = []
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for idx, line in enumerate(lines[2 : 2 + n_atoms]):
        fields = line.split()
        symbols.append(fields[0])
        coords[idx] = [float(fields[1]), float(fields[2]), float(fields[3])]
    return symbols, coords


def one_hot_from_symbols(symbols: list[str], device: torch.device) -> torch.Tensor:
    arr = np.zeros((len(symbols), len(SYMBOLS)), dtype=np.float32)
    for idx, symbol in enumerate(symbols):
        arr[idx, SYMBOLS.index(symbol)] = 1.0
    return torch.tensor(arr, dtype=torch.float32, device=device)


def make_batch(symbols: list[str], coords: np.ndarray, device: torch.device) -> Batch:
    atomic_numbers = torch.tensor([SYMBOL_TO_Z[symbol] for symbol in symbols], dtype=torch.long, device=device)
    data = Data(
        pos=torch.tensor(coords, dtype=torch.float32, device=device),
        one_hot=one_hot_from_symbols(symbols, device),
        charges=atomic_numbers.to(torch.float32),
        natoms=torch.tensor([len(symbols)], dtype=torch.long, device=device),
        ae=torch.zeros(1, dtype=torch.float32, device=device),
        energy=torch.zeros(1, dtype=torch.float32, device=device),
    )
    return Batch.from_data_list([data]).to(device)


def load_model(horm_dir: Path, checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    sys.path.insert(0, str(horm_dir.resolve()))
    from training_module import PotentialModule  # noqa: PLC0415

    module = PotentialModule.load_from_checkpoint(str(checkpoint), strict=False, map_location=device)
    model = module.potential.to(device)
    model.eval()
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    if model_name != "LEFTNet":
        raise ValueError(f"Expected a LeftNet-CF checkpoint with model name LEFTNet, got {model_name}")
    return model, model_name


def compute_hessian_from_forces(coords: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
    rows = []
    for force_component in forces.reshape(-1):
        rows.append(torch.autograd.grad((-force_component).sum(), coords, retain_graph=True)[0])
    return torch.stack(rows).reshape(coords.numel(), coords.numel())


def predict(model: torch.nn.Module, symbols: list[str], coords: np.ndarray, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    batch = make_batch(symbols, coords, device)
    batch.pos.requires_grad_(True)
    energy, forces = model.forward_autograd(batch)
    hessian = compute_hessian_from_forces(batch.pos, forces)
    energy_ev = float(energy.detach().cpu().reshape(-1)[0].item())
    forces_np = forces.detach().cpu().numpy().reshape(len(symbols), 3).astype(np.float64)
    hessian_np = hessian.detach().cpu().numpy().astype(np.float64)
    return energy_ev, forces_np, hessian_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mep-dir", type=Path, default=project_root() / "runs" / "glycine_pt_mep_145")
    parser.add_argument("--horm-dir", type=Path, default=project_root().parent / "HORM")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    mep_dir = args.mep_dir
    checkpoint = args.checkpoint or args.horm_dir / "ckpt" / "left.ckpt"
    output = args.output or mep_dir / "leftnet_cf_arrays.npz"
    summary_csv = args.summary_csv or mep_dir / "leftnet_cf_predictions.csv"

    manifest = pd.read_csv(mep_dir / "scan_manifest.csv").sort_values("grid_id").reset_index(drop=True)
    if args.max_frames is not None:
        manifest = manifest.iloc[: args.max_frames].copy()
    model, model_name = load_model(args.horm_dir, checkpoint, device)
    print(f"Loaded {model_name} from {checkpoint} on {device}", flush=True)

    energies: list[float] = []
    forces_rows: list[np.ndarray] = []
    hessian_rows: list[np.ndarray] = []
    coords_rows: list[np.ndarray] = []
    atomic_numbers: np.ndarray | None = None
    rows: list[dict[str, float | int | str]] = []
    start = time.perf_counter()

    for idx, row in enumerate(manifest.to_dict(orient="records"), start=1):
        symbols, coords = read_xyz(Path(row["xyz_path"]))
        if atomic_numbers is None:
            atomic_numbers = np.asarray([SYMBOL_TO_Z[symbol] for symbol in symbols], dtype=int)
        energy_ev, forces, hessian = predict(model, symbols, coords, device)
        energies.append(energy_ev)
        forces_rows.append(forces)
        hessian_rows.append(hessian)
        coords_rows.append(coords)
        rows.append(
            {
                "grid_id": int(row["grid_id"]),
                "frame_id": int(row["frame_id"]),
                "xi": float(row["xi"]),
                "q_nh": float(row["q_nh"]),
                "q_oh": float(row["q_oh"]),
                "energy_ev": energy_ev,
                "force_norm_ev_ang": float(np.linalg.norm(forces.reshape(-1))),
                "fmax_ev_ang": float(np.max(np.abs(forces))),
                "hessian_asymmetry_mae_ev_ang2": float(np.mean(np.abs(hessian - hessian.T))),
            }
        )
        if idx % 10 == 0 or idx == len(manifest):
            elapsed = time.perf_counter() - start
            print(f"[leftnet-cf] evaluated {idx}/{len(manifest)} frames in {elapsed:.1f}s", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        atomic_numbers=np.asarray(atomic_numbers, dtype=int),
        frame_id=manifest["frame_id"].to_numpy(dtype=int),
        q_nh=manifest["q_nh"].to_numpy(dtype=float),
        q_oh=manifest["q_oh"].to_numpy(dtype=float),
        xi=manifest["xi"].to_numpy(dtype=float),
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
