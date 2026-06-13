#!/usr/bin/env python
"""Prepare a glycine proton-transfer 2D scan and run HIP-v2 Hessians."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.frequency_analysis import analyze_frequencies_torch
from hip.transition1x_dataset import Transition1xDataset


SPLIT = "test"
SAMPLE_ID = 5
N_ATOM = 4
O_ATOM = 3
H_ATOM = 9
NEG_EIGVAL_THRESHOLD = 1e-6


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_transition1x_h5() -> Path:
    return _project_root() / "data" / "transition1x.h5"


def _default_checkpoint() -> Path:
    return _project_root() / "ckpt" / "hip_v2.ckpt"


def _float_grid(start: float, stop: float, n: int) -> np.ndarray:
    if n < 2:
        return np.array([start], dtype=float)
    return np.linspace(start, stop, n, dtype=float)


def _symbols(atomic_nums: np.ndarray) -> list[str]:
    z_to_symbol = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}
    return [z_to_symbol[int(z)] for z in atomic_nums.tolist()]


def _perpendicular_reference(coords: np.ndarray) -> np.ndarray:
    n_pos = coords[N_ATOM]
    o_pos = coords[O_ATOM]
    h_pos = coords[H_ATOM]
    axis = o_pos - n_pos
    axis = axis / np.linalg.norm(axis)
    h_rel = h_pos - n_pos
    perp = h_rel - np.dot(h_rel, axis) * axis
    norm = np.linalg.norm(perp)
    if norm > 1e-8:
        return perp / norm

    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, axis)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    perp = trial - np.dot(trial, axis) * axis
    return perp / np.linalg.norm(perp)


def place_transfer_hydrogen(
    ts_coords: np.ndarray,
    q_nh: float,
    q_oh: float,
    perp_ref: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray | None:
    """Move H9 to satisfy d(N4,H9)=q_nh and d(O3,H9)=q_oh."""
    coords = np.array(ts_coords, dtype=float, copy=True)
    n_pos = coords[N_ATOM]
    o_pos = coords[O_ATOM]
    axis_vec = o_pos - n_pos
    r_no = float(np.linalg.norm(axis_vec))
    axis = axis_vec / r_no

    if q_nh + q_oh < r_no - tol or abs(q_nh - q_oh) > r_no + tol:
        return None

    x_along = (q_nh**2 - q_oh**2 + r_no**2) / (2.0 * r_no)
    h2 = max(0.0, q_nh**2 - x_along**2)
    coords[H_ATOM] = n_pos + x_along * axis + np.sqrt(h2) * perp_ref
    return coords


def write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str) -> None:
    with path.open("w") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{comment}\n")
        for symbol, xyz in zip(symbols, coords, strict=False):
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
        handle.write(f"{route}\n\n")
        handle.write("%pal nprocs 16 end\n")
        handle.write("%maxcore 4000\n\n")
        handle.write(f"* xyz {charge} {multiplicity}\n")
        for symbol, xyz in zip(symbols, coords, strict=False):
            handle.write(f"  {symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")
        handle.write("*\n")


def _as_single_structure(tensor: torch.Tensor, n_atoms: int) -> torch.Tensor:
    if tensor.dim() >= 1 and tensor.shape[0] == 1 and tensor.numel() == n_atoms * 3:
        return tensor.squeeze(0)
    return tensor


def force_max(forces: torch.Tensor) -> float:
    if forces.dim() == 3 and forces.shape[0] == 1:
        forces = forces[0]
    return float(forces.reshape(-1).abs().max().item())


def count_negative_eigenvalues(evals_vib: torch.Tensor) -> int:
    return int((evals_vib < -NEG_EIGVAL_THRESHOLD).sum().item())


def run_hip_predictions(
    grid_rows: list[dict[str, object]],
    atomic_nums: np.ndarray,
    atomsymbols: list[str],
    checkpoint: Path,
    device: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    calculator = EquiformerTorchCalculator(
        checkpoint_path=str(checkpoint),
        hessian_method="predict",
        device=device,
    )
    atomic_nums_t = torch.tensor(atomic_nums, dtype=torch.long, device=device)

    rows: list[dict[str, object]] = []
    hessians: list[np.ndarray] = []
    forces_out: list[np.ndarray] = []
    eigvec0_out: list[np.ndarray] = []
    n_atoms = int(atomic_nums.size)

    for row in grid_rows:
        coords_np = np.asarray(row["coords"], dtype=np.float32).reshape(n_atoms, 3)
        coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
        out = calculator.predict(coords=coords, atomic_nums=atomic_nums_t, do_hessian=True)

        forces = _as_single_structure(out["forces"], n_atoms).reshape(n_atoms, 3).detach()
        hessian = out["hessian"].reshape(3 * n_atoms, 3 * n_atoms).detach()
        energy = out.get("energy")
        if isinstance(energy, torch.Tensor):
            energy_value = float(energy.detach().cpu().reshape(-1)[0].item())
        else:
            energy_value = float("nan") if energy is None else float(energy)

        freq = analyze_frequencies_torch(
            hessian.double(),
            coords.double(),
            atomsymbols,
            ev_thresh=-NEG_EIGVAL_THRESHOLD,
        )
        evals_vib = freq["eigvals"]
        eigvecs_vib = freq["eigvecs"]
        eig0_vec = eigvecs_vib[:, 0].detach().cpu().numpy()
        hessians.append(hessian.cpu().numpy())
        forces_out.append(forces.cpu().numpy())
        eigvec0_out.append(eig0_vec)

        rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"coords"}
            }
            | {
                "hip_v2_energy": energy_value,
                "hip_v2_fmax": force_max(forces),
                "hip_v2_n_negative_vib": count_negative_eigenvalues(evals_vib),
                "hip_v2_eig0": float(evals_vib[0].detach().cpu().item()),
                "hip_v2_eig1": float(evals_vib[1].detach().cpu().item()),
                "hip_v2_eig2": float(evals_vib[2].detach().cpu().item()),
            }
        )

    return (
        pd.DataFrame(rows),
        np.stack(hessians, axis=0),
        np.stack(forces_out, axis=0),
        np.stack(eigvec0_out, axis=0),
    )


def write_hip_energy_tables(hip_df: pd.DataFrame, out_dir: Path) -> None:
    energy_cols = [
        "grid_id",
        "split",
        "sample_id",
        "formula",
        "rxn",
        "i_nh",
        "i_oh",
        "q_nh",
        "q_oh",
        "xyz_path",
        "orca_input_path",
        "hip_v2_energy",
    ]
    energy_df = hip_df[energy_cols].copy()
    energy_df["hip_v2_energy_relative"] = (
        energy_df["hip_v2_energy"] - energy_df["hip_v2_energy"].min()
    )
    energy_df.to_parquet(out_dir / "hip_v2_energies.parquet", index=False)
    energy_df.to_csv(out_dir / "hip_v2_energies.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=_default_transition1x_h5())
    parser.add_argument("--checkpoint", type=Path, default=_default_checkpoint())
    parser.add_argument("--output-dir", type=Path, default=Path("runs/glycine_pt_scan"))
    parser.add_argument("--q-nh-min", type=float, default=1.0)
    parser.add_argument("--q-nh-max", type=float, default=2.30)
    parser.add_argument("--q-oh-min", type=float, default=0.95)
    parser.add_argument("--q-oh-max", type=float, default=2.55)
    parser.add_argument("--n-grid", type=int, default=9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument(
        "--orca-route",
        default="! wB97X-D3 6-31G(d) TightSCF Grid5 FinalGrid6 Freq",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    xyz_dir = out_dir / "xyz"
    orca_dir = out_dir / "orca_inputs"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    orca_dir.mkdir(parents=True, exist_ok=True)

    dataset = Transition1xDataset(str(args.h5), split=SPLIT, max_samples=SAMPLE_ID + 1)
    sample = dataset[SAMPLE_ID]
    atomic_nums = sample.z.detach().cpu().numpy().astype(int)
    symbols = _symbols(atomic_nums)
    ts_coords = sample.pos_transition.detach().cpu().numpy().reshape(-1, 3)
    reactant_coords = sample.pos_reactant.detach().cpu().numpy().reshape(-1, 3)
    product_coords = sample.pos_product.detach().cpu().numpy().reshape(-1, 3)
    perp_ref = _perpendicular_reference(ts_coords)

    write_xyz(xyz_dir / "reference_reactant.xyz", symbols, reactant_coords, "Transition1x reactant")
    write_xyz(xyz_dir / "reference_ts.xyz", symbols, ts_coords, "Transition1x transition_state")
    write_xyz(xyz_dir / "reference_product.xyz", symbols, product_coords, "Transition1x product")

    q_nh_values = _float_grid(args.q_nh_min, args.q_nh_max, args.n_grid)
    q_oh_values = _float_grid(args.q_oh_min, args.q_oh_max, args.n_grid)
    grid_rows: list[dict[str, object]] = []
    manifest_path = out_dir / "scan_manifest.csv"
    orca_input_list_path = out_dir / "orca_input_list.txt"

    with manifest_path.open("w", newline="") as manifest_handle, orca_input_list_path.open("w") as list_handle:
        fieldnames = [
            "grid_id",
            "i_nh",
            "i_oh",
            "q_nh",
            "q_oh",
            "xyz_path",
            "orca_input_path",
        ]
        writer = csv.DictWriter(manifest_handle, fieldnames=fieldnames)
        writer.writeheader()
        grid_id = 0
        for i_nh, q_nh in enumerate(q_nh_values):
            for i_oh, q_oh in enumerate(q_oh_values):
                coords = place_transfer_hydrogen(ts_coords, float(q_nh), float(q_oh), perp_ref)
                if coords is None:
                    continue
                name = f"grid_{grid_id:04d}_qNH_{q_nh:.3f}_qOH_{q_oh:.3f}"
                xyz_path = xyz_dir / f"{name}.xyz"
                inp_path = orca_dir / f"{name}.inp"
                comment = (
                    f"split={SPLIT} sample_id={SAMPLE_ID} rxn={sample.rxn} "
                    f"q_nh={q_nh:.6f} q_oh={q_oh:.6f}"
                )
                write_xyz(xyz_path, symbols, coords, comment)
                write_orca_input(
                    inp_path,
                    symbols,
                    coords,
                    args.orca_route,
                    args.charge,
                    args.multiplicity,
                )
                row = {
                    "grid_id": grid_id,
                    "split": SPLIT,
                    "sample_id": SAMPLE_ID,
                    "formula": str(sample.formula),
                    "rxn": str(sample.rxn),
                    "i_nh": i_nh,
                    "i_oh": i_oh,
                    "q_nh": float(q_nh),
                    "q_oh": float(q_oh),
                    "coords": coords,
                    "xyz_path": str(xyz_path.resolve()),
                    "orca_input_path": str(inp_path.resolve()),
                }
                grid_rows.append(row)
                writer.writerow({key: row[key] for key in fieldnames})
                list_handle.write(f"{row['orca_input_path']}\n")
                grid_id += 1

    hip_df, hessians, forces, eigvec0 = run_hip_predictions(
        grid_rows,
        atomic_nums,
        symbols,
        args.checkpoint,
        args.device,
    )
    hip_df.to_parquet(out_dir / "hip_v2_predictions.parquet", index=False)
    hip_df.to_csv(out_dir / "hip_v2_predictions.csv", index=False)
    write_hip_energy_tables(hip_df, out_dir)
    np.savez_compressed(
        out_dir / "hip_v2_arrays.npz",
        atomic_numbers=atomic_nums,
        hessians_cartesian=hessians,
        forces=forces,
        unstable_modes_mw=eigvec0,
    )

    metadata = {
        "split": SPLIT,
        "sample_id": SAMPLE_ID,
        "formula": str(sample.formula),
        "rxn": str(sample.rxn),
        "n_atom": N_ATOM,
        "o_atom": O_ATOM,
        "h_atom": H_ATOM,
        "checkpoint": str(args.checkpoint),
        "orca_route": args.orca_route,
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "n_grid_requested": args.n_grid,
        "n_grid_written": len(grid_rows),
        "q_nh_values": q_nh_values.tolist(),
        "q_oh_values": q_oh_values.tolist(),
    }
    pd.Series(metadata).to_json(out_dir / "metadata.json", indent=2)

    print(f"Wrote {len(grid_rows)} grid geometries")
    print(f"Manifest: {manifest_path}")
    print(f"ORCA input list: {orca_input_list_path}")
    print(f"ORCA inputs: {orca_dir}")
    print(f"HIP predictions: {out_dir / 'hip_v2_predictions.parquet'}")
    print(f"HIP arrays: {out_dir / 'hip_v2_arrays.npz'}")


if __name__ == "__main__":
    main()
