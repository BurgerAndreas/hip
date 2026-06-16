#!/usr/bin/env python
"""Generate T1x-val line-scan geometries for DFT/EqV2 force spectra.

The default is 100 geometries x 2 directions x 51 points = 10,200 force
evaluations. Geometry indices follow the same deterministic random order used by
``scripts/eval_horm.py``: torch.randperm(len(dataset), seed=42).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hip.ff_lmdb import LmdbDataset, Z_TO_ATOM_SYMBOL
from hip.path_config import fix_dataset_path


DEFAULT_ORCA_ROUTE = "! wB97X-D3 6-31G(d) TightSCF EnGrad"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def encode_float(value: float) -> str:
    return f"{value:+.6f}".replace("+", "p").replace("-", "m").replace(".", "p")


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = direction - direction.mean(axis=0, keepdims=True)
    norm = float(np.linalg.norm(direction.reshape(-1)))
    if norm < 1e-12:
        raise ValueError("Generated near-zero direction")
    return direction / norm


def random_direction(n_atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalize_direction(rng.normal(size=(n_atoms, 3)))


def write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str) -> None:
    with path.open("w") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{comment}\n")
        for symbol, xyz in zip(symbols, coords, strict=True):
            handle.write(f"{symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


def write_orca_input(
    path: Path,
    symbols: list[str],
    coords: np.ndarray,
    route: str,
    charge: int,
    multiplicity: int,
    nprocs: int,
    maxcore: int,
) -> None:
    with path.open("w") as handle:
        handle.write(f"{route}\n\n")
        handle.write(f"%pal nprocs {nprocs} end\n")
        handle.write(f"%maxcore {maxcore}\n\n")
        handle.write(f"* xyz {charge} {multiplicity}\n")
        for symbol, xyz in zip(symbols, coords, strict=True):
            handle.write(f"{symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")
        handle.write("*\n")


def eval_horm_indices(dataset_len: int, max_samples: int, seed: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(dataset_len, generator=generator).tolist()
    return [int(idx) for idx in indices[:max_samples]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ts1x-val.lmdb")
    parser.add_argument("--candidate-samples", type=int, default=1000)
    parser.add_argument("--n-geometries", type=int, default=100)
    parser.add_argument("--n-directions", type=int, default=2)
    parser.add_argument("--n-points", type=int, default=51)
    parser.add_argument("--amplitude", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "runs" / "t1x_val_force_spectra_100x2x51",
    )
    parser.add_argument("--orca-route", default=DEFAULT_ORCA_ROUTE)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--orca-nprocs", type=int, default=4)
    parser.add_argument("--orca-maxcore", type=int, default=1000)
    args = parser.parse_args()

    if args.n_geometries > args.candidate_samples:
        raise ValueError("--n-geometries must be <= --candidate-samples")
    if args.n_points < 3 or args.n_points % 2 == 0:
        raise ValueError("--n-points must be an odd integer >= 3")

    out_dir = args.output_dir
    xyz_dir = out_dir / "xyz"
    orca_dir = out_dir / "orca_inputs"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    orca_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = fix_dataset_path(args.dataset)
    dataset = LmdbDataset(dataset_path)
    candidate_indices = eval_horm_indices(len(dataset), args.candidate_samples, args.seed)
    selected_indices = candidate_indices[: args.n_geometries]
    lambdas = np.linspace(-args.amplitude, args.amplitude, args.n_points)

    rows: list[dict[str, object]] = []
    input_paths: list[str] = []
    for geom_rank, dataset_idx in enumerate(selected_indices):
        data = dataset[dataset_idx]
        coords0 = data.pos.detach().cpu().numpy().astype(np.float64)
        z = data.z.detach().cpu().numpy().astype(int)
        symbols = [Z_TO_ATOM_SYMBOL[int(zz)] for zz in z]
        n_atoms = int(len(symbols))

        for direction_id in range(args.n_directions):
            direction_seed = args.seed * 1_000_000 + int(dataset_idx) * 10 + direction_id
            direction = random_direction(n_atoms, direction_seed)
            direction_path = out_dir / f"direction_g{geom_rank:04d}_d{direction_id}.npy"
            np.save(direction_path, direction)

            for point_id, lam in enumerate(lambdas):
                coords = coords0 + float(lam) * direction
                stem = (
                    f"g{geom_rank:04d}_idx{dataset_idx:06d}_"
                    f"d{direction_id}_p{point_id:03d}_lam{encode_float(float(lam))}"
                )
                xyz_path = xyz_dir / f"{stem}.xyz"
                inp_path = orca_dir / f"{stem}.inp"
                comment = (
                    f"dataset={Path(dataset_path).name} dataset_idx={dataset_idx} "
                    f"geom_rank={geom_rank} direction_id={direction_id} lambda={lam:.8f}"
                )
                write_xyz(xyz_path, symbols, coords, comment)
                write_orca_input(
                    inp_path,
                    symbols,
                    coords,
                    args.orca_route,
                    args.charge,
                    args.multiplicity,
                    args.orca_nprocs,
                    args.orca_maxcore,
                )
                input_paths.append(str(inp_path.resolve()))
                rows.append(
                    {
                        "scan_point_id": len(rows),
                        "geom_rank": geom_rank,
                        "dataset_idx": dataset_idx,
                        "n_atoms": n_atoms,
                        "direction_id": direction_id,
                        "direction_kind": f"random_translation_free_{direction_id}",
                        "direction_seed": direction_seed,
                        "direction_path": str(direction_path.resolve()),
                        "point_id": point_id,
                        "lambda_ang": float(lam),
                        "xyz_path": str(xyz_path.resolve()),
                        "orca_input_path": str(inp_path.resolve()),
                    }
                )

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "scan_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    manifest.to_parquet(out_dir / "scan_manifest.parquet", index=False)
    with (out_dir / "orca_input_list.txt").open("w") as handle:
        handle.write("\n".join(input_paths))
        handle.write("\n")
    with (out_dir / "generation_config.json").open("w") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "dataset_path": dataset_path,
                "candidate_samples": args.candidate_samples,
                "n_geometries": args.n_geometries,
                "n_directions": args.n_directions,
                "n_points": args.n_points,
                "amplitude": args.amplitude,
                "seed": args.seed,
                "orca_route": args.orca_route,
                "charge": args.charge,
                "multiplicity": args.multiplicity,
                "orca_nprocs": args.orca_nprocs,
                "orca_maxcore": args.orca_maxcore,
                "selected_indices": selected_indices,
            },
            handle,
            indent=2,
        )

    print(f"Wrote {len(manifest)} scan points")
    print(f"Manifest: {manifest_path}")
    print(f"ORCA input list: {out_dir / 'orca_input_list.txt'}")


if __name__ == "__main__":
    main()
