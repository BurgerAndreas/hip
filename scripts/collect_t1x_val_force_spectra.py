#!/usr/bin/env python
"""Collect ORCA EnGrad outputs for the T1x-val force-spectra scan.

ORCA `.engrad` gradients are in Hartree/bohr. This collector stores DFT forces as
`-gradient` converted to eV/Angstrom and writes one compact NPZ per
geometry/direction line, aligned to `scan_manifest.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANG = HARTREE_TO_EV / BOHR_TO_ANG


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_SCAN_DIR = project_root() / "runs" / "t1x_val_force_spectra_100x2x51"


def parse_engrad(path: Path) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    values: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            values.append(stripped)

    n_atoms = int(values[0])
    energy_hartree = float(values[1])
    grad_start = 2
    grad_stop = grad_start + 3 * n_atoms
    gradient = np.array([float(x) for x in values[grad_start:grad_stop]], dtype=np.float64)
    coord_values = values[grad_stop : grad_stop + n_atoms]
    coords_bohr = np.array([[float(x) for x in row.split()[1:4]] for row in coord_values], dtype=np.float64)
    atomic_numbers = np.array([int(row.split()[0]) for row in coord_values], dtype=np.int64)
    return n_atoms, energy_hartree, gradient.reshape(n_atoms, 3), coords_bohr, atomic_numbers


def output_for_input(input_path: str, output_dir: Path, suffix: str) -> Path:
    return output_dir / f"{Path(input_path).stem}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=DEFAULT_SCAN_DIR,
    )
    parser.add_argument("--orca-output-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    scan_dir = args.scan_dir
    orca_output_dir = args.orca_output_dir or scan_dir / "orca_outputs"
    output_dir = args.output_dir or scan_dir / "dft_force_outputs"
    line_dir = output_dir / "line_npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    line_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(scan_dir / "scan_manifest.csv")

    rows: list[dict[str, object]] = []
    parsed_by_scan_point: dict[int, dict[str, object]] = {}
    for row in manifest.to_dict(orient="records"):
        engrad_path = output_for_input(str(row["orca_input_path"]), orca_output_dir, ".engrad")
        out_path = output_for_input(str(row["orca_input_path"]), orca_output_dir, ".out")
        if not engrad_path.exists():
            raise FileNotFoundError(f"Missing ORCA EnGrad file for scan_point_id={row['scan_point_id']}: {engrad_path}")

        n_atoms, energy_hartree, gradient_hartree_bohr, coords_bohr, atomic_numbers = parse_engrad(engrad_path)
        if n_atoms != int(row["n_atoms"]):
            raise ValueError(f"n_atoms mismatch for {engrad_path}: {n_atoms} != {row['n_atoms']}")
        forces_ev_ang = -gradient_hartree_bohr * HARTREE_PER_BOHR_TO_EV_PER_ANG

        parsed = {
            "energy_hartree": energy_hartree,
            "gradient_hartree_bohr": gradient_hartree_bohr,
            "forces_ev_ang": forces_ev_ang,
            "coords_bohr": coords_bohr,
            "coords_ang": coords_bohr * BOHR_TO_ANG,
            "atomic_numbers": atomic_numbers,
            "engrad_path": str(engrad_path.resolve()),
            "orca_output_path": str(out_path.resolve()),
        }
        parsed_by_scan_point[int(row["scan_point_id"])] = parsed
        rows.append(
            {
                "scan_point_id": int(row["scan_point_id"]),
                "geom_rank": int(row["geom_rank"]),
                "dataset_idx": int(row["dataset_idx"]),
                "direction_id": int(row["direction_id"]),
                "point_id": int(row["point_id"]),
                "lambda_ang": float(row["lambda_ang"]),
                "dft_energy_hartree": energy_hartree,
                "dft_force_norm_ev_ang": float(np.linalg.norm(forces_ev_ang.reshape(-1))),
                "dft_fmax_ev_ang": float(np.max(np.abs(forces_ev_ang))),
                "dft_gradient_norm_hartree_bohr": float(np.linalg.norm(gradient_hartree_bohr.reshape(-1))),
                "engrad_path": str(engrad_path.resolve()),
                "orca_output_path": str(out_path.resolve()),
            }
        )

    point_summary = pd.DataFrame(rows).sort_values("scan_point_id")
    point_summary.to_csv(output_dir / "dft_force_summary.csv", index=False)
    point_summary.to_parquet(output_dir / "dft_force_summary.parquet", index=False)

    line_rows: list[dict[str, object]] = []
    grouped = manifest.groupby(["geom_rank", "dataset_idx", "direction_id"], sort=True)
    for (geom_rank, dataset_idx, direction_id), frame in grouped:
        frame = frame.sort_values("point_id")
        scan_ids = frame["scan_point_id"].to_numpy(dtype=np.int64)
        parsed_line = [parsed_by_scan_point[int(scan_id)] for scan_id in scan_ids]
        atomic_numbers = np.asarray(parsed_line[0]["atomic_numbers"], dtype=np.int64)
        for parsed in parsed_line[1:]:
            if not np.array_equal(atomic_numbers, np.asarray(parsed["atomic_numbers"], dtype=np.int64)):
                raise ValueError(f"Atomic numbers changed in line g{geom_rank} d{direction_id}")

        line_path = line_dir / f"g{int(geom_rank):04d}_idx{int(dataset_idx):06d}_d{int(direction_id)}.npz"
        np.savez_compressed(
            line_path,
            lambda_ang=frame["lambda_ang"].to_numpy(dtype=np.float64),
            point_id=frame["point_id"].to_numpy(dtype=np.int64),
            scan_point_id=scan_ids,
            atomic_numbers=atomic_numbers,
            coords_bohr=np.stack([np.asarray(parsed["coords_bohr"], dtype=np.float64) for parsed in parsed_line]),
            coords_ang=np.stack([np.asarray(parsed["coords_ang"], dtype=np.float64) for parsed in parsed_line]),
            energy_hartree=np.asarray([parsed["energy_hartree"] for parsed in parsed_line], dtype=np.float64),
            gradient_hartree_bohr=np.stack([np.asarray(parsed["gradient_hartree_bohr"], dtype=np.float64) for parsed in parsed_line]),
            forces_ev_ang=np.stack([np.asarray(parsed["forces_ev_ang"], dtype=np.float64) for parsed in parsed_line]),
        )
        line_rows.append(
            {
                "geom_rank": int(geom_rank),
                "dataset_idx": int(dataset_idx),
                "direction_id": int(direction_id),
                "n_points": int(len(frame)),
                "n_atoms": int(frame["n_atoms"].iloc[0]),
                "dft_line_npz_path": str(line_path.resolve()),
            }
        )

    line_summary = pd.DataFrame(line_rows)
    line_summary.to_csv(output_dir / "dft_line_summary.csv", index=False)
    line_summary.to_parquet(output_dir / "dft_line_summary.parquet", index=False)

    print(f"Wrote DFT point rows: {len(point_summary)}")
    print(f"Wrote DFT line rows: {len(line_summary)}")
    print(f"Output directory: {output_dir}")
    print(f"Force conversion: forces_ev_ang = -gradient_hartree_bohr * {HARTREE_PER_BOHR_TO_EV_PER_ANG:.12g}")


if __name__ == "__main__":
    main()
