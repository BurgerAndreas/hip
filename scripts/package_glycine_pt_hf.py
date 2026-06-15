#!/usr/bin/env python
"""Package the glycine proton-transfer ORCA scan for Hugging Face upload."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DATASET_NAME = "orca_wb97x_631gd_glycine_pt_nh_oh_scan_1285"


def parse_engrad(path: Path) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    values: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            values.append(stripped)

    n_atoms = int(values[0])
    energy = float(values[1])
    grad_start = 2
    grad_stop = grad_start + 3 * n_atoms
    gradient = np.array([float(x) for x in values[grad_start:grad_stop]], dtype=np.float64)
    coord_values = values[grad_stop: grad_stop + n_atoms]
    coords = np.array([[float(x) for x in row.split()[1:4]] for row in coord_values], dtype=np.float64)
    atomic_numbers = np.array([int(row.split()[0]) for row in coord_values], dtype=np.int64)
    return n_atoms, energy, gradient.reshape(n_atoms, 3), coords, atomic_numbers


def parse_hessian(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "$hessian":
            size = int(lines[i + 1].strip())
            start = i + 2
            break
    else:
        raise ValueError(f"No $hessian block in {path}")

    hessian = np.zeros((size, size), dtype=np.float64)
    current_cols: list[int] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$"):
            break
        parts = stripped.split()
        if all(part.lstrip("-").isdigit() for part in parts):
            current_cols = [int(part) for part in parts]
            continue
        row = int(parts[0])
        vals = [float(x) for x in parts[1:]]
        for col, val in zip(current_cols, vals, strict=False):
            hessian[row, col] = val
    return 0.5 * (hessian + hessian.T)


def copy_xyz_files(scan_dir: Path, out_dir: Path) -> None:
    xyz_dir = out_dir / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((scan_dir / "xyz").glob("grid_*.xyz")):
        shutil.copy2(path, xyz_dir / path.name)
    for name in ["reference_reactant.xyz", "reference_ts.xyz", "reference_product.xyz"]:
        src = scan_dir / "xyz" / name
        if src.exists():
            shutil.copy2(src, xyz_dir / name)


def write_raw_archive(scan_dir: Path, out_dir: Path, orca_output_dir: Path, orca_engrad_dir: Path) -> None:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "orca_outputs.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(orca_output_dir.glob("*")):
            if path.is_file():
                tar.add(path, arcname=f"orca_outputs/{path.name}")
        for path in sorted(orca_engrad_dir.glob("*.engrad")):
            if path.is_file():
                tar.add(path, arcname=f"orca_outputs/{path.name}")


def write_readme(out_dir: Path, n_samples: int) -> None:
    (out_dir / "README.md").write_text(
        f"""# ORCA wB97X/6-31G(d) Glycine Proton-Transfer Scan

This folder contains {n_samples} ORCA analytical Hessian calculations for a 2D scan of
glycine intramolecular proton transfer.

- Source reaction: Transition1x `test` split, `sample_id=5`, `rxn1961`
- Formula: `C2H5NO2`
- Method: wB97X
- Basis: 6-31G(d)
- ORCA version: 6.1.1
- Scan coordinates: `q_nh = d(N4,H9)` and `q_oh = d(O3,H9)`
- Geometry construction: heavy atoms fixed at the Transition1x TS geometry;
  the transferring H atom is moved to satisfy the two scan distances.
- Geometry files: `xyz/*.xyz` in Angstrom
- Targets: `h5/glycine_pt_scan.h5`
- Raw ORCA outputs: `raw/orca_outputs.tar.gz`; Hessian outputs and `.hess`
  files come from the analytical-frequency runs, and `.engrad` files come from
  matching gradient-only ORCA runs.

The HDF5 file uses atomic units for model targets:
`coordinates_bohr`, `energy_hartree`, `gradient_hartree_per_bohr`,
`forces_hartree_per_bohr`, and `hessian_hartree_per_bohr2`.
The scan coordinates are stored as `q_nh_angstrom` and `q_oh_angstrom`.
See `metadata.csv` for the per-geometry mapping.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", type=Path, default=Path("runs/glycine_pt_scan"))
    parser.add_argument(
        "--orca-output-dir",
        type=Path,
        default=None,
        help="Directory containing ORCA Hessian outputs. Defaults to scan-dir/orca_outputs.",
    )
    parser.add_argument(
        "--orca-engrad-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing ORCA .engrad files. Defaults to --orca-output-dir, "
            "or scan-dir/orca_engrad_outputs when present."
        ),
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    scan_dir = args.scan_dir
    orca_output_dir = args.orca_output_dir or scan_dir / "orca_outputs"
    orca_engrad_dir = args.orca_engrad_dir or (
        scan_dir / "orca_engrad_outputs"
        if (scan_dir / "orca_engrad_outputs").exists()
        else orca_output_dir
    )
    out_dir = args.output_dir or Path("runs/hf_upload") / args.dataset_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "h5").mkdir(parents=True)

    manifest = pd.read_csv(scan_dir / "scan_manifest.csv")
    energies = pd.read_csv(scan_dir / "orca_energies.csv")
    rows = manifest.merge(
        energies[
            [
                "grid_id",
                "orca_output_path",
                "orca_terminated_normally",
                "orca_scf_converged",
                "orca_energy_hartree",
                "orca_energy_relative_kcalmol",
            ]
        ],
        on="grid_id",
        how="left",
    ).sort_values("grid_id")

    atomic_numbers_list = []
    coordinates_bohr = []
    energies_hartree = []
    gradients = []
    forces = []
    hessians = []
    xyz_filenames = []
    output_filenames = []

    for row in rows.to_dict(orient="records"):
        stem = Path(row["orca_input_path"]).stem
        engrad_path = orca_engrad_dir / f"{stem}.engrad"
        hess_path = orca_output_dir / f"{stem}.hess"
        n_atoms, energy, gradient, coords_bohr, atomic_numbers = parse_engrad(engrad_path)
        hessian = parse_hessian(hess_path)
        if hessian.shape != (3 * n_atoms, 3 * n_atoms):
            raise ValueError(f"Unexpected Hessian shape for {stem}: {hessian.shape}")

        atomic_numbers_list.append(atomic_numbers)
        coordinates_bohr.append(coords_bohr)
        energies_hartree.append(energy)
        gradients.append(gradient)
        forces.append(-gradient)
        hessians.append(hessian)
        xyz_filenames.append(f"xyz/{Path(row['xyz_path']).name}")
        output_filenames.append(f"raw/orca_outputs.tar.gz:orca_outputs/{stem}.out")

    atomic_numbers_arr = np.stack(atomic_numbers_list)
    coordinates_bohr_arr = np.stack(coordinates_bohr)
    energies_arr = np.array(energies_hartree, dtype=np.float64)
    gradients_arr = np.stack(gradients)
    forces_arr = np.stack(forces)
    hessians_arr = np.stack(hessians)
    symbols = np.array(["C", "O", "C", "O", "N", "H", "H", "H", "H", "H"], dtype="S2")

    h5_path = out_dir / "h5" / "glycine_pt_scan.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("atomic_numbers", data=atomic_numbers_arr, compression="gzip")
        h5.create_dataset("symbols", data=symbols)
        h5.create_dataset("coordinates_bohr", data=coordinates_bohr_arr, compression="gzip")
        h5.create_dataset("energy_hartree", data=energies_arr)
        h5.create_dataset("gradient_hartree_per_bohr", data=gradients_arr, compression="gzip")
        h5.create_dataset("forces_hartree_per_bohr", data=forces_arr, compression="gzip")
        h5.create_dataset("hessian_hartree_per_bohr2", data=hessians_arr, compression="gzip")
        h5.create_dataset("q_nh_angstrom", data=rows["q_nh"].to_numpy(dtype=np.float64))
        h5.create_dataset("q_oh_angstrom", data=rows["q_oh"].to_numpy(dtype=np.float64))
        h5.create_dataset("grid_id", data=rows["grid_id"].to_numpy(dtype=np.int64))
        h5.attrs["dataset"] = args.dataset_name
        h5.attrs["method"] = "wB97X"
        h5.attrs["basis"] = "6-31G(d)"
        h5.attrs["charge"] = 0
        h5.attrs["multiplicity"] = 1

    metadata_rows = []
    for idx, row in rows.reset_index(drop=True).iterrows():
        metadata_rows.append(
            {
                "name": f"glycine_pt_scan/grid_{int(row.grid_id):04d}",
                "job_id": f"grid_{int(row.grid_id):04d}",
                "atoms": 10,
                "source_path": row.xyz_path,
                "source_format": "xyz",
                "charge": 0,
                "multiplicity": 1,
                "method": "wB97X",
                "basis": "6-31G(d)",
                "energy_units": "hartree",
                "forces_units": "hartree/bohr",
                "hessian_units": "hartree/bohr^2",
                "q_nh_angstrom": row.q_nh,
                "q_oh_angstrom": row.q_oh,
                "energy_hartree": energies_arr[idx],
                "energy_relative_kcalmol": row.orca_energy_relative_kcalmol,
                "xyz_path": xyz_filenames[idx],
                "h5_path": "h5/glycine_pt_scan.h5",
                "h5_index": idx,
                "orca_output": output_filenames[idx],
                "terminated_normally": bool(row.orca_terminated_normally),
                "scf_converged": bool(row.orca_scf_converged),
            }
        )

    with (out_dir / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    summary = {
        "dataset": args.dataset_name,
        "num_samples": int(len(rows)),
        "atom_counts": {"min": 10, "max": 10},
        "source": {
            "transition1x_split": "test",
            "transition1x_sample_id": 5,
            "rxn": "rxn1961",
            "formula": "C2H5NO2",
        },
        "method": "wB97X",
        "basis": "6-31G(d)",
        "orca_version": "6.1.1",
        "scan_coordinates": {
            "q_nh_angstrom": {"atoms": [4, 9], "description": "N4-H9 distance"},
            "q_oh_angstrom": {"atoms": [3, 9], "description": "O3-H9 distance"},
        },
        "h5_datasets": [
            "atomic_numbers",
            "symbols",
            "coordinates_bohr",
            "energy_hartree",
            "gradient_hartree_per_bohr",
            "forces_hartree_per_bohr",
            "hessian_hartree_per_bohr2",
            "q_nh_angstrom",
            "q_oh_angstrom",
            "grid_id",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    upload_metadata = json.loads((scan_dir / "metadata.json").read_text())
    (out_dir / "scan_metadata.json").write_text(
        json.dumps(upload_metadata, indent=2, sort_keys=True) + "\n"
    )

    copy_xyz_files(scan_dir, out_dir)
    write_raw_archive(scan_dir, out_dir, orca_output_dir, orca_engrad_dir)
    write_readme(out_dir, len(rows))

    print(f"Wrote {out_dir}")
    print(f"HDF5: {h5_path}")
    print(f"Rows: {len(rows)}")
    print(f"Energy range: {energies_arr.min():.12f}..{energies_arr.max():.12f} Eh")


if __name__ == "__main__":
    main()
