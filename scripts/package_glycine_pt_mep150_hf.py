#!/usr/bin/env python
"""Package the 150-frame glycine proton-transfer MEP for Hugging Face upload.

Mirrors the layout produced by ``package_glycine_pt_mep_hf.py`` for the 73-frame
path, but sources values directly from the ORCA outputs because the n150
``orca_energies.csv`` manifest was never backfilled with parsed energies/flags.
"""
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

from package_glycine_pt_hf import parse_engrad, parse_hessian


HARTREE_TO_KCALMOL = 627.5094740631
DATASET_NAME = "orca_wb97x_631gd_glycine_pt_mep_150"
H5_NAME = "glycine_pt_mep_150.h5"
RAW_EXTENSIONS = (".out", ".hess", ".engrad", ".inp")


def atomic_symbols(atomic_numbers: np.ndarray) -> np.ndarray:
    symbols_by_z = {1: "H", 6: "C", 7: "N", 8: "O"}
    return np.array([symbols_by_z[int(z)] for z in atomic_numbers], dtype="S2")


def out_status(out_path: Path) -> tuple[bool, bool]:
    if not out_path.exists():
        return False, False
    text = out_path.read_text(errors="ignore")
    terminated = "ORCA TERMINATED NORMALLY" in text
    scf_converged = "SCF CONVERGED AFTER" in text
    return terminated, scf_converged


def copy_xyz_files(mep_dir: Path, out_dir: Path) -> None:
    xyz_dir = out_dir / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((mep_dir / "xyz").glob("path_*.xyz")):
        shutil.copy2(path, xyz_dir / path.name)


def write_raw_archive(out_dir: Path, orca_output_dir: Path) -> None:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "orca_outputs.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(orca_output_dir.glob("*")):
            if path.is_file() and any(path.name.endswith(ext) for ext in RAW_EXTENSIONS):
                tar.add(path, arcname=f"orca_outputs/{path.name}")


def copy_diagnostics(mep_dir: Path, out_dir: Path, energies: pd.DataFrame) -> None:
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    src = mep_dir / "orca_vib_summary.csv"
    if src.exists():
        shutil.copy2(src, diagnostics_dir / src.name)
    energies.to_csv(diagnostics_dir / "orca_energies.csv", index=False)
    energies.to_parquet(diagnostics_dir / "orca_energies.parquet", index=False)


def write_readme(out_dir: Path, n_samples: int, route: str) -> None:
    (out_dir / "README.md").write_text(
        f"""# ORCA wB97X-D3/6-31G(d) Glycine Proton-Transfer MEP (150 frames)

This folder contains {n_samples} ORCA analytical Hessian calculations along a
150-frame minimum-energy path for glycine intramolecular proton transfer. It is
a denser version of `orca_wb97x_631gd_glycine_pt_mep_73`, covering the same
reaction.

- Source reaction: Transition1x `test` split, `sample_id=5`, `rxn1961`
- Formula: `C2H5NO2`
- Method: wB97X-D3
- Basis: 6-31G(d)
- ORCA version: 6.1.1
- ORCA route: `{route}`
- Path coordinate: `xi`; scan descriptors are `q_nh = d(N4,H9)` and
  `q_oh = d(O3,H9)`. `x_along` is the native arc-length path coordinate.
- Geometry files: `xyz/*.xyz` in Angstrom
- Targets: `h5/{H5_NAME}`
- Diagnostics: `diagnostics/orca_vib_summary.csv`
- Raw ORCA files: `raw/orca_outputs.tar.gz` containing `.out`, `.hess`,
  `.engrad`, and `.inp` files

The HDF5 file uses atomic units for model targets:
`coordinates_bohr`, `energy_hartree`, `gradient_hartree_per_bohr`,
`forces_hartree_per_bohr`, and `hessian_hartree_per_bohr2`.
The MEP coordinates are stored as `frame_id`, `xi`, `q_nh_angstrom`,
`q_oh_angstrom`, and `x_along`. See `metadata.csv` for the per-frame mapping.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mep-dir", type=Path, default=Path("runs/glycine_pt_path_n150"))
    parser.add_argument(
        "--orca-output-dir",
        type=Path,
        default=None,
        help="Directory containing ORCA outputs. Defaults to mep-dir/orca_outputs.",
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    mep_dir = args.mep_dir
    orca_output_dir = args.orca_output_dir or mep_dir / "orca_outputs"
    out_dir = args.output_dir or Path("runs/hf_upload") / args.dataset_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "h5").mkdir(parents=True)

    scan_metadata = json.loads((mep_dir / "path_metadata.json").read_text())
    route = scan_metadata.get("orca_route", "! wB97X-D3 6-31G(d) TightSCF Grid5 FinalGrid6 Freq")

    rows = pd.read_csv(mep_dir / "orca_energies.csv").sort_values("grid_id").reset_index(drop=True)

    atomic_numbers_list = []
    coordinates_bohr = []
    energies_hartree = []
    gradients = []
    forces = []
    hessians = []
    xyz_filenames = []
    output_filenames = []
    terminated_flags = []
    scf_flags = []

    for row in rows.to_dict(orient="records"):
        stem = Path(row["orca_input_path"]).stem
        engrad_path = orca_output_dir / f"{stem}.engrad"
        hess_path = orca_output_dir / f"{stem}.hess"
        out_path = orca_output_dir / f"{stem}.out"
        n_atoms, energy, gradient, coords_bohr, atomic_numbers = parse_engrad(engrad_path)
        hessian = parse_hessian(hess_path)
        if hessian.shape != (3 * n_atoms, 3 * n_atoms):
            raise ValueError(f"Unexpected Hessian shape for {stem}: {hessian.shape}")
        terminated, scf_converged = out_status(out_path)

        atomic_numbers_list.append(atomic_numbers)
        coordinates_bohr.append(coords_bohr)
        energies_hartree.append(energy)
        gradients.append(gradient)
        forces.append(-gradient)
        hessians.append(hessian)
        xyz_filenames.append(f"xyz/{Path(row['xyz_path']).name}")
        output_filenames.append(f"raw/orca_outputs.tar.gz:orca_outputs/{stem}.out")
        terminated_flags.append(terminated)
        scf_flags.append(scf_converged)

    atomic_numbers_arr = np.stack(atomic_numbers_list)
    coordinates_bohr_arr = np.stack(coordinates_bohr)
    energies_arr = np.array(energies_hartree, dtype=np.float64)
    gradients_arr = np.stack(gradients)
    forces_arr = np.stack(forces)
    hessians_arr = np.stack(hessians)
    symbols = atomic_symbols(atomic_numbers_arr[0])

    grid_id_arr = rows["grid_id"].to_numpy(dtype=np.int64)
    frame_id_arr = grid_id_arr.copy()
    q_nh_arr = rows["q_nh"].to_numpy(dtype=np.float64)
    q_oh_arr = rows["q_oh"].to_numpy(dtype=np.float64)
    xi_arr = q_nh_arr - q_oh_arr
    x_along_arr = rows["x_along"].to_numpy(dtype=np.float64)

    energy_relative_hartree = energies_arr - np.min(energies_arr)
    energy_relative_kcalmol = energy_relative_hartree * HARTREE_TO_KCALMOL

    h5_path = out_dir / "h5" / H5_NAME
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("atomic_numbers", data=atomic_numbers_arr, compression="gzip")
        h5.create_dataset("symbols", data=symbols)
        h5.create_dataset("coordinates_bohr", data=coordinates_bohr_arr, compression="gzip")
        h5.create_dataset("energy_hartree", data=energies_arr)
        h5.create_dataset("gradient_hartree_per_bohr", data=gradients_arr, compression="gzip")
        h5.create_dataset("forces_hartree_per_bohr", data=forces_arr, compression="gzip")
        h5.create_dataset("hessian_hartree_per_bohr2", data=hessians_arr, compression="gzip")
        h5.create_dataset("frame_id", data=frame_id_arr)
        h5.create_dataset("grid_id", data=grid_id_arr)
        h5.create_dataset("xi", data=xi_arr)
        h5.create_dataset("q_nh_angstrom", data=q_nh_arr)
        h5.create_dataset("q_oh_angstrom", data=q_oh_arr)
        h5.create_dataset("x_along", data=x_along_arr)
        h5.attrs["dataset"] = args.dataset_name
        h5.attrs["method"] = "wB97X-D3"
        h5.attrs["basis"] = "6-31G(d)"
        h5.attrs["charge"] = 0
        h5.attrs["multiplicity"] = 1

    diagnostics_energies = pd.DataFrame(
        {
            "grid_id": grid_id_arr,
            "frame_id": frame_id_arr,
            "split": "test",
            "sample_id": 5,
            "formula": "C2H5NO2",
            "rxn": "rxn1961",
            "q_nh": q_nh_arr,
            "q_oh": q_oh_arr,
            "xi": xi_arr,
            "x_along": x_along_arr,
            "xyz_path": [Path(p).name for p in rows["xyz_path"]],
            "orca_terminated_normally": terminated_flags,
            "orca_scf_converged": scf_flags,
            "orca_energy_hartree": energies_arr,
            "orca_energy_relative_hartree": energy_relative_hartree,
            "orca_energy_relative_kcalmol": energy_relative_kcalmol,
        }
    )

    metadata_rows = []
    for idx in range(len(rows)):
        frame_id = int(frame_id_arr[idx])
        metadata_rows.append(
            {
                "name": f"glycine_pt_mep/frame_{frame_id:04d}",
                "job_id": f"mep_{frame_id:04d}",
                "atoms": int(atomic_numbers_arr.shape[1]),
                "source_path": rows["xyz_path"].iloc[idx],
                "source_format": "xyz",
                "charge": 0,
                "multiplicity": 1,
                "method": "wB97X-D3",
                "basis": "6-31G(d)",
                "energy_units": "hartree",
                "forces_units": "hartree/bohr",
                "hessian_units": "hartree/bohr^2",
                "frame_id": frame_id,
                "grid_id": int(grid_id_arr[idx]),
                "xi": float(xi_arr[idx]),
                "q_nh_angstrom": float(q_nh_arr[idx]),
                "q_oh_angstrom": float(q_oh_arr[idx]),
                "x_along": float(x_along_arr[idx]),
                "energy_hartree": float(energies_arr[idx]),
                "energy_relative_kcalmol": float(energy_relative_kcalmol[idx]),
                "xyz_path": xyz_filenames[idx],
                "h5_path": f"h5/{H5_NAME}",
                "h5_index": idx,
                "orca_output": output_filenames[idx],
                "terminated_normally": bool(terminated_flags[idx]),
                "scf_converged": bool(scf_flags[idx]),
            }
        )

    with (out_dir / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    summary = {
        "dataset": args.dataset_name,
        "num_samples": int(len(rows)),
        "atom_counts": {"min": int(atomic_numbers_arr.shape[1]), "max": int(atomic_numbers_arr.shape[1])},
        "source": {
            "transition1x_split": "test",
            "transition1x_sample_id": 5,
            "rxn": "rxn1961",
            "formula": "C2H5NO2",
        },
        "method": "wB97X-D3",
        "basis": "6-31G(d)",
        "orca_version": "6.1.1",
        "orca_route": route,
        "path_coordinates": {
            "xi": {"description": "Reaction-path coordinate from the 150-frame MEP"},
            "x_along": {"description": "Native arc-length path coordinate"},
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
            "frame_id",
            "grid_id",
            "xi",
            "q_nh_angstrom",
            "q_oh_angstrom",
            "x_along",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "scan_metadata.json").write_text(json.dumps(scan_metadata, indent=2, sort_keys=True) + "\n")

    copy_xyz_files(mep_dir, out_dir)
    copy_diagnostics(mep_dir, out_dir, diagnostics_energies)
    write_raw_archive(out_dir, orca_output_dir)
    write_readme(out_dir, len(rows), route)

    print(f"Wrote {out_dir}")
    print(f"HDF5: {h5_path}")
    print(f"Rows: {len(rows)}")
    print(f"Terminated normally: {int(np.sum(terminated_flags))}/{len(rows)}")
    print(f"SCF converged: {int(np.sum(scf_flags))}/{len(rows)}")
    print(f"Energy range: {energies_arr.min():.12f}..{energies_arr.max():.12f} Eh")


if __name__ == "__main__":
    main()
