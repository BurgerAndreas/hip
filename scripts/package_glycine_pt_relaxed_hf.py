#!/usr/bin/env python
"""Package the relaxed glycine proton-transfer ORCA scan for Hugging Face upload.

Mirrors the layout of ``package_glycine_pt_hf.py`` / ``package_glycine_pt_mep_hf.py``,
but for the GFN2-xTB constrained-relaxed 2D scan in ``(s, sigma)`` coordinates
(``scripts/glycine_pt_scan_relaxed.py``).

Unlike the rigid ``orca_wb97x_631gd_glycine_pt_nh_oh_scan_1285`` dataset:

* geometries are xTB-relaxed with fixed ``q_nh`` / ``q_oh`` bond lengths,
* the grid is regular in ``s = q_nh - q_oh`` and ``sigma = q_nh + q_oh``,
* DFT uses wB97X-D3 with separate ORCA Freq (``.hess``) and EnGrad (``.engrad``) passes.

Parses energies/gradients/Hessians directly from ORCA outputs (no ``orca_energies.csv``).
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
DATASET_NAME = "orca_wb97x_d3_631gd_glycine_pt_relaxed_579"
H5_NAME = "glycine_pt_relaxed.h5"
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


def copy_xyz_files(scan_dir: Path, out_dir: Path) -> None:
    xyz_dir = out_dir / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((scan_dir / "xyz").glob("grid_*.xyz")):
        shutil.copy2(path, xyz_dir / path.name)
    for pattern in (
        "reference_*_relaxed.xyz",
        "ts_xtb_saddle.xyz",
    ):
        for path in sorted((scan_dir / "xyz").glob(pattern)):
            shutil.copy2(path, xyz_dir / path.name)


def write_raw_archive(
    out_dir: Path,
    orca_output_dir: Path,
    orca_engrad_dir: Path,
) -> None:
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "orca_outputs.tar.gz"
    seen: set[str] = set()
    with tarfile.open(archive, "w:gz") as tar:
        for directory in (orca_output_dir, orca_engrad_dir):
            for path in sorted(directory.glob("*")):
                if not path.is_file():
                    continue
                if not any(path.name.endswith(ext) for ext in RAW_EXTENSIONS):
                    continue
                if path.name in seen:
                    continue
                seen.add(path.name)
                tar.add(path, arcname=f"orca_outputs/{path.name}")


def copy_diagnostics(scan_dir: Path, out_dir: Path, energies: pd.DataFrame) -> None:
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "orca_vib_summary.csv",
        "orca_vib_cache.npz",
        "xtb_relaxed_arrays.npz",
        "stationary_points.json",
        "ts_xtb_saddle.json",
    ):
        src = scan_dir / name
        if src.exists():
            shutil.copy2(src, diagnostics_dir / name)
    energies.to_csv(diagnostics_dir / "orca_energies.csv", index=False)
    energies.to_parquet(diagnostics_dir / "orca_energies.parquet", index=False)


def write_readme(out_dir: Path, n_samples: int, route_freq: str, route_engrad: str) -> None:
    (out_dir / "README.md").write_text(
        f"""# ORCA wB97X-D3/6-31G(d) Glycine Proton-Transfer Relaxed Scan

This folder contains {n_samples} ORCA analytical Hessian calculations for a 2D scan of
glycine intramolecular proton transfer on **GFN2-xTB relaxed** geometries.

- Source reaction: Transition1x `test` split, `sample_id=5`, `rxn1961`
- Formula: `C2H5NO2`
- Method: wB97X-D3
- Basis: 6-31G(d)
- ORCA version: 6.1.1
- ORCA Freq route: `{route_freq}`
- ORCA EnGrad route: `{route_engrad}`
- Grid coordinates: `s = q_nh - q_oh`, `sigma = q_nh + q_oh` (Å)
- Bond constraints during xTB relaxation: `q_nh = d(N4,H9)`, `q_oh = d(O3,H9)`
- Geometry construction: GFN2-xTB constrained relaxation at each grid node; all other
  degrees of freedom relax while the two O-H / N-H bond lengths are held fixed.
- Geometry files: `xyz/*.xyz` in Angstrom
- Targets: `h5/{H5_NAME}`
- Diagnostics: `diagnostics/orca_vib_summary.csv`, optional `orca_vib_cache.npz`
- Raw ORCA files: `raw/orca_outputs.tar.gz` containing `.out`, `.hess`, `.engrad`,
  and `.inp` files from the Freq and EnGrad passes.

The HDF5 file uses atomic units for model targets:
`coordinates_bohr`, `energy_hartree`, `gradient_hartree_per_bohr`,
`forces_hartree_per_bohr`, and `hessian_hartree_per_bohr2`.
Scan descriptors are stored as `s`, `sigma`, `q_nh_angstrom`, `q_oh_angstrom`
(relaxed bond lengths), plus target grid indices `i_s`, `i_sigma`.
See `metadata.csv` for the per-geometry mapping.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=Path("runs/glycine_pt_scan_relaxed"))
    parser.add_argument(
        "--orca-output-dir",
        type=Path,
        default=None,
        help="Directory containing ORCA Freq outputs (.hess). Defaults to scan-dir/orca_outputs.",
    )
    parser.add_argument(
        "--orca-engrad-dir",
        type=Path,
        default=None,
        help="Directory containing ORCA .engrad files. Defaults to scan-dir/orca_engrad_outputs.",
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    scan_dir = args.scan_dir
    orca_output_dir = args.orca_output_dir or scan_dir / "orca_outputs"
    orca_engrad_dir = args.orca_engrad_dir or scan_dir / "orca_engrad_outputs"
    out_dir = args.output_dir or Path("runs/hf_upload") / args.dataset_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "h5").mkdir(parents=True)

    scan_metadata = json.loads((scan_dir / "metadata.json").read_text())
    route_freq = scan_metadata.get("orca_freq_route", "! wB97X-D3 6-31G(d) TightSCF Freq")
    route_engrad = scan_metadata.get("orca_engrad_route", "! wB97X-D3 6-31G(d) TightSCF EnGrad")

    rows = pd.read_csv(scan_dir / "scan_manifest.csv").sort_values("grid_id").reset_index(drop=True)

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
        engrad_path = orca_engrad_dir / f"{stem}.engrad"
        hess_path = orca_output_dir / f"{stem}.hess"
        out_path = orca_output_dir / f"{stem}.out"
        if not engrad_path.exists():
            raise FileNotFoundError(f"Missing EnGrad file: {engrad_path}")
        if not hess_path.exists():
            raise FileNotFoundError(f"Missing Hessian file: {hess_path}")

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
    i_s_arr = rows["i_s"].to_numpy(dtype=np.int64)
    i_sigma_arr = rows["i_sigma"].to_numpy(dtype=np.int64)
    s_arr = rows["s"].to_numpy(dtype=np.float64)
    sigma_arr = rows["sigma"].to_numpy(dtype=np.float64)
    q_nh_target_arr = rows["q_nh_target"].to_numpy(dtype=np.float64)
    q_oh_target_arr = rows["q_oh_target"].to_numpy(dtype=np.float64)
    q_nh_arr = rows["q_nh_relaxed"].to_numpy(dtype=np.float64)
    q_oh_arr = rows["q_oh_relaxed"].to_numpy(dtype=np.float64)
    xtb_energy_ev_arr = rows["xtb_energy_ev"].to_numpy(dtype=np.float64)
    xtb_converged_arr = rows["converged"].astype(bool).to_numpy()

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
        h5.create_dataset("grid_id", data=grid_id_arr)
        h5.create_dataset("i_s", data=i_s_arr)
        h5.create_dataset("i_sigma", data=i_sigma_arr)
        h5.create_dataset("s", data=s_arr)
        h5.create_dataset("sigma", data=sigma_arr)
        h5.create_dataset("q_nh_target_angstrom", data=q_nh_target_arr)
        h5.create_dataset("q_oh_target_angstrom", data=q_oh_target_arr)
        h5.create_dataset("q_nh_angstrom", data=q_nh_arr)
        h5.create_dataset("q_oh_angstrom", data=q_oh_arr)
        h5.create_dataset("xtb_energy_ev", data=xtb_energy_ev_arr)
        h5.create_dataset("xtb_converged", data=xtb_converged_arr)
        h5.attrs["dataset"] = args.dataset_name
        h5.attrs["method"] = "wB97X-D3"
        h5.attrs["basis"] = "6-31G(d)"
        h5.attrs["charge"] = 0
        h5.attrs["multiplicity"] = 1
        h5.attrs["geometry_source"] = "GFN2-xTB constrained relaxation"
        h5.attrs["orca_freq_route"] = route_freq
        h5.attrs["orca_engrad_route"] = route_engrad

    diagnostics_energies = pd.DataFrame(
        {
            "grid_id": grid_id_arr,
            "i_s": i_s_arr,
            "i_sigma": i_sigma_arr,
            "s": s_arr,
            "sigma": sigma_arr,
            "q_nh_target": q_nh_target_arr,
            "q_oh_target": q_oh_target_arr,
            "q_nh": q_nh_arr,
            "q_oh": q_oh_arr,
            "xtb_energy_ev": xtb_energy_ev_arr,
            "xtb_converged": xtb_converged_arr,
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
        metadata_rows.append(
            {
                "name": f"glycine_pt_relaxed/grid_{int(grid_id_arr[idx]):04d}",
                "job_id": f"grid_{int(grid_id_arr[idx]):04d}",
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
                "geometry_source": "GFN2-xTB constrained relaxation",
                "grid_id": int(grid_id_arr[idx]),
                "i_s": int(i_s_arr[idx]),
                "i_sigma": int(i_sigma_arr[idx]),
                "s": float(s_arr[idx]),
                "sigma": float(sigma_arr[idx]),
                "q_nh_target_angstrom": float(q_nh_target_arr[idx]),
                "q_oh_target_angstrom": float(q_oh_target_arr[idx]),
                "q_nh_angstrom": float(q_nh_arr[idx]),
                "q_oh_angstrom": float(q_oh_arr[idx]),
                "xtb_energy_ev": float(xtb_energy_ev_arr[idx]),
                "xtb_converged": bool(xtb_converged_arr[idx]),
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
        "orca_freq_route": route_freq,
        "orca_engrad_route": route_engrad,
        "geometry_source": "GFN2-xTB constrained relaxation",
        "scan_coordinates": {
            "s": {"description": "q_nh - q_oh reaction coordinate (Å)"},
            "sigma": {"description": "q_nh + q_oh donor-acceptor compression (Å)"},
            "q_nh_angstrom": {"atoms": [4, 9], "description": "Relaxed N4-H9 distance"},
            "q_oh_angstrom": {"atoms": [3, 9], "description": "Relaxed O3-H9 distance"},
        },
        "h5_datasets": [
            "atomic_numbers",
            "symbols",
            "coordinates_bohr",
            "energy_hartree",
            "gradient_hartree_per_bohr",
            "forces_hartree_per_bohr",
            "hessian_hartree_per_bohr2",
            "grid_id",
            "i_s",
            "i_sigma",
            "s",
            "sigma",
            "q_nh_target_angstrom",
            "q_oh_target_angstrom",
            "q_nh_angstrom",
            "q_oh_angstrom",
            "xtb_energy_ev",
            "xtb_converged",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "scan_metadata.json").write_text(json.dumps(scan_metadata, indent=2, sort_keys=True) + "\n")

    copy_xyz_files(scan_dir, out_dir)
    copy_diagnostics(scan_dir, out_dir, diagnostics_energies)
    write_raw_archive(out_dir, orca_output_dir, orca_engrad_dir)
    write_readme(out_dir, len(rows), route_freq, route_engrad)

    print(f"Wrote {out_dir}")
    print(f"HDF5: {h5_path}")
    print(f"Rows: {len(rows)}")
    print(f"Terminated normally: {int(np.sum(terminated_flags))}/{len(rows)}")
    print(f"SCF converged: {int(np.sum(scf_flags))}/{len(rows)}")
    print(f"Energy range: {energies_arr.min():.12f}..{energies_arr.max():.12f} Eh")


if __name__ == "__main__":
    main()
