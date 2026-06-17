#!/usr/bin/env python3
"""Calculate large-molecule wB97X/6-31G* Hessians with PySCF or gpu4pyscf.

The output HDF5 schema intentionally matches the ORCA Hessian reference files
used by the HIP evaluation scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


FORMAL_CHARGE_CODES = {
    0: 0,
    1: 3,
    2: 2,
    3: 1,
    4: 0,
    5: -1,
    6: -2,
    7: -3,
}

ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Br": 35,
    "I": 53,
}


@dataclass(frozen=True)
class MoleculeJob:
    name: str
    job_id: str
    atoms: int
    source_path: str
    source_format: str
    charge: int
    multiplicity: int
    output_path: str


def read_molecule_table(path: Path, min_atoms: int, max_atoms: int) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        row
        for row in rows
        if row["Format"].lower() in {"sdf", "xyz"}
        and min_atoms <= int(row["Atoms"]) <= max_atoms
    ]


def read_sdf_xyz(path: Path) -> tuple[list[tuple[str, float, float, float]], int]:
    lines = path.read_text().splitlines()
    if len(lines) < 4:
        raise ValueError(f"{path} is too short to be an SDF file")

    counts = lines[3].split()
    natoms = int(counts[0])
    nbonds = int(counts[1])

    atoms: list[tuple[str, float, float, float]] = []
    charge = 0
    for idx, line in enumerate(lines[4 : 4 + natoms], start=1):
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(f"Malformed atom line {idx} in {path}: {line}")

        x, y, z = map(float, fields[:3])
        symbol = fields[3]
        charge_code = int(fields[5])
        charge += FORMAL_CHARGE_CODES.get(charge_code, 0)
        atoms.append((symbol, x, y, z))

    property_start = 4 + natoms + nbonds
    for line in lines[property_start:]:
        if line.startswith("M  END"):
            break
        if line.startswith("M  CHG"):
            fields = line.split()
            count = int(fields[2])
            charge = sum(int(fields[4 + 2 * i]) for i in range(count))

    return atoms, charge


def read_xyz(path: Path) -> list[tuple[str, float, float, float]]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} is too short to be an XYZ file")

    natoms = int(lines[0].strip())
    atoms: list[tuple[str, float, float, float]] = []
    for idx, line in enumerate(lines[2 : 2 + natoms], start=1):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed XYZ atom line {idx} in {path}: {line}")

        symbol = fields[0]
        x, y, z = map(float, fields[1:4])
        atoms.append((symbol, x, y, z))

    return atoms


def read_geometry(path: Path, source_format: str) -> tuple[list[tuple[str, float, float, float]], int]:
    if source_format == "sdf":
        return read_sdf_xyz(path)
    if source_format == "xyz":
        return read_xyz(path), 0
    raise ValueError(f"Unsupported geometry format: {source_format}")


def import_backend(backend: str) -> tuple[Any, Any, str]:
    from pyscf import gto

    if backend == "cpu":
        from pyscf import dft

        return gto, dft, "pyscf"
    if backend == "gpu":
        try:
            from gpu4pyscf import dft
        except ImportError as error:
            raise ImportError(
                "Backend 'gpu' requires gpu4pyscf. Install it in this environment "
                "or rerun with --backend cpu."
            ) from error
        return gto, dft, "gpu4pyscf"
    raise ValueError(f"Unsupported backend: {backend}")


def rks_factory(dft_module: Any, mol: Any) -> Any:
    if hasattr(dft_module, "RKS"):
        return dft_module.RKS(mol)
    if hasattr(dft_module, "rks") and hasattr(dft_module.rks, "RKS"):
        return dft_module.rks.RKS(mol)
    raise AttributeError("Could not find an RKS class in the selected DFT module")


def to_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array, dtype=np.float64)


def dense_hessian(hessian: Any, natoms: int) -> np.ndarray:
    hessian_np = to_numpy(hessian)
    if hessian_np.shape == (3 * natoms, 3 * natoms):
        return hessian_np
    if hessian_np.shape == (natoms, natoms, 3, 3):
        return hessian_np.transpose(0, 2, 1, 3).reshape(3 * natoms, 3 * natoms)
    raise ValueError(
        f"Unexpected Hessian shape {hessian_np.shape}; expected "
        f"{(natoms, natoms, 3, 3)} or {(3 * natoms, 3 * natoms)}"
    )


def make_jobs(args: argparse.Namespace) -> list[MoleculeJob]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    molecule_table = Path(args.molecule_table).expanduser()
    if not molecule_table.is_absolute():
        molecule_table = repo_root / molecule_table
    source_dir = Path(args.source_dir).expanduser()
    if not source_dir.is_absolute():
        source_dir = repo_root / source_dir
    output_dir = Path(args.output_dir).expanduser().resolve()

    rows = read_molecule_table(molecule_table, args.min_atoms, args.max_atoms)
    jobs: list[MoleculeJob] = []
    for row in rows:
        name = row["Name"]
        source_format = row["Format"].lower()
        source_path = source_dir / f"{name}.{source_format}"
        atoms, charge = read_geometry(source_path, source_format)
        expected_atoms = int(row["Atoms"])
        if len(atoms) != expected_atoms:
            raise ValueError(f"{source_path} has {len(atoms)} atoms, expected {expected_atoms}")

        job_id = name.replace("/", "__")
        jobs.append(
            MoleculeJob(
                name=name,
                job_id=job_id,
                atoms=expected_atoms,
                source_path=str(source_path),
                source_format=source_format,
                charge=charge,
                multiplicity=args.multiplicity,
                output_path=str(output_dir / f"{job_id}.h5"),
            )
        )

    if args.sample_name is not None:
        jobs = [job for job in jobs if job.name == args.sample_name]
        if not jobs:
            raise ValueError(f"No molecule named {args.sample_name!r} in {molecule_table}")
    if args.sample_index is not None:
        jobs = [jobs[args.sample_index]]
    if args.shard_index is not None or args.shard_size is not None:
        if args.shard_index is None or args.shard_size is None:
            raise ValueError("--shard-index and --shard-size must be provided together")
        start = args.shard_index * args.shard_size
        jobs = jobs[start : start + args.shard_size]
    return jobs


def build_mol(
    gto_module: Any,
    atoms: list[tuple[str, float, float, float]],
    charge: int,
    multiplicity: int,
    basis: str,
    verbose: int,
    max_memory_mb: int | None,
) -> Any:
    mol = gto_module.Mole()
    mol.atom = [(symbol, (x, y, z)) for symbol, x, y, z in atoms]
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.verbose = verbose
    if max_memory_mb is not None:
        mol.max_memory = max_memory_mb
    mol.build()
    return mol


def resolve_max_memory_mb(args: argparse.Namespace) -> int | None:
    if args.max_memory_mb is not None:
        return args.max_memory_mb

    slurm_mem_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem_mb and slurm_mem_mb.isdigit():
        return max(1000, int(int(slurm_mem_mb) * args.slurm_memory_fraction))

    mem_per_cpu_mb = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if mem_per_cpu_mb and cpus and mem_per_cpu_mb.isdigit() and cpus.isdigit():
        return max(1000, int(int(mem_per_cpu_mb) * int(cpus) * args.slurm_memory_fraction))

    return None


def run_pyscf(job: MoleculeJob, args: argparse.Namespace) -> dict[str, Any]:
    gto_module, dft_module, backend_name = import_backend(args.backend)
    if args.num_threads is not None:
        from pyscf import lib

        lib.num_threads(args.num_threads)

    atoms, charge = read_geometry(Path(job.source_path), job.source_format)
    max_memory_mb = resolve_max_memory_mb(args)
    mol = build_mol(
        gto_module,
        atoms,
        charge=charge,
        multiplicity=job.multiplicity,
        basis=args.basis,
        verbose=args.pyscf_verbose,
        max_memory_mb=max_memory_mb,
    )

    mf = rks_factory(dft_module, mol)
    if max_memory_mb is not None:
        mf.max_memory = max_memory_mb
    mf.xc = args.xc
    mf.conv_tol = args.conv_tol
    mf.max_cycle = args.max_cycle
    if hasattr(mf, "grids"):
        mf.grids.level = args.grid_level
    if hasattr(mf, "nlcgrids"):
        mf.nlcgrids.level = args.nlc_grid_level
    if args.density_fit:
        if args.auxbasis is None:
            mf = mf.density_fit()
        else:
            mf = mf.density_fit(auxbasis=args.auxbasis)

    start = time.perf_counter()
    energy = float(mf.kernel())
    if not getattr(mf, "converged", False) and not args.allow_unconverged:
        raise RuntimeError(f"SCF did not converge for {job.name}")

    gradient = to_numpy(mf.nuc_grad_method().kernel()).reshape(job.atoms, 3)
    hessian_raw = mf.Hessian().kernel()
    hessian = dense_hessian(hessian_raw, job.atoms)
    elapsed_s = time.perf_counter() - start

    coordinates_bohr = to_numpy(mol.atom_coords(unit="Bohr"))
    symbols = [atom[0] for atom in atoms]
    atomic_numbers = np.asarray([ATOMIC_NUMBERS[symbol] for symbol in symbols], dtype=np.int16)
    forces = -gradient

    output_path = Path(job.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.attrs["name"] = job.name
        handle.attrs["method"] = args.xc
        handle.attrs["basis"] = args.basis
        handle.attrs["program"] = backend_name
        handle.attrs["backend"] = args.backend
        handle.attrs["density_fit"] = bool(args.density_fit)
        handle.attrs["auxbasis"] = "" if args.auxbasis is None else args.auxbasis
        handle.attrs["charge"] = charge
        handle.attrs["multiplicity"] = job.multiplicity
        handle.attrs["max_memory_mb"] = -1 if max_memory_mb is None else max_memory_mb
        handle.attrs["source_path"] = job.source_path
        handle.attrs["energy_units"] = "hartree"
        handle.attrs["gradient_units"] = "hartree/bohr"
        handle.attrs["forces_units"] = "hartree/bohr"
        handle.attrs["hessian_units"] = "hartree/bohr^2"
        handle.create_dataset("atomic_numbers", data=atomic_numbers)
        handle.create_dataset("symbols", data=np.asarray(symbols, dtype=h5py.string_dtype()))
        handle.create_dataset("coordinates_bohr", data=coordinates_bohr)
        handle.create_dataset("energy_hartree", data=np.asarray(energy))
        handle.create_dataset("gradient_hartree_per_bohr", data=gradient.reshape(-1))
        handle.create_dataset("forces_hartree_per_bohr", data=forces.reshape(-1))
        handle.create_dataset("hessian_hartree_per_bohr2", data=hessian)

    return {
        **asdict(job),
        "status": "ok",
        "backend": args.backend,
        "density_fit": bool(args.density_fit),
        "xc": args.xc,
        "basis": args.basis,
        "max_memory_mb": max_memory_mb,
        "energy_hartree": energy,
        "time_s": elapsed_s,
    }


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")

    if not rows:
        return
    metrics_path = output_dir / "metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--molecule-table", default="data/large/molecules.txt")
    parser.add_argument("--source-dir", default="data/large")
    parser.add_argument("--output-dir", default="~/scratch/hip/pyscf_hessians/results")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--density-fit", action="store_true")
    parser.add_argument("--auxbasis", default=None)
    parser.add_argument("--xc", default="wb97x")
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--min-atoms", type=int, default=30)
    parser.add_argument("--max-atoms", type=int, default=100)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--sample-name", default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--grid-level", type=int, default=5)
    parser.add_argument("--nlc-grid-level", type=int, default=5)
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--max-memory-mb", type=int, default=None)
    parser.add_argument("--slurm-memory-fraction", type=float, default=0.9)
    parser.add_argument("--pyscf-verbose", type=int, default=3)
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    jobs = make_jobs(args)
    print(f"Selected {len(jobs)} molecule(s)")

    rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        output_path = Path(job.output_path)
        if output_path.exists() and not args.redo:
            print(f"[{index + 1}/{len(jobs)}] Skipping {job.name}: {output_path} exists")
            rows.append({**asdict(job), "status": "skipped"})
            continue

        print(
            f"[{index + 1}/{len(jobs)}] Running {args.backend} PySCF "
            f"df={args.density_fit} for {job.name} ({job.atoms} atoms)"
        )
        try:
            row = run_pyscf(job, args)
            print(f"Wrote {job.output_path} in {row['time_s']:.1f} s")
        except Exception as error:
            row = {
                **asdict(job),
                "status": "error",
                "backend": args.backend,
                "density_fit": bool(args.density_fit),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            print(row["error"])
            if not args.keep_going:
                rows.append(row)
                write_summary(output_dir, rows)
                raise
        rows.append(row)
        write_summary(output_dir, rows)


if __name__ == "__main__":
    main()
