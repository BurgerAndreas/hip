#!/usr/bin/env python3
"""Prepare ORCA analytical Hessian inputs for selected large-molecule geometries."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class Job:
    name: str
    job_id: str
    atoms: int
    source_path: str
    source_format: str
    workdir: str
    input_path: str
    output_path: str
    hessian_path: str
    engrad_input_path: str
    engrad_output_path: str
    engrad_path: str
    hdf5_path: str
    charge: int
    multiplicity: int


def read_molecule_table(path: Path, min_atoms: int, max_atoms: int) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [
        row
        for row in rows
        if row["Format"].lower() in {"sdf", "xyz"} and min_atoms <= int(row["Atoms"]) <= max_atoms
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


def write_orca_input(
    path: Path,
    atoms: list[tuple[str, float, float, float]],
    charge: int,
    multiplicity: int,
    nprocs: int,
    maxcore_mb: int,
    task: str,
) -> None:
    lines = [
        f"! wB97X 6-31G(d) TightSCF {task}",
        "",
        f"%pal nprocs {nprocs} end",
        f"%maxcore {maxcore_mb}",
        "",
        f"* xyz {charge} {multiplicity}",
    ]
    lines.extend(f"{symbol:<2} {x:16.10f} {y:16.10f} {z:16.10f}" for symbol, x, y, z in atoms)
    lines.append("*")
    lines.append("")
    path.write_text("\n".join(lines))


def prepare_jobs(args: argparse.Namespace) -> list[Job]:
    repo_root = Path(args.repo_root).resolve()
    scratch_root = Path(args.scratch_root).expanduser().resolve()
    table_path = repo_root / args.molecule_table
    rows = read_molecule_table(table_path, args.min_atoms, args.max_atoms)

    jobs: list[Job] = []
    for row in rows:
        name = row["Name"]
        job_id = name.replace("/", "__")
        source_format = row["Format"].lower()
        source_path = repo_root / "data" / "large" / f"{name}.{source_format}"
        atoms, charge = read_geometry(source_path, source_format)
        expected_atoms = int(row["Atoms"])
        if len(atoms) != expected_atoms:
            raise ValueError(f"{source_path} has {len(atoms)} atoms, expected {expected_atoms}")

        workdir = scratch_root / "jobs" / job_id
        results_dir = scratch_root / "results"
        workdir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        input_path = workdir / f"{job_id}.inp"
        output_path = workdir / f"{job_id}.out"
        hessian_path = workdir / f"{job_id}.hess"
        engrad_input_path = workdir / f"{job_id}_engrad.inp"
        engrad_output_path = workdir / f"{job_id}_engrad.out"
        engrad_path = workdir / f"{job_id}_engrad.engrad"
        hdf5_path = results_dir / f"{job_id}.h5"
        write_orca_input(
            input_path,
            atoms,
            charge=charge,
            multiplicity=args.multiplicity,
            nprocs=args.nprocs,
            maxcore_mb=args.maxcore_mb,
            task="Freq",
        )
        write_orca_input(
            engrad_input_path,
            atoms,
            charge=charge,
            multiplicity=args.multiplicity,
            nprocs=args.nprocs,
            maxcore_mb=args.maxcore_mb,
            task="EnGrad",
        )

        jobs.append(
            Job(
                name=name,
                job_id=job_id,
                atoms=expected_atoms,
                source_path=str(source_path),
                source_format=source_format,
                workdir=str(workdir),
                input_path=str(input_path),
                output_path=str(output_path),
                hessian_path=str(hessian_path),
                engrad_input_path=str(engrad_input_path),
                engrad_output_path=str(engrad_output_path),
                engrad_path=str(engrad_path),
                hdf5_path=str(hdf5_path),
                charge=charge,
                multiplicity=args.multiplicity,
            )
        )

    scratch_root.mkdir(parents=True, exist_ok=True)
    manifest_path = scratch_root / "manifest.json"
    manifest_tmp_path = manifest_path.with_suffix(".json.tmp")
    manifest_tmp_path.write_text(json.dumps([asdict(job) for job in jobs], indent=2) + "\n")
    manifest_tmp_path.replace(manifest_path)

    jobs_tsv_path = scratch_root / "jobs.tsv"
    jobs_tsv_tmp_path = jobs_tsv_path.with_suffix(".tsv.tmp")
    with jobs_tsv_tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(jobs[0]).keys()), delimiter="\t")
        writer.writeheader()
        for job in jobs:
            writer.writerow(asdict(job))
    jobs_tsv_tmp_path.replace(jobs_tsv_path)

    if not args.quiet:
        print(f"Prepared {len(jobs)} ORCA inputs in {scratch_root}")
        for index, job in enumerate(jobs):
            print(f"{index}: {job.name} ({job.atoms} atoms) -> {job.input_path}")

    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scratch-root", default="~/scratch/hip/orca_hessians")
    parser.add_argument("--molecule-table", default="data/large/molecules.txt")
    parser.add_argument("--min-atoms", type=int, default=30)
    parser.add_argument("--max-atoms", type=int, default=100)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--nprocs", type=int, default=8)
    parser.add_argument("--maxcore-mb", type=int, default=4000)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare_jobs(parse_args())
