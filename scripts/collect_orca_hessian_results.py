#!/usr/bin/env python3
"""Collect ORCA energies, gradients/forces, and Hessians into per-sample HDF5 files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import h5py
import numpy as np


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


def section(lines: list[str], name: str) -> list[str]:
    start = None
    marker = f"${name}"
    for index, line in enumerate(lines):
        if line.strip() == marker:
            start = index + 1
            break
    if start is None:
        raise ValueError(f"Missing ${name} section")

    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].strip().startswith("$"):
            end = index
            break
    return [line.rstrip() for line in lines[start:end] if line.strip()]


def parse_hessian(lines: list[str]) -> np.ndarray:
    block = section(lines, "hessian")
    dim = int(block[0].split()[0])
    hessian = np.zeros((dim, dim), dtype=np.float64)
    index = 1
    while index < len(block):
        columns = [int(item) for item in block[index].split()]
        index += 1
        for _ in range(dim):
            fields = block[index].split()
            row = int(fields[0])
            values = [float(item) for item in fields[1:]]
            for column, value in zip(columns, values, strict=True):
                hessian[row, column] = value
            index += 1
    return hessian


def parse_vector(lines: list[str], name: str) -> np.ndarray:
    block = section(lines, name)
    dim = int(block[0].split()[0])
    values: list[float] = []
    for line in block[1:]:
        fields = line.split()
        if len(fields) == 2 and fields[0].lstrip("+-").isdigit():
            values.append(float(fields[1]))
        else:
            values.extend(float(item) for item in fields)
        if len(values) >= dim:
            break
    vector = np.asarray(values[:dim], dtype=np.float64)
    if vector.shape != (dim,):
        raise ValueError(f"${name} has {vector.size} values, expected {dim}")
    return vector


def parse_scalar(lines: list[str], name: str) -> float:
    block = section(lines, name)
    return float(block[0].split()[0])


def parse_atoms(lines: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    block = section(lines, "atoms")
    natoms = int(block[0].split()[0])
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in block[1 : 1 + natoms]:
        fields = line.split()
        symbols.append(fields[0])
        coords.append([float(fields[-3]), float(fields[-2]), float(fields[-1])])
    atomic_numbers = np.asarray([ATOMIC_NUMBERS[symbol] for symbol in symbols], dtype=np.int16)
    coordinates_bohr = np.asarray(coords, dtype=np.float64)
    return atomic_numbers, coordinates_bohr, symbols


def read_final_energy_from_output(path: Path) -> float | None:
    if not path.exists():
        return None
    pattern = re.compile(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)")
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None


def parse_engrad(path: Path) -> tuple[float, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ORCA gradient file: {path}")

    values: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            values.append(stripped)

    if len(values) < 2:
        raise ValueError(f"Malformed ORCA gradient file: {path}")

    natoms = int(values[0])
    energy = float(values[1])
    gradient_values = [float(value) for value in values[2 : 2 + 3 * natoms]]
    gradient = np.asarray(gradient_values, dtype=np.float64)
    if gradient.shape != (3 * natoms,):
        raise ValueError(f"{path} has {gradient.size} gradient values, expected {3 * natoms}")
    return energy, gradient


def rows_from_jobs_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def selected_rows(rows: list[dict[str, str]], sample_name: str | None, sample_index: int | None) -> list[dict[str, str]]:
    if sample_name is not None:
        return [row for row in rows if row["name"] == sample_name]
    if sample_index is not None:
        return [rows[sample_index]]
    return rows


def collect(row: dict[str, str]) -> Path:
    hessian_path = Path(row["hessian_path"])
    output_path = Path(row["output_path"])
    engrad_path = Path(row["engrad_path"])
    hdf5_path = Path(row["hdf5_path"])
    if not hessian_path.exists():
        raise FileNotFoundError(f"Missing ORCA Hessian file: {hessian_path}")

    lines = hessian_path.read_text(errors="replace").splitlines()
    hessian = parse_hessian(lines)
    engrad_energy, gradient = parse_engrad(engrad_path)
    forces = -gradient
    energy = read_final_energy_from_output(output_path)
    if energy is None:
        energy = engrad_energy
    atomic_numbers, coordinates_bohr, symbols = parse_atoms(lines)

    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as handle:
        handle.attrs["name"] = row["name"]
        handle.attrs["method"] = "wB97X"
        handle.attrs["basis"] = "6-31G(d)"
        handle.attrs["orca_hessian_path"] = str(hessian_path)
        handle.attrs["orca_output_path"] = str(output_path)
        handle.attrs["orca_engrad_path"] = str(engrad_path)
        handle.attrs["energy_units"] = "hartree"
        handle.attrs["gradient_units"] = "hartree/bohr"
        handle.attrs["forces_units"] = "hartree/bohr"
        handle.attrs["hessian_units"] = "hartree/bohr^2"
        handle.create_dataset("atomic_numbers", data=atomic_numbers)
        handle.create_dataset("symbols", data=np.asarray(symbols, dtype=h5py.string_dtype()))
        handle.create_dataset("coordinates_bohr", data=coordinates_bohr)
        handle.create_dataset("energy_hartree", data=np.asarray(energy))
        handle.create_dataset("gradient_hartree_per_bohr", data=gradient)
        handle.create_dataset("forces_hartree_per_bohr", data=forces)
        handle.create_dataset("hessian_hartree_per_bohr2", data=hessian)

    print(f"Wrote {hdf5_path}")
    return hdf5_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", default="~/scratch/hip/orca_hessians")
    parser.add_argument("--jobs-tsv", default=None)
    parser.add_argument("--sample-name", default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scratch_root = Path(args.scratch_root).expanduser().resolve()
    jobs_tsv = Path(args.jobs_tsv).expanduser().resolve() if args.jobs_tsv else scratch_root / "jobs.tsv"
    rows = rows_from_jobs_tsv(jobs_tsv)
    for row in selected_rows(rows, args.sample_name, args.sample_index):
        collect(row)


if __name__ == "__main__":
    main()
