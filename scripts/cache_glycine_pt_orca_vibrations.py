#!/usr/bin/env python
"""Cache ORCA vibrational diagnostics for the glycine proton-transfer scan."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
HARTREE_PER_BOHR2_TO_EV_PER_ANG2 = HARTREE_TO_EV / (BOHR_TO_ANGSTROM**2)
O_ATOM = 3
N_ATOM = 4
H_ATOM = 9
NEG_EIGVAL_THRESHOLD = 1e-6
SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=Path("runs/glycine_pt_scan_n36"))
    parser.add_argument("--orca-output-dir", type=Path, default=None)
    parser.add_argument(
        "--orca-engrad-dir",
        type=Path,
        default=None,
        help="Directory containing .engrad files. Defaults to --orca-output-dir.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def read_orca_matrix(lines: list[str], section: str) -> np.ndarray:
    start = lines.index(section) + 1
    n_rows = int(lines[start].split()[0])
    matrix = np.zeros((n_rows, n_rows), dtype=float)
    idx = start + 1
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("$"):
            break
        parts = line.split()
        if not parts:
            idx += 1
            continue
        if parts[0].startswith("#"):
            break
        # ORCA block headers are just column indices.
        if parts[0].isdigit() and len(parts) > 1:
            columns = [int(part) for part in parts]
            idx += 1
            while idx < len(lines):
                row_parts = lines[idx].split()
                if not row_parts:
                    idx += 1
                    continue
                if lines[idx].startswith("$") or lines[idx].lstrip().startswith("#"):
                    break
                row = int(row_parts[0])
                values = [float(value) for value in row_parts[1:]]
                if len(values) != len(columns):
                    break
                for col, value in zip(columns, values, strict=True):
                    matrix[row, col] = value
                idx += 1
            continue
        idx += 1
    return matrix


def read_orca_vector(lines: list[str], section: str) -> np.ndarray:
    start = lines.index(section) + 1
    n_rows = int(lines[start].split()[0])
    values = np.zeros(n_rows, dtype=float)
    for line in lines[start + 1 :]:
        if line.startswith("$"):
            break
        parts = line.split()
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
            values[int(parts[0])] = float(parts[1])
    return values


def read_atoms(lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = lines.index("$atoms") + 1
    n_atoms = int(lines[start].split()[0])
    symbols: list[str] = []
    masses: list[float] = []
    coords_bohr: list[list[float]] = []
    for line in lines[start + 1 : start + 1 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        masses.append(float(parts[1]))
        coords_bohr.append([float(parts[2]), float(parts[3]), float(parts[4])])
    atomic_numbers = np.array([SYMBOL_TO_Z[symbol] for symbol in symbols], dtype=int)
    return (
        np.asarray(symbols, dtype="U2"),
        atomic_numbers,
        np.asarray(masses, dtype=float),
        np.asarray(coords_bohr, dtype=float),
    )


def parse_hess(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text(errors="replace").splitlines()
    hessian_hartree_bohr2 = read_orca_matrix(lines, "$hessian")
    frequencies_cm1 = read_orca_vector(lines, "$vibrational_frequencies")
    normal_modes = read_orca_matrix(lines, "$normal_modes")
    symbols, atomic_numbers, masses_amu, coords_bohr = read_atoms(lines)
    return {
        "hessian_hartree_per_bohr2": hessian_hartree_bohr2,
        "hessian_ev_ang2": hessian_hartree_bohr2 * HARTREE_PER_BOHR2_TO_EV_PER_ANG2,
        "frequencies_cm1": frequencies_cm1,
        "normal_modes_orca": normal_modes,
        "symbols": symbols,
        "atomic_numbers": atomic_numbers,
        "masses_amu": masses_amu,
        "coords_bohr": coords_bohr,
        "coords_angstrom": coords_bohr * BOHR_TO_ANGSTROM,
    }


def parse_engrad(path: Path) -> dict[str, np.ndarray | float]:
    lines = path.read_text(errors="replace").splitlines()
    values: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            values.append(stripped)

    n_atoms = int(values[0])
    energy = float(values[1])
    grad_start = 2
    grad_stop = grad_start + 3 * n_atoms
    gradient = np.array([float(x) for x in values[grad_start:grad_stop]], dtype=float).reshape(n_atoms, 3)
    coord_values = values[grad_stop : grad_stop + n_atoms]
    coords_bohr = np.array([[float(x) for x in row.split()[1:4]] for row in coord_values], dtype=float)
    atomic_numbers = np.array([int(row.split()[0]) for row in coord_values], dtype=int)
    forces = -gradient
    return {
        "energy_hartree": energy,
        "gradient_hartree_per_bohr": gradient,
        "forces_hartree_per_bohr": forces,
        "forces_ev_ang": forces * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
        "coords_bohr": coords_bohr,
        "atomic_numbers": atomic_numbers,
    }


def distance_gradient(coords: np.ndarray, atom_a: int, atom_b: int) -> np.ndarray:
    grad = np.zeros_like(coords, dtype=float)
    vec = coords[atom_a] - coords[atom_b]
    unit = vec / max(float(np.linalg.norm(vec)), 1e-12)
    grad[atom_a] = unit
    grad[atom_b] = -unit
    return grad


def eckart_generators(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    masses = np.asarray(masses, dtype=float).reshape(-1)
    n_atoms = xyz.shape[0]
    sqrt_m = np.sqrt(masses)
    sqrt_m3 = np.repeat(sqrt_m, 3)
    com = (xyz * masses[:, None]).sum(axis=0) / masses.sum()
    rel = xyz - com[None, :]
    cols = []
    for axis in np.eye(3):
        col = sqrt_m3 * np.tile(axis, n_atoms)
        cols.append(col / max(float(np.linalg.norm(col)), 1e-12))
    rx, ry, rz = rel[:, 0], rel[:, 1], rel[:, 2]
    rotations = (
        np.stack([np.zeros_like(rx), -rz, ry], axis=1),
        np.stack([rz, np.zeros_like(ry), -rx], axis=1),
        np.stack([-ry, rx, np.zeros_like(rz)], axis=1),
    )
    for rot in rotations:
        col = (rot * sqrt_m[:, None]).reshape(-1)
        norm = np.linalg.norm(col)
        if norm > 1e-12:
            cols.append(col / norm)
    return np.stack(cols, axis=1)


def vibrational_basis(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    generators = eckart_generators(coords, masses)
    q, r = np.linalg.qr(generators, mode="reduced")
    diag = np.abs(np.diag(r))
    rank = max(int((diag > 1e-6).sum()), 1)
    u, _, _ = np.linalg.svd(q[:, :rank], full_matrices=True)
    return u[:, rank:]


def vibrational_eigh(
    hessian_ev_ang2: np.ndarray,
    coords_angstrom: np.ndarray,
    masses_amu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = masses_amu.size
    hessian = np.asarray(hessian_ev_ang2, dtype=float).reshape(3 * n_atoms, 3 * n_atoms)
    hessian = 0.5 * (hessian + hessian.T)
    m3 = np.repeat(masses_amu, 3)
    hessian_mw = hessian / np.sqrt(np.outer(m3, m3))
    q_vib = vibrational_basis(coords_angstrom, masses_amu)
    hessian_red = q_vib.T @ hessian_mw @ q_vib
    hessian_red = 0.5 * (hessian_red + hessian_red.T)
    evals, evecs_red = np.linalg.eigh(hessian_red)
    return evals, q_vib @ evecs_red


def normalized_curvature(hessian: np.ndarray, direction: np.ndarray) -> float:
    flat = direction.reshape(-1)
    denom = float(np.dot(flat, flat))
    return float(flat @ hessian @ flat / max(denom, 1e-12))


def mode_alignment(mode_mw: np.ndarray, direction_cart: np.ndarray, masses_amu: np.ndarray) -> float:
    direction_mw = direction_cart.reshape(-1) / np.sqrt(np.repeat(masses_amu, 3))
    denom = float(np.linalg.norm(mode_mw) * np.linalg.norm(direction_mw))
    return float(np.dot(mode_mw, direction_mw) / max(denom, 1e-12))


def main() -> None:
    args = parse_args()
    scan_dir = args.scan_dir
    orca_output_dir = args.orca_output_dir or scan_dir / "orca_outputs"
    orca_engrad_dir = args.orca_engrad_dir or orca_output_dir
    output = args.output or scan_dir / "orca_vib_cache.npz"
    summary_csv = args.summary_csv or scan_dir / "orca_vib_summary.csv"
    manifest = pd.read_csv(scan_dir / "scan_manifest.csv").sort_values("grid_id").reset_index(drop=True)

    hessians_hartree = []
    hessians_ev = []
    frequencies = []
    normal_modes_orca = []
    coords_bohr = []
    coords_angstrom = []
    evals = []
    modes_mw = []
    n_negative = []
    n_negative_freq = []
    q_nh_dirs = []
    q_oh_dirs = []
    pt_dirs = []
    energies_engrad = []
    gradients = []
    forces_hartree = []
    forces_ev_ang = []
    curv_q_nh = []
    curv_q_oh = []
    curv_pt = []
    unstable_pt_alignment = []
    rc_block_frob = []
    atomic_numbers = None
    masses_amu = None
    symbols = None

    for idx, row in enumerate(manifest.to_dict(orient="records"), start=1):
        hess_path = orca_output_dir / f"{Path(row['orca_input_path']).stem}.hess"
        if not hess_path.exists():
            raise FileNotFoundError(hess_path)
        parsed = parse_hess(hess_path)
        engrad_path = orca_engrad_dir / f"{Path(row['orca_input_path']).stem}.engrad"
        parsed_engrad = parse_engrad(engrad_path) if engrad_path.exists() else None
        if atomic_numbers is None:
            atomic_numbers = parsed["atomic_numbers"]
            masses_amu = parsed["masses_amu"]
            symbols = parsed["symbols"]
        if parsed_engrad is not None and not np.array_equal(atomic_numbers, parsed_engrad["atomic_numbers"]):
            raise ValueError(f"Atomic numbers do not match between {hess_path} and {engrad_path}")
        hessian_ev = parsed["hessian_ev_ang2"]
        coords = parsed["coords_angstrom"]
        vib_evals, vib_modes = vibrational_eigh(hessian_ev, coords, parsed["masses_amu"])
        q_nh = distance_gradient(coords, N_ATOM, H_ATOM)
        q_oh = distance_gradient(coords, O_ATOM, H_ATOM)
        pt = q_nh - q_oh
        rc_idx = np.array([3 * atom + comp for atom in (O_ATOM, N_ATOM, H_ATOM) for comp in range(3)])
        rc_block = hessian_ev[rc_idx[:, None], rc_idx]

        hessians_hartree.append(parsed["hessian_hartree_per_bohr2"])
        hessians_ev.append(hessian_ev)
        frequencies.append(parsed["frequencies_cm1"])
        normal_modes_orca.append(parsed["normal_modes_orca"])
        coords_bohr.append(parsed["coords_bohr"])
        coords_angstrom.append(coords)
        evals.append(vib_evals)
        modes_mw.append(vib_modes)
        n_negative.append(int((vib_evals < -NEG_EIGVAL_THRESHOLD).sum()))
        n_negative_freq.append(int((parsed["frequencies_cm1"] < -1e-6).sum()))
        q_nh_dirs.append(q_nh)
        q_oh_dirs.append(q_oh)
        pt_dirs.append(pt)
        if parsed_engrad is not None:
            energies_engrad.append(float(parsed_engrad["energy_hartree"]))
            gradients.append(parsed_engrad["gradient_hartree_per_bohr"])
            forces_hartree.append(parsed_engrad["forces_hartree_per_bohr"])
            forces_ev_ang.append(parsed_engrad["forces_ev_ang"])
        curv_q_nh.append(normalized_curvature(hessian_ev, q_nh))
        curv_q_oh.append(normalized_curvature(hessian_ev, q_oh))
        curv_pt.append(normalized_curvature(hessian_ev, pt))
        unstable_pt_alignment.append(mode_alignment(vib_modes[:, 0], pt, parsed["masses_amu"]))
        rc_block_frob.append(float(np.linalg.norm(0.5 * (rc_block + rc_block.T))))
        if idx % 100 == 0 or idx == len(manifest):
            print(f"parsed {idx}/{len(manifest)} Hessians", flush=True)

    evals_arr = np.stack(evals)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid_id": manifest["grid_id"].to_numpy(dtype=int),
        "q_nh": manifest["q_nh"].to_numpy(dtype=float),
        "q_oh": manifest["q_oh"].to_numpy(dtype=float),
        "atomic_numbers": np.asarray(atomic_numbers, dtype=int),
        "masses_amu": np.asarray(masses_amu, dtype=float),
        "symbols": np.asarray(symbols),
        "coords_bohr": np.stack(coords_bohr),
        "coords_angstrom": np.stack(coords_angstrom),
        "hessian_hartree_per_bohr2": np.stack(hessians_hartree),
        "hessian_ev_ang2": np.stack(hessians_ev),
        "frequencies_cm1": np.stack(frequencies),
        "normal_modes_orca": np.stack(normal_modes_orca),
        "vib_evals_ev_ang2_amu": evals_arr,
        "vib_modes_mw": np.stack(modes_mw),
        "n_negative": np.asarray(n_negative, dtype=int),
        "n_negative_frequency": np.asarray(n_negative_freq, dtype=int),
        "q_nh_direction": np.stack(q_nh_dirs),
        "q_oh_direction": np.stack(q_oh_dirs),
        "pt_direction": np.stack(pt_dirs),
        "curvature_q_nh_ev_ang2": np.asarray(curv_q_nh, dtype=float),
        "curvature_q_oh_ev_ang2": np.asarray(curv_q_oh, dtype=float),
        "curvature_pt_ev_ang2": np.asarray(curv_pt, dtype=float),
        "unstable_mode_pt_alignment": np.asarray(unstable_pt_alignment, dtype=float),
        "unstable_mode_pt_abs_alignment": np.abs(np.asarray(unstable_pt_alignment, dtype=float)),
        "reaction_center_hessian_frobenius_ev_ang2": np.asarray(rc_block_frob, dtype=float),
    }
    if len(forces_ev_ang) == len(manifest):
        payload.update(
            {
                "energy_hartree_engrad": np.asarray(energies_engrad, dtype=float),
                "gradient_hartree_per_bohr": np.stack(gradients),
                "forces_hartree_per_bohr": np.stack(forces_hartree),
                "forces_ev_ang": np.stack(forces_ev_ang),
            }
        )
    else:
        missing = len(manifest) - len(forces_ev_ang)
        print(f"Warning: {missing} .engrad files missing; ORCA forces will not be cached.", flush=True)
    np.savez_compressed(output, **payload)

    summary = manifest[["grid_id", "q_nh", "q_oh"]].copy()
    summary["lowest_frequency_cm1"] = np.stack(frequencies)[:, 6:].min(axis=1)
    summary["lowest_vib_eval_ev_ang2_amu"] = evals_arr[:, 0]
    summary["n_negative"] = np.asarray(n_negative, dtype=int)
    summary["n_negative_frequency"] = np.asarray(n_negative_freq, dtype=int)
    summary["curvature_q_nh_ev_ang2"] = np.asarray(curv_q_nh, dtype=float)
    summary["curvature_q_oh_ev_ang2"] = np.asarray(curv_q_oh, dtype=float)
    summary["curvature_pt_ev_ang2"] = np.asarray(curv_pt, dtype=float)
    summary["unstable_mode_pt_alignment"] = np.asarray(unstable_pt_alignment, dtype=float)
    summary["unstable_mode_pt_abs_alignment"] = np.abs(np.asarray(unstable_pt_alignment, dtype=float))
    summary["reaction_center_hessian_frobenius_ev_ang2"] = np.asarray(rc_block_frob, dtype=float)
    summary.to_csv(summary_csv, index=False)

    print(f"Wrote vibrational cache: {output}")
    print(f"Wrote vibrational summary: {summary_csv}")
    print(f"Rows: {len(summary)}")


if __name__ == "__main__":
    main()
