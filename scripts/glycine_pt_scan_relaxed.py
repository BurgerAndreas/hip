#!/usr/bin/env python
"""Relaxed 2D glycine proton-transfer scan in difference coordinates.

This is an independent companion to ``scripts/glycine_pt_scan.py``. Instead of a
rigid scan that freezes every atom except the transferring hydrogen at the TS
geometry, this script:

* lays the grid out in the chemically meaningful collective variables
  ``s = q_NH - q_OH`` (antisymmetric stretch / reaction coordinate) and
  ``sigma = q_NH + q_OH`` (heavy-atom compression), and
* performs a GFN2-xTB constrained relaxation at every grid node, holding the two
  bond distances ``q_NH = d(N4, H9)`` and ``q_OH = d(O3, H9)`` fixed while every
  other degree of freedom relaxes.

The reactant / TS / product reference geometries are additionally relaxed
without constraints so their collective-variable values anchor the grid.

Outputs (energies, forces, Hessians) at DFT / HIP / autograd are produced later
by evaluating those models on the relaxed geometries written here; this script
also emits matching ORCA single-point input files for that downstream step.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixInternals
from ase.io import read
from ase.optimize import BFGS
from tblite.ase import TBLite

N_ATOM = 4
O_ATOM = 3
H_ATOM = 9
EV_TO_KCALMOL = 23.060541945329334


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_reference(name: str) -> Path:
    return _project_root() / "runs" / "glycine_pt_scan_n36" / "xyz" / name


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
    """Initial guess: move H9 to satisfy d(N4,H9)=q_nh and d(O3,H9)=q_oh."""
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


def _bond(pos: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(pos[i] - pos[j]))


def make_calculator(method: str) -> TBLite:
    return TBLite(method=method, verbosity=0)


def relax(
    atoms: Atoms,
    method: str,
    fmax: float,
    max_steps: int,
    constraints: FixInternals | None,
) -> tuple[bool, float, np.ndarray]:
    work = atoms.copy()
    if constraints is not None:
        work.set_constraint(constraints)
    work.calc = make_calculator(method)
    opt = BFGS(work, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    energy_ev = float(work.get_potential_energy())
    return bool(opt.converged()), energy_ev, work.get_positions()


def relax_reference(
    path: Path,
    label: str,
    method: str,
    fmax: float,
    max_steps: int,
    out_xyz_dir: Path,
    symbols: list[str],
) -> dict[str, object]:
    atoms = read(str(path))
    converged, energy_ev, coords = relax(atoms, method, fmax, max_steps, None)
    q_nh = _bond(coords, N_ATOM, H_ATOM)
    q_oh = _bond(coords, O_ATOM, H_ATOM)
    write_xyz(
        out_xyz_dir / f"reference_{label}_relaxed.xyz",
        symbols,
        coords,
        f"{label} GFN2-xTB relaxed q_nh={q_nh:.4f} q_oh={q_oh:.4f}",
    )
    return {
        "label": label,
        "q_nh": q_nh,
        "q_oh": q_oh,
        "s": q_nh - q_oh,
        "sigma": q_nh + q_oh,
        "energy_ev": energy_ev,
        "converged": converged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ts-xyz", type=Path, default=_default_reference("reference_ts.xyz"))
    parser.add_argument("--reactant-xyz", type=Path, default=_default_reference("reference_reactant.xyz"))
    parser.add_argument("--product-xyz", type=Path, default=_default_reference("reference_product.xyz"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/glycine_pt_scan_relaxed"))
    parser.add_argument("--xtb-method", default="GFN2-xTB")
    parser.add_argument("--s-min", type=float, default=-1.8)
    parser.add_argument("--s-max", type=float, default=1.8)
    parser.add_argument("--n-s", type=int, default=31)
    parser.add_argument("--sigma-min", type=float, default=2.4)
    parser.add_argument("--sigma-max", type=float, default=3.8)
    parser.add_argument("--n-sigma", type=int, default=21)
    parser.add_argument("--q-min", type=float, default=0.85)
    parser.add_argument("--q-max", type=float, default=2.75)
    parser.add_argument("--energy-ceiling-kcal", type=float, default=None)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument(
        "--orca-route",
        default="! wB97X-D3 6-31G(d) TightSCF Grid5 FinalGrid6 EnGrad Freq",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    xyz_dir = out_dir / "xyz"
    orca_dir = out_dir / "orca_inputs"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    orca_dir.mkdir(parents=True, exist_ok=True)

    ts_atoms = read(str(args.ts_xyz))
    symbols = ts_atoms.get_chemical_symbols()
    atomic_nums = ts_atoms.get_atomic_numbers().astype(int)
    ts_coords = ts_atoms.get_positions()
    perp_ref = _perpendicular_reference(ts_coords)

    # Anchor: relax the reactant / TS / product references without constraints.
    references = []
    for label, ref_path in (
        ("reactant", args.reactant_xyz),
        ("ts", args.ts_xyz),
        ("product", args.product_xyz),
    ):
        ref = relax_reference(ref_path, label, args.xtb_method, args.fmax, args.max_steps, xyz_dir, symbols)
        references.append(ref)
        print(
            f"[anchor] {label:9s} q_nh={ref['q_nh']:.3f} q_oh={ref['q_oh']:.3f} "
            f"s={ref['s']:+.3f} sigma={ref['sigma']:.3f} converged={ref['converged']}",
            flush=True,
        )

    s_values = np.linspace(args.s_min, args.s_max, args.n_s)
    sigma_values = np.linspace(args.sigma_min, args.sigma_max, args.n_sigma)

    manifest_path = out_dir / "scan_manifest.csv"
    orca_list_path = out_dir / "orca_input_list.txt"
    fieldnames = [
        "grid_id",
        "i_s",
        "i_sigma",
        "s",
        "sigma",
        "q_nh_target",
        "q_oh_target",
        "q_nh_relaxed",
        "q_oh_relaxed",
        "xtb_energy_ev",
        "converged",
        "xyz_path",
        "orca_input_path",
    ]

    rows: list[dict[str, object]] = []
    coords_all: list[np.ndarray] = []
    grid_id = 0
    n_skipped_bounds = 0
    n_skipped_place = 0

    for i_s, s in enumerate(s_values):
        for i_sigma, sigma in enumerate(sigma_values):
            q_nh = 0.5 * (sigma + s)
            q_oh = 0.5 * (sigma - s)
            if not (args.q_min <= q_nh <= args.q_max and args.q_min <= q_oh <= args.q_max):
                n_skipped_bounds += 1
                continue
            guess = place_transfer_hydrogen(ts_coords, float(q_nh), float(q_oh), perp_ref)
            if guess is None:
                n_skipped_place += 1
                continue

            atoms = Atoms(numbers=atomic_nums, positions=guess)
            constraints = FixInternals(
                bonds=[[float(q_nh), [N_ATOM, H_ATOM]], [float(q_oh), [O_ATOM, H_ATOM]]]
            )
            try:
                converged, energy_ev, coords = relax(
                    atoms, args.xtb_method, args.fmax, args.max_steps, constraints
                )
            except Exception as exc:  # noqa: BLE001 - record and continue on solver failure
                print(f"[warn] grid_id={grid_id} s={s:+.3f} sigma={sigma:.3f} relax failed: {exc}", flush=True)
                continue

            q_nh_rel = _bond(coords, N_ATOM, H_ATOM)
            q_oh_rel = _bond(coords, O_ATOM, H_ATOM)
            name = f"grid_{grid_id:04d}_s_{s:+.3f}_sig_{sigma:.3f}"
            xyz_path = xyz_dir / f"{name}.xyz"
            inp_path = orca_dir / f"{name}.inp"
            comment = (
                f"glycine_pt relaxed xtb={args.xtb_method} "
                f"s={s:.6f} sigma={sigma:.6f} q_nh={q_nh_rel:.6f} q_oh={q_oh_rel:.6f}"
            )
            write_xyz(xyz_path, symbols, coords, comment)
            write_orca_input(inp_path, symbols, coords, args.orca_route, args.charge, args.multiplicity)

            rows.append(
                {
                    "grid_id": grid_id,
                    "i_s": i_s,
                    "i_sigma": i_sigma,
                    "s": float(s),
                    "sigma": float(sigma),
                    "q_nh_target": float(q_nh),
                    "q_oh_target": float(q_oh),
                    "q_nh_relaxed": q_nh_rel,
                    "q_oh_relaxed": q_oh_rel,
                    "xtb_energy_ev": energy_ev,
                    "converged": converged,
                    "xyz_path": str(xyz_path.resolve()),
                    "orca_input_path": str(inp_path.resolve()),
                }
            )
            coords_all.append(coords)
            grid_id += 1
            if grid_id % 25 == 0:
                print(f"[progress] relaxed {grid_id} nodes (latest s={s:+.3f} sigma={sigma:.3f})", flush=True)

    if not rows:
        raise RuntimeError("No grid nodes were relaxed; check bounds.")

    energies_ev = np.array([row["xtb_energy_ev"] for row in rows], dtype=float)
    energies_rel = (energies_ev - energies_ev.min()) * EV_TO_KCALMOL
    for row, rel in zip(rows, energies_rel, strict=True):
        row["xtb_energy_relative_kcalmol"] = float(rel)

    # Optional energy-ceiling trim (drops high-energy unphysical corners).
    if args.energy_ceiling_kcal is not None:
        keep = energies_rel <= args.energy_ceiling_kcal
        n_dropped = int((~keep).sum())
        rows = [row for row, k in zip(rows, keep, strict=True) if k]
        coords_all = [c for c, k in zip(coords_all, keep, strict=True) if k]
        print(f"[ceiling] dropped {n_dropped} nodes above {args.energy_ceiling_kcal} kcal/mol", flush=True)

    fieldnames_out = fieldnames + ["xtb_energy_relative_kcalmol"]
    with manifest_path.open("w", newline="") as mh, orca_list_path.open("w") as lh:
        writer = csv.DictWriter(mh, fieldnames=fieldnames_out)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames_out})
            lh.write(f"{row['orca_input_path']}\n")

    np.savez_compressed(
        out_dir / "xtb_relaxed_arrays.npz",
        atomic_numbers=atomic_nums,
        coords_angstrom=np.stack(coords_all, axis=0),
        xtb_energy_ev=np.array([row["xtb_energy_ev"] for row in rows], dtype=float),
        xtb_energy_relative_kcalmol=np.array(
            [row["xtb_energy_relative_kcalmol"] for row in rows], dtype=float
        ),
        s=np.array([row["s"] for row in rows], dtype=float),
        sigma=np.array([row["sigma"] for row in rows], dtype=float),
        q_nh=np.array([row["q_nh_relaxed"] for row in rows], dtype=float),
        q_oh=np.array([row["q_oh_relaxed"] for row in rows], dtype=float),
    )

    metadata = {
        "description": "GFN2-xTB constrained-relaxed 2D glycine proton-transfer scan in (s, sigma)",
        "xtb_method": args.xtb_method,
        "n_atom": N_ATOM,
        "o_atom": O_ATOM,
        "h_atom": H_ATOM,
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "orca_route": args.orca_route,
        "fmax": args.fmax,
        "max_steps": args.max_steps,
        "s_range": [args.s_min, args.s_max],
        "n_s": args.n_s,
        "sigma_range": [args.sigma_min, args.sigma_max],
        "n_sigma": args.n_sigma,
        "q_bounds": [args.q_min, args.q_max],
        "energy_ceiling_kcal": args.energy_ceiling_kcal,
        "n_nodes_written": len(rows),
        "n_skipped_q_bounds": n_skipped_bounds,
        "n_skipped_placement": n_skipped_place,
        "references": references,
        "ts_xyz": str(args.ts_xyz),
        "reactant_xyz": str(args.reactant_xyz),
        "product_xyz": str(args.product_xyz),
    }
    with (out_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote {len(rows)} relaxed grid geometries", flush=True)
    print(f"Skipped (q bounds): {n_skipped_bounds}, skipped (placement): {n_skipped_place}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Relaxed XYZ: {xyz_dir}", flush=True)
    print(f"ORCA inputs: {orca_dir}", flush=True)
    print(f"Arrays: {out_dir / 'xtb_relaxed_arrays.npz'}", flush=True)


if __name__ == "__main__":
    main()
