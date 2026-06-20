#!/usr/bin/env python
"""Locate the genuine glycine proton-transfer TS at the GFN2-xTB level.

Uses ASE's dimer (min-mode) saddle search with the GFN2-xTB (tblite) calculator,
seeded from the Transition1x TS geometry and the proton-transfer direction
q_NH - q_OH as the initial eigenmode. Confirms exactly one imaginary mode whose
eigenvector aligns with q_NH - q_OH, and writes the TS geometry plus its
(s, sigma) location for overlay on the relaxed surface. Runs in the pinned
project env (numpy 1.26 + tblite); no sella required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write
from tblite.ase import TBLite

try:
    from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
except ImportError:  # older ASE layout
    from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate

O_ATOM, N_ATOM, H_ATOM = 3, 4, 9

EV = 1.602176634e-19
AMU = 1.66053906660e-27
ANG = 1.0e-10
C_CM = 2.99792458e10


def distance_gradient(coords: np.ndarray, a: int, b: int) -> np.ndarray:
    grad = np.zeros_like(coords)
    diff = coords[a] - coords[b]
    unit = diff / np.linalg.norm(diff)
    grad[a] = unit
    grad[b] = -unit
    return grad


def numerical_hessian(atoms, delta: float = 0.005) -> np.ndarray:
    n = len(atoms)
    base = atoms.get_positions()
    hess = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for c in range(3):
            for sign in (+1, -1):
                pos = base.copy()
                pos[i, c] += sign * delta
                atoms.set_positions(pos)
                f = atoms.get_forces().reshape(-1)
                hess[:, 3 * i + c] += -sign * f / (2 * delta)
    atoms.set_positions(base)
    return 0.5 * (hess + hess.T)


def vibrational_spectrum(hess_ev_ang2, coords, masses):
    """Return (eigvals in eV/Å²/amu, mass-weighted eigvecs) with trans/rot removed."""
    m3 = np.repeat(masses, 3)
    hmw = hess_ev_ang2 / np.sqrt(np.outer(m3, m3))
    # build translation + rotation basis (mass-weighted), orthonormalize, project out
    n = len(masses)
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    trans = np.zeros((3 * n, 3))
    for c in range(3):
        v = np.zeros(3 * n)
        v[c::3] = 1.0
        trans[:, c] = v * sqrt_m
    com = coords - np.average(coords, axis=0, weights=masses)
    rot = np.zeros((3 * n, 3))
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = 1.0
        cross = np.cross(np.tile(e, (n, 1)), com).reshape(-1)
        rot[:, axis] = cross * sqrt_m
    ext = np.hstack([trans, rot])
    q, _ = np.linalg.qr(ext)
    proj = np.eye(3 * n) - q @ q.T
    hred = proj @ hmw @ proj
    hred = 0.5 * (hred + hred.T)
    evals, evecs = np.linalg.eigh(hred)
    return evals, evecs


def eval_to_cm1(lam: float) -> float:
    omega2 = lam * EV / (ANG**2 * AMU)
    if omega2 >= 0:
        return float(np.sqrt(omega2) / (2 * np.pi * C_CM))
    return float(-np.sqrt(-omega2) / (2 * np.pi * C_CM))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=Path,
                   default=Path("runs/glycine_pt_scan_n36/xyz/reference_ts.xyz"))
    p.add_argument("--out-xyz", type=Path,
                   default=Path("runs/glycine_pt_scan_relaxed/xyz/ts_xtb_saddle.xyz"))
    p.add_argument("--out-json", type=Path,
                   default=Path("runs/glycine_pt_scan_relaxed/ts_xtb_saddle.json"))
    p.add_argument("--fmax", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()

    atoms = read(args.seed)
    atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)

    pt_seed = (distance_gradient(atoms.get_positions(), N_ATOM, H_ATOM)
               - distance_gradient(atoms.get_positions(), O_ATOM, H_ATOM))

    control = DimerControl(
        initial_eigenmode_method="displacement",
        displacement_method="vector",
        logfile=None,
        dimer_separation=0.01,
        trial_angle=np.pi / 4.0,
        max_num_rot=10,
    )
    dimer_atoms = MinModeAtoms(atoms, control)
    dimer_atoms.displace(displacement_vector=0.01 * pt_seed)
    relax = MinModeTranslate(dimer_atoms, logfile=None, trajectory=None)
    relax.run(fmax=args.fmax, steps=args.steps)

    coords = atoms.get_positions()
    masses = atoms.get_masses()
    q_nh = float(np.linalg.norm(coords[N_ATOM] - coords[H_ATOM]))
    q_oh = float(np.linalg.norm(coords[O_ATOM] - coords[H_ATOM]))
    s = q_nh - q_oh
    sigma = q_nh + q_oh

    hess = numerical_hessian(atoms)
    evals, evecs = vibrational_spectrum(hess, coords, masses)
    n_imag = int((evals < -1e-6).sum())
    freqs = [eval_to_cm1(v) for v in evals[:6]]

    pt_dir = (distance_gradient(coords, N_ATOM, H_ATOM)
              - distance_gradient(coords, O_ATOM, H_ATOM)).reshape(-1)
    pt_mw = pt_dir / np.sqrt(np.repeat(masses, 3))
    soft = evecs[:, 0]
    alignment = float(abs(np.dot(soft, pt_mw) / (np.linalg.norm(soft) * np.linalg.norm(pt_mw))))

    args.out_xyz.parent.mkdir(parents=True, exist_ok=True)
    write(args.out_xyz, atoms)
    payload = {
        "method": "GFN2-xTB dimer (min-mode) saddle search",
        "seed": str(args.seed),
        "q_nh": q_nh, "q_oh": q_oh, "s": s, "sigma": sigma,
        "n_imaginary": n_imag,
        "lowest_freqs_cm1": freqs,
        "softest_mode_pt_alignment": alignment,
        "fmax_reached": float(np.abs(atoms.get_forces()).max()),
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
