from __future__ import annotations

import os
import argparse
from typing import List, Tuple

import numpy as np
import h5py
from tqdm import tqdm
import scipy.constants as spc

from pyscf import dft, gto


# Bohr radius in m
BOHR2M = spc.value("Bohr radius")
# Bohr -> Å conversion factor
BOHR2ANG = BOHR2M * 1e10
# Hartree to eV
AU2EV = spc.value("Hartree energy in eV")


def build_molecule(
    atoms_bohr: List[Tuple[int, Tuple[float, float, float]]],
    charge: int = 0,
    multiplicity: int = 1,
) -> gto.Mole:
    """
    Build a PySCF molecule from a list like [(Z, (x,y,z)), ...] where coordinates are in Bohr.
    """
    spin = multiplicity - 1  # 2S = multiplicity - 1
    mol = gto.Mole()
    mol.atom = atoms_bohr  # list[(Z|(symbol), (x,y,z))]
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.basis = "6-31g(d)"  # wB97x_6-31G(d)
    mol.unit = "Bohr"
    mol.build()
    return mol


def compute_hessian_au_bohr2(
    mol: gto.Mole, multiplicity: int = 1, xc: str = "wb97x", debug_hint: str = ""
) -> np.ndarray | None:
    """
    Compute DFT Hessian with PySCF. Returns Hessian in atomic units (Hartree/Bohr^2)
    with shape (3N, 3N). Returns None if SCF fails.
    """
    is_open_shell = multiplicity != 1
    if is_open_shell:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.verbose = 0
    # Denser, unpruned grid for XC integration
    # mf.grids.level = 5
    mf.grids.atom_grid = (99, 590)  # try (175, 974) if needed
    mf.grids.prune = None
    try:
        mf.kernel()
    except Exception as e:
        print("\n" + ">" * 40)
        print(f"Error in SCF: {debug_hint}: \n{e}")
        print("<" * 40)
        return None
    if not mf.converged:
        print("\n" + ">" * 40)
        print(f"SCF did not converge: {debug_hint}")
        print("<" * 40)
        return None

    hobj = mf.Hessian()
    # If these attributes exist in your PySCF version:
    setattr(hobj, "conv_tol", 1e-10)
    setattr(hobj, "max_cycle", 100)
    hessian = hobj.kernel()
    # (N, N, 3, 3) where N is number of atoms
    N = mol.natm
    hes = hessian.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    return hes


def au_bohr2_to_ev_ang2(hess_au_bohr2: np.ndarray) -> np.ndarray:
    """Convert Hessian from Hartree/Bohr^2 to eV/Å^2."""
    return hess_au_bohr2 * (AU2EV / (BOHR2ANG * BOHR2ANG))


def main():
    ap = argparse.ArgumentParser(
        description="Compute DFT Hessians for Transition1x val reactants"
    )
    ap.add_argument(
        "--source_h5",
        type=str,
        default=os.path.abspath(
            os.path.join("Transition1x", "data", "transition1x.h5")
        ),
        help="Path to the original Transition1x HDF5 file",
    )
    ap.add_argument(
        "--dest_h5",
        type=str,
        default=os.path.abspath(
            os.path.join("data", "t1x_val_reactant_hessian_100.h5")
        ),
        help="Path to output HDF5 file with Hessians (eV/Å^2)",
    )
    ap.add_argument(
        "--noiserms",
        type=float,
        default=0.0,
        help="Per-atom RMS displacement (Å) added to geometry before Hessian; 0 disables noise",
    )
    ap.add_argument(
        "--limit", type=int, default=100, help="Number of val reactants to process"
    )
    args = ap.parse_args()

    if args.noiserms > 0.0:
        args.dest_h5 = args.dest_h5.replace(".h5", f"_noiserms{args.noiserms:.2f}.h5")

    print(f"Will save to {args.dest_h5}")

    os.makedirs(os.path.dirname(args.dest_h5), exist_ok=True)

    with h5py.File(args.source_h5, "r") as src, h5py.File(args.dest_h5, "w") as dst:
        if "val" not in src:
            raise RuntimeError("'val' split not found in source HDF5")
        val_src = src["val"]
        val_dst = dst.create_group("val")

        rng = np.random.default_rng(seed=42)

        count = 0
        val_index = 0  # index within original val split, by traversal order
        pbar = tqdm(total=args.limit, desc="Computing Hessians (val reactants)")
        for formula, grp in val_src.items():
            if formula not in val_dst:
                g_formula = val_dst.create_group(formula)
            else:
                g_formula = val_dst[formula]

            for rxn, subgrp in grp.items():
                if "reactant" not in subgrp:
                    continue

                # Copy full original group to preserve all keys and shapes
                dst.copy(subgrp, g_formula, name=rxn)

                g_rxn = g_formula[rxn]
                g_reactant = g_rxn["reactant"]

                reactant_grp = subgrp["reactant"]

                # Extract geometry for Hessian computation
                atomic_numbers = np.array(reactant_grp["atomic_numbers"], dtype=int)
                positions_all = np.array(
                    reactant_grp["positions"]
                )  # shape (T, N, 3) or (N, 3)

                positions = (
                    positions_all[0] if positions_all.ndim == 3 else positions_all
                )

                # Optionally add zero-mean Gaussian noise with specified per-atom RMS (Å)
                positions_used = positions.copy()
                if args.noiserms and args.noiserms > 0.0:
                    noise = rng.normal(0.0, 1.0, size=positions.shape)
                    # Scale noise so RMS of per-atom Euclidean displacement equals noiserms
                    current_rms = float(np.sqrt(np.mean(np.sum(noise * noise, axis=1))))
                    scale = (args.noiserms / current_rms) if current_rms > 0.0 else 0.0
                    displacement = scale * noise
                    positions_used = positions + displacement

                atoms_bohr: List[Tuple[int, Tuple[float, float, float]]] = [
                    (
                        int(Z),
                        (float(x / BOHR2ANG), float(y / BOHR2ANG), float(z / BOHR2ANG)),
                    )
                    for Z, (x, y, z) in zip(atomic_numbers, positions_used)
                ]

                mol = build_molecule(atoms_bohr)
                hessian_au = compute_hessian_au_bohr2(
                    mol, multiplicity=1, xc="wb97x", debug_hint=f"{formula}/{rxn}"
                )
                if hessian_au is None:
                    # On failure, remove the copied reaction to avoid partial data
                    del g_formula[rxn]
                    val_index += 1
                    print(f"Skipping {formula}/{rxn} due to SCF failure")
                    continue

                hessian_ev_ang2 = au_bohr2_to_ev_ang2(hessian_au)

                # New Hessian key in eV/Å^2 under reactant
                g_reactant.create_dataset(
                    "wB97x_6-31G(d).hessian", data=hessian_ev_ang2, compression="gzip"
                )
                # Store noise info and noised geometry (Å) if noise was applied
                if "noiserms" in g_reactant:
                    del g_reactant["noiserms"]
                g_reactant.create_dataset(
                    "noiserms", data=np.array(args.noiserms, dtype=np.float64)
                )
                if args.noiserms and args.noiserms > 0.0:
                    if "positions_noised" in g_reactant:
                        del g_reactant["positions_noised"]
                    g_reactant.create_dataset(
                        "positions_noised",
                        data=positions_used.astype(np.float64),
                        compression="gzip",
                    )
                # Store original val index for this reactant
                if "idx" in g_reactant:
                    del g_reactant["idx"]
                g_reactant.create_dataset(
                    "idx", data=np.array(val_index, dtype=np.int64)
                )

                count += 1
                pbar.update(1)
                val_index += 1
                if count >= args.limit:
                    if "count" in dst:
                        del dst["count"]
                    dst.create_dataset("count", data=np.array(count, dtype=np.int64))
                    pbar.close()
                    return

        pbar.close()
        if "count" in dst:
            del dst["count"]
        dst.create_dataset("count", data=np.array(count, dtype=np.int64))

    print(f"\nDone! Saved to {args.dest_h5}")


if __name__ == "__main__":
    """
    Add noise to geometry before Hessian computation.
    Molecules: 0.01 Å (tiny), 0.05 Å (typical), 0.10 Å (hard), 0.20 Å (extreme)

    # add 0.05 Å RMS per atom 
    uv run compute_dft_hessian_t1x.py --noiserms 0.05
    """
    main()
