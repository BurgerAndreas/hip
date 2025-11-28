import os
import time
import numpy as np
import pandas as pd
from pyscf import gto, dft
import scipy.constants as spc

from hip.ff_lmdb import LmdbDataset, Z_TO_ATOM_SYMBOL

# from pysisyphus.constants import AU2J, BOHR2ANG, C, R, AU2KJPERMOL, NA
# Bohr radius in m
BOHR2M = spc.value("Bohr radius")
# Bohr -> Å conversion factor
BOHR2ANG = BOHR2M * 1e10
# Å -> Bohr conversion factor
ANG2BOHR = 1 / BOHR2ANG
# Hartree to J
AU2J = spc.value("Hartree energy")
# Speed of light in m/s
C = spc.c
NA = spc.Avogadro
HARTREE2EV = spc.physical_constants["Hartree energy in eV"][0]


def build_molecule(atom_types, positions):
    """
    Constructs a PySCF Mole object from atom types and positions.
    """
    atom_str = ""
    for at, pos in zip(atom_types, positions):
        atom_str += f"{at} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}; "

    mol = gto.M(
        atom=atom_str,
        unit="Angstrom",
        verbose=0,  # Suppress generic output
        charge=0,
        spin=0,
    )
    mol.build()
    return mol


def run_dft_calculation(mol, functional, basis_set, label):
    """
    Runs a DFT calculation for Energy, Force, and Hessian.
    """
    print(f"\n--- Starting {label} Calculation ---")
    print(f"Theory: {functional}/{basis_set}")

    start = time.perf_counter()

    # 1. Setup Mean Field (RKS)
    mf = dft.RKS(mol)
    mf.xc = functional
    mf.basis = basis_set

    # CRITICAL: Meta-GGAs (like wB97M-V) require dense integration grids
    # to avoid noise in gradients/Hessians.
    mf.grids.level = 5

    # Enable Non-Local Correlation (NLC) grids if the functional requires it (wB97M-V)
    if "wb97m" in functional.lower():
        mf.nlcgrids.level = 5

    # 2. Calculate Energy
    try:
        mf.kernel()
        energy = mf.e_tot
        print(f"Energy calculated: {energy:.6f} Hartree")
    except Exception as e:
        print(f"Energy convergence failed: {e}")
        return None

    # 3. Calculate Forces (Gradients)
    # Force = -Gradient
    try:
        g = mf.nuc_grad_method()
        gradients = g.kernel()
        forces = -gradients
        print(f"Forces calculated (Max Force: {np.max(np.abs(forces)):.6f} a.u.)")
    except Exception as e:
        print(f"Force calculation failed: {e}")
        return None

    # 4. Calculate Hessian
    # Note: This is the most expensive step.
    try:
        h_obj = mf.Hessian()
        hessian_mat = h_obj.kernel()
        print(f"Hessian calculated (Shape: {hessian_mat.shape})")
    except Exception as e:
        print(
            f"Hessian calculation failed (likely not implemented for this functional in current Libxc): {e}"
        )
        hessian_mat = None

    elapsed = time.perf_counter() - start
    print(f"{label} wall time: {elapsed:.2f} s")

    return {
        "energy": energy,  # Hartree
        "forces": forces,  # Hartree/Bohr
        "hessian": hessian_mat,  # Hartree/Bohr^2
        "runtime": elapsed,  # Seconds
    }


def main():
    dataset_dir = os.path.expanduser(
        "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/"
    )
    lmdb_path = os.path.join(dataset_dir, "ts1x-val.lmdb")

    dataset = LmdbDataset(lmdb_path)
    n_total = len(dataset)
    n_samples = min(100, n_total)

    rng = np.random.default_rng()
    indices = rng.choice(n_total, size=n_samples, replace=False)

    csv_path = "compare_dft_ts1x_val.csv"

    for idx in indices:
        data = dataset[int(idx)]

        z_list = data.z.cpu().numpy().tolist()
        pos = data.pos.cpu().numpy()
        atom_types = [Z_TO_ATOM_SYMBOL[int(z)] for z in z_list]

        mol = build_molecule(atom_types, pos)

        results_horm = run_dft_calculation(
            mol,
            functional="wb97x",
            basis_set="6-31g*",
            label=f"HORM (Hessian Dataset) idx={int(idx)}",
        )

        results_omol = run_dft_calculation(
            mol,
            functional="wb97m_v",
            basis_set="def2-tzvpd",
            label=f"OMol25 idx={int(idx)}",
        )

        if not (results_horm and results_omol):
            continue

        e_diff = (results_omol["energy"] - results_horm["energy"]) * HARTREE2EV

        f_diff = np.abs(results_omol["forces"] - results_horm["forces"])
        mean_force_diff = np.mean(f_diff)
        force_ev_ang = mean_force_diff * HARTREE2EV / BOHR2ANG

        h_diff = np.nan
        h_diff_ev_ang = np.nan
        if results_omol["hessian"] is not None and results_horm["hessian"] is not None:
            n_atoms = len(atom_types)
            h_omol_2d = results_omol["hessian"].reshape(n_atoms * 3, n_atoms * 3)
            h_horm_2d = results_horm["hessian"].reshape(n_atoms * 3, n_atoms * 3)
            h_diff = np.mean(np.abs(h_omol_2d - h_horm_2d))
            h_diff_ev_ang = h_diff * HARTREE2EV / BOHR2ANG**2

        row = {
            "dataset_index": int(idx),
            "n_atoms": len(atom_types),
            "energy_horm_hartree": results_horm["energy"],
            "energy_omol_hartree": results_omol["energy"],
            "energy_diff_ev": e_diff,
            "mean_force_diff_hartree_per_bohr": mean_force_diff,
            "mean_force_diff_ev_per_ang": force_ev_ang,
            "mean_hessian_diff_hartree_per_bohr2": h_diff,
            "mean_hessian_diff_ev_per_ang2": h_diff_ev_ang,
            "runtime_horm_s": results_horm["runtime"],
            "runtime_omol_s": results_omol["runtime"],
        }

        df = pd.DataFrame([row])
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", header=header, index=False)


if __name__ == "__main__":
    main()
