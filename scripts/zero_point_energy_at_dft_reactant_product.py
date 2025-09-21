"""
Compute zero-point energies (ZPE) for DFT-relaxed reactant and product geometries,
then report per-geometry ZPEs and per-reaction ΔZPE and ΔΔZPE metrics for DFT and ML models.

Definitions:
- ZPE_method(geom) from vibrational frequencies via Eckart-projected, mass-weighted Hessian
- ΔZPE_method = ZPE_method(product) - ZPE_method(reactant)
- ΔΔZPE = |ΔZPE_ML - ΔZPE_DFT|

All units are eV per molecule.
"""

import argparse
import os
import time
import pathlib
import numpy as np
import torch

# pysisyphus + Equiformer
from pysisyphus.Geometry import Geometry
from pysisyphus.optimizers.RFOptimizer import RFOptimizer
from pysisyphus.constants import BOHR2ANG, AU2EV
from pysisyphus.calculators.PySCF import PySCF as PysisPySCF

# dataset + project paths
from hip.t1x_dft_dataloader import Dataloader as T1xDFTDataloader
from hip.path_config import ROOT_DIR

from hip.frequency_analysis import analyze_frequencies, eckart_projection_notmw  # noqa: F401
from hip.frequency_analysis import eigval_to_wavenumber

from hip.training_module import PotentialModule
from ocpmodels.common.relaxation.ase_utils import (
    coord_atoms_to_torch_geometric_hessian,
)
from nets.prediction_utils import compute_extra_props

# DFT (PySCF)
from pyscf import dft, gto
import scipy.constants as spc
import pandas as pd


Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}


def _write_xyz(path, atomssymbols, coords_ang):
    n = len(atomssymbols)
    lines = [str(n), "relaxed geometry"]
    for el, (x, y, z) in zip(atomssymbols, coords_ang):
        lines.append(f"{el} {x:.8f} {y:.8f} {z:.8f}")
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def _read_xyz(path):
    lines = pathlib.Path(path).read_text().strip().splitlines()
    try:
        n = int(lines[0].strip())
    except Exception:
        raise ValueError("XYZ: first line must be atom count")
    body = lines[2 : 2 + n]
    atoms, coords3d = [], []
    for ln in body:
        el, x, y, z = ln.split()[:4]
        atoms.append(el)
        coords3d.append([float(x), float(y), float(z)])
    return atoms, np.asarray(coords3d, float)


def _get_derivatives(x, y, retain_graph=None, create_graph=False):
    grad = torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph
    )[0]
    return grad


def compute_hessian(coords, energy, forces=None):
    if forces is None:
        forces = -_get_derivatives(coords, energy, create_graph=True)
    n_comp = forces.reshape(-1).shape[0]
    hess = []
    for f in forces.reshape(-1):
        hess_row = _get_derivatives(coords, -f, retain_graph=True)
        hess.append(hess_row)
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


def _load_model(ckpt, device):
    m = PotentialModule.load_from_checkpoint(ckpt, strict=False).potential
    return m.to(device).eval()


def _hessian_autograd_with_model(
    model, _coords_ang, _atomic_numbers, device, do_autograd=True
):
    use_pbc = getattr(model, "use_pbc", False)
    batch = coord_atoms_to_torch_geometric_hessian(
        coords=_coords_ang,
        atomic_nums=_atomic_numbers,
        cutoff=getattr(model, "cutoff", 12.0),
        max_neighbors=getattr(model, "max_neighbors", None),
        use_pbc=use_pbc,
        with_grad=True,
        cutoff_hessian=getattr(model, "cutoff_hessian", 100.0),
    )
    batch.ae = torch.tensor(np.array([len(_atomic_numbers)]), dtype=torch.int64)
    batch = batch.to(device)
    batch = compute_extra_props(batch, pos_require_grad=True)

    model_name = model.__class__.__name__.lower()
    n_atoms = batch.pos.shape[0]
    if "leftnet" in model_name:
        batch.pos.requires_grad_()
        energy_model, force_model = model.forward_autograd(batch)
        hessian_model = compute_hessian(batch.pos, energy_model, force_model)
    elif "equiformer" in model_name:
        if do_autograd:
            batch.pos.requires_grad_()
            energy_model, force_model, out = model.forward(
                batch, otf_graph=False, hessian=False
            )
            hessian_model = compute_hessian(batch.pos, energy_model, force_model)
        else:
            with torch.no_grad():
                energy_model, force_model, out = model.forward(
                    batch, otf_graph=False, hessian=True, add_props=True
                )
            hessian_model = out["hessian"].reshape(n_atoms * 3, n_atoms * 3)
    else:
        # AlphaNet (default)
        batch.pos.requires_grad_()
        energy_model, force_model = model.forward(batch)
        hessian_model = compute_hessian(batch.pos, energy_model, force_model)

    hessian_model = hessian_model.reshape(n_atoms * 3, n_atoms * 3)
    return hessian_model.detach().cpu().numpy()


def _to_au(h_ev_ang2):
    return h_ev_ang2 / AU2EV * (BOHR2ANG * BOHR2ANG)


def _zpe_from_hessian_au(h_au, coords_bohr, atomsymbols):
    proj = eckart_projection_notmw(h_au, coords_bohr.reshape(-1), atomsymbols)
    eigvals, _ = np.linalg.eigh(proj)
    # keep strictly positive vibrational modes
    eigvals = eigvals[eigvals > 0]
    wavenumbers_cm = eigval_to_wavenumber(eigvals)
    zpe_J = 0.5 * spc.h * spc.c * np.sum(wavenumbers_cm * 100.0)
    return zpe_J / spc.e


def relax_with_dft(
    atomssymbols, coords_ang, *, thresh="gau", max_cycles=200, out_dir="."
):
    coords_bohr = coords_ang / BOHR2ANG
    geom = Geometry(atomssymbols, coords_bohr, coord_type="redund")
    base_calc = PysisPySCF(
        basis="6-31g(d)",
        xc="wb97x",
        method="dft",
        charge=0,
        mult=1,
        mem=4000,
        pal=1,
        verbose=0,
    )
    geom.set_calculator(base_calc)
    opt = RFOptimizer(
        geom,
        thresh=thresh,
        trust_radius=0.3,
        hessian_init="unit",
        hessian_update="bfgs",
        hessian_recalc=None,
        line_search=True,
        out_dir=out_dir,
        max_cycles=max_cycles,
    )
    opt.run()
    final_coords_ang = (geom._coords).reshape(-1, 3) * BOHR2ANG
    return geom, final_coords_ang


def compute_dft_hessian_at_geometry(atomic_numbers, geom_coords_bohr):
    atoms_bohr = [
        (int(Z), (float(x), float(y), float(z)))
        for Z, (x, y, z) in zip(atomic_numbers, geom_coords_bohr.reshape(-1, 3))
    ]
    mol = gto.Mole()
    mol.atom = atoms_bohr
    mol.charge = 0
    mol.spin = 0
    mol.basis = "6-31g(d)"
    mol.unit = "Bohr"
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = "wb97x"
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.verbose = 0
    mf.grids.atom_grid = (99, 590)
    mf.grids.prune = None
    mf.kernel()
    if not mf.converged:
        print("PySCF SCF did not converge")
        return None

    hobj = mf.Hessian()
    setattr(hobj, "conv_tol", 1e-10)
    setattr(hobj, "max_cycle", 100)
    hessian_au = hobj.kernel()  # (N,N,3,3)
    N = mol.natm
    hess_cart_au = hessian_au.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    return hess_cart_au


def print_deltadelta_latex(delta_csv_path, decimals=4):
    if not os.path.isfile(delta_csv_path):
        raise FileNotFoundError(f"CSV not found: {delta_csv_path}")
    df = pd.read_csv(delta_csv_path)
    required = {"model", "method", "delta_zpe_eV", "delta_zpe_dft_eV"}
    missing = required - set(df.columns)
    if len(missing) > 0:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Use signed difference for STD, absolute for MAE
    df["error"] = df["delta_zpe_eV"] - df["delta_zpe_dft_eV"]

    # One row per method/model (excluding DFT itself)
    models = [
        ("AlphaNet", "autograd"),
        ("LeftNet", "autograd"),
        ("LeftNet-DF", "autograd"),
        ("EquiformerV2", "autograd"),
        ("EquiformerV2", "predict"),
    ]

    rows = []
    for model_name, method_value in models:
        sub = df[(df["model"] == model_name) & (df["method"] == method_value)]
        if len(sub) == 0:
            zpe_err_str = "-"
        else:
            err = sub["error"].to_numpy()
            mae = float(np.mean(np.abs(err))) if len(err) > 0 else np.nan
            std = float(np.std(err, ddof=0)) if len(err) > 0 else np.nan
            if np.isnan(mae) or np.isnan(std):
                zpe_err_str = "-"
            else:
                zpe_err_str = f"{mae:.{int(decimals)}f} ({std:.{int(decimals)}f})"
        rows.append(
            {
                "Hessian": method_value,
                "Model": model_name,
                "ΔΔZPE MAE (Std) [eV]": zpe_err_str,
            }
        )

    table = pd.DataFrame(rows, columns=["Hessian", "Model", "ΔΔZPE MAE (Std) [eV]"])
    latex = table.to_latex(index=False, escape=True)
    print()
    print(latex)
    return latex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="data/t1x_val_reactant_hessian_100.h5",
        help="Path to T1x HDF5 file",
    )
    ap.add_argument("--max_samples", type=int, default=10)
    ap.add_argument("--max_cycles", type=int, default=200)
    ap.add_argument("--thresh", type=str, default="gau")
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--redo_relax", type=bool, default=False)
    ap.add_argument("--redo_dft", type=bool, default=False)
    ap.add_argument("--verbose", type=bool, default=False)
    ap.add_argument("--ckpt_alpha", type=str, default="ckpt/alpha.ckpt")
    ap.add_argument("--ckpt_left", type=str, default="ckpt/left.ckpt")
    ap.add_argument("--ckpt_left_df", type=str, default="ckpt/left-df.ckpt")
    ap.add_argument("--ckpt_eqv2_autograd", type=str, default="ckpt/eqv2.ckpt")
    ap.add_argument(
        "--ckpt_eqv2_predict",
        type=str,
        default="/ssd/Code/ReactBench/ckpt/hesspred/hesspredalldatanumlayershessian3presetluca8w10onlybz128-581483-20250826-074746.ckpt",
    )
    args = ap.parse_args()

    # Resolve output directory
    if args.out_dir is None:
        source_label = os.path.splitext(os.path.basename(args.dataset))[0]
        args.out_dir = os.path.join(
            ROOT_DIR,
            "runs_zpe_rp",
            f"{source_label}_dftrelax_{args.thresh.replace('_', '')}",
        )

    os.makedirs(args.out_dir, exist_ok=True)
    xyz_dir = os.path.join(args.out_dir, "relaxed_xyz")
    os.makedirs(xyz_dir, exist_ok=True)
    dft_dir = os.path.join(args.out_dir, "dft")
    dft_grad_dir = os.path.join(dft_dir, "gradients")
    dft_hess_dir = os.path.join(dft_dir, "hessians")
    os.makedirs(dft_grad_dir, exist_ok=True)
    os.makedirs(dft_hess_dir, exist_ok=True)

    # Save outputs
    zpe_dir = os.path.join(args.out_dir, "zpe")
    os.makedirs(zpe_dir, exist_ok=True)
    zpe_csv = os.path.join(zpe_dir, "zpe_reactant_product_{args.max_samples}.csv")
    delta_csv = os.path.join(zpe_dir, "delta_zpe_{args.max_samples}.csv")

    # Early exit if delta CSV exists and not redoing
    if os.path.isfile(delta_csv) and (not args.redo_relax) and (not args.redo_dft):
        print(f"Found existing ΔZPE CSV: {delta_csv}. Loading instead of recomputing.")
        print_deltadelta_latex(delta_csv)
        return

    # Load models
    device = args.device
    model_alpha = _load_model(args.ckpt_alpha, device)
    model_left = _load_model(args.ckpt_left, device)
    model_left_df = _load_model(args.ckpt_left_df, device)
    model_eqv2_predict = _load_model(args.ckpt_eqv2_predict, device)
    model_eqv2_autograd = _load_model(args.ckpt_eqv2_autograd, device)

    dataset = T1xDFTDataloader(args.dataset, datasplit="val", only_final=True)

    np.random.seed(42)
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    started = time.perf_counter()
    cnt_done = 0

    rows = []  # per-geometry ZPEs
    delta_rows = []  # per-reaction ΔZPE and ΔΔZPE

    for idx, entry in enumerate(dataset):
        if cnt_done >= args.max_samples:
            break

        reactant = entry["reactant"]
        product = entry["product"]
        rxn = entry.get("rxn", idx)

        print("=" * 80)
        print(f"Sample {idx}: rxn={rxn}")
        print("=" * 80)

        def _prep(mol):
            coords_ang = np.asarray(mol["positions"], dtype=float)
            atomic_numbers = np.asarray(mol["atomic_numbers"], dtype=int)
            atomssymbols = [Z_TO_SYMBOL.get(int(z), "X") for z in atomic_numbers]
            return coords_ang, atomic_numbers, atomssymbols

        r_coords_ang, r_Z, r_syms = _prep(reactant)
        p_coords_ang, p_Z, p_syms = _prep(product)

        # Sanity: same composition/order
        if not (len(r_Z) == len(p_Z) and np.all(r_Z == p_Z)):
            raise ValueError(f"Reactant/Product atom mismatch for reaction {rxn}")

        out_dir_rxn = os.path.join(args.out_dir, f"rxn_{idx:05d}")
        os.makedirs(out_dir_rxn, exist_ok=True)

        # Relax with DFT (reactant)
        r_xyz = os.path.join(xyz_dir, f"reactant_{idx:05d}.xyz")
        if (not os.path.isfile(r_xyz)) or args.redo_relax:
            print(f"# Did not find {r_xyz}, relaxing with DFT")
            r_geom, r_final_ang = relax_with_dft(
                r_syms,
                r_coords_ang,
                thresh=args.thresh,
                max_cycles=args.max_cycles,
                out_dir=os.path.join(out_dir_rxn, "reactant"),
            )
            _write_xyz(r_xyz, r_syms, r_final_ang)
        else:
            print(f"# Found {r_xyz}, skipping relaxation")
            atoms_xyz, coords_xyz = _read_xyz(r_xyz)
            if any(a != b for a, b in zip(atoms_xyz, r_syms)):
                raise ValueError(f"Atom symbols mismatch in {r_xyz}")
            r_final_ang = coords_xyz
            r_geom = Geometry(
                r_syms, (r_final_ang / BOHR2ANG).reshape(-1), coord_type="redund"
            )

        # Relax with DFT (product)
        p_xyz = os.path.join(xyz_dir, f"product_{idx:05d}.xyz")
        if (not os.path.isfile(p_xyz)) or args.redo_relax:
            print(f"# Did not find {p_xyz}, relaxing with DFT")
            p_geom, p_final_ang = relax_with_dft(
                p_syms,
                p_coords_ang,
                thresh=args.thresh,
                max_cycles=args.max_cycles,
                out_dir=os.path.join(out_dir_rxn, "product"),
            )
            _write_xyz(p_xyz, p_syms, p_final_ang)
        else:
            print(f"# Found {p_xyz}, skipping relaxation")
            atoms_xyz, coords_xyz = _read_xyz(p_xyz)
            if any(a != b for a, b in zip(atoms_xyz, p_syms)):
                raise ValueError(f"Atom symbols mismatch in {p_xyz}")
            p_final_ang = coords_xyz
            p_geom = Geometry(
                p_syms, (p_final_ang / BOHR2ANG).reshape(-1), coord_type="redund"
            )

        # DFT Hessians
        r_hess_path = os.path.join(dft_hess_dir, f"reactant_{idx:05d}.hessian_au.npy")
        p_hess_path = os.path.join(dft_hess_dir, f"product_{idx:05d}.hessian_au.npy")

        if (not os.path.isfile(r_hess_path)) or args.redo_dft:
            print(f"# Did not find {r_hess_path}, computing with DFT")
            r_hess_au = compute_dft_hessian_at_geometry(r_Z, r_geom._coords)
            if r_hess_au is None:
                np.save(r_hess_path, np.array([], dtype=np.float64))
                continue
        else:
            print(f"# Found {r_hess_path}, skipping DFT Hessian computation")
            r_hess_au = np.load(r_hess_path)
            if r_hess_au.shape[0] == 0:
                print(f"# {r_hess_path} is empty, skipping")
                continue

        if (not os.path.isfile(p_hess_path)) or args.redo_dft:
            print(f"# Did not find {p_hess_path}, computing with DFT")
            p_hess_au = compute_dft_hessian_at_geometry(p_Z, p_geom._coords)
            if p_hess_au is None:
                np.save(p_hess_path, np.array([], dtype=np.float64))
                continue
            np.save(p_hess_path, p_hess_au)
        else:
            print(f"# Found {p_hess_path}, skipping DFT Hessian computation")
            p_hess_au = np.load(p_hess_path)
            if p_hess_au.shape[0] == 0:
                print(f"# {p_hess_path} is empty, skipping")
                continue

        # ZPE for DFT
        r_zpe_dft = _zpe_from_hessian_au(r_hess_au, r_geom._coords, r_syms)
        p_zpe_dft = _zpe_from_hessian_au(p_hess_au, p_geom._coords, p_syms)

        rows.extend(
            [
                {
                    "idx": int(idx),
                    "rxn": rxn,
                    "geometry": "reactant",
                    "model": "DFT",
                    "method": "DFT",
                    "zpe_eV": r_zpe_dft,
                },
                {
                    "idx": int(idx),
                    "rxn": rxn,
                    "geometry": "product",
                    "model": "DFT",
                    "method": "DFT",
                    "zpe_eV": p_zpe_dft,
                },
            ]
        )

        # Model Hessians at DFT-relaxed geometries (Angstrom input)
        def _model_hess_all(coords_ang, Z):
            return {
                ("AlphaNet", "autograd"): _hessian_autograd_with_model(
                    model_alpha, coords_ang.copy(), Z.copy(), device, True
                ),
                ("LeftNet", "autograd"): _hessian_autograd_with_model(
                    model_left, coords_ang.copy(), Z.copy(), device, True
                ),
                ("LeftNet-DF", "autograd"): _hessian_autograd_with_model(
                    model_left_df, coords_ang.copy(), Z.copy(), device, True
                ),
                ("EquiformerV2", "autograd"): _hessian_autograd_with_model(
                    model_eqv2_autograd, coords_ang.copy(), Z.copy(), device, True
                ),
                ("EquiformerV2", "predict"): _hessian_autograd_with_model(
                    model_eqv2_predict, coords_ang.copy(), Z.copy(), device, False
                ),
            }

        r_hess_ev_all = _model_hess_all(r_final_ang, r_Z)
        p_hess_ev_all = _model_hess_all(p_final_ang, p_Z)

        # ZPE for models
        for (model_name, method_kind), r_hess_ev in r_hess_ev_all.items():
            p_hess_ev = p_hess_ev_all[(model_name, method_kind)]
            r_hess_au_m = _to_au(r_hess_ev)
            p_hess_au_m = _to_au(p_hess_ev)
            r_zpe_m = _zpe_from_hessian_au(r_hess_au_m, r_geom._coords, r_syms)
            p_zpe_m = _zpe_from_hessian_au(p_hess_au_m, p_geom._coords, p_syms)

            rows.extend(
                [
                    {
                        "idx": int(idx),
                        "rxn": rxn,
                        "geometry": "reactant",
                        "model": model_name,
                        "method": method_kind,
                        "zpe_eV": r_zpe_m,
                    },
                    {
                        "idx": int(idx),
                        "rxn": rxn,
                        "geometry": "product",
                        "model": model_name,
                        "method": method_kind,
                        "zpe_eV": p_zpe_m,
                    },
                ]
            )

        # Per-reaction deltas
        delta_dft = p_zpe_dft - r_zpe_dft
        # Add DFT row for completeness
        delta_rows.append(
            {
                "idx": int(idx),
                "rxn": rxn,
                "model": "DFT",
                "method": "DFT",
                "delta_zpe_eV": float(delta_dft),
                "delta_zpe_dft_eV": float(delta_dft),
                "delta_delta_zpe_eV": 0.0,
            }
        )

        for (model_name, method_kind), r_hess_ev in r_hess_ev_all.items():
            # Retrieve ZPEs already computed above (could recompute or capture from loop)
            r_zpe_m = next(
                x["zpe_eV"]
                for x in rows
                if x["idx"] == idx
                and x["rxn"] == rxn
                and x["geometry"] == "reactant"
                and x["model"] == model_name
                and x["method"] == method_kind
            )
            p_zpe_m = next(
                x["zpe_eV"]
                for x in rows
                if x["idx"] == idx
                and x["rxn"] == rxn
                and x["geometry"] == "product"
                and x["model"] == model_name
                and x["method"] == method_kind
            )
            delta_m = p_zpe_m - r_zpe_m
            delta_rows.append(
                {
                    "idx": int(idx),
                    "rxn": rxn,
                    "model": model_name,
                    "method": method_kind,
                    "delta_zpe_eV": float(delta_m),
                    "delta_zpe_dft_eV": float(delta_dft),
                    "delta_delta_zpe_eV": float(abs(delta_m - delta_dft)),
                }
            )

        cnt_done += 1

    # Save outputs
    df_zpe = pd.DataFrame(rows)
    df_zpe.to_csv(zpe_csv, index=False)
    print(f"Saved per-geometry ZPEs to {zpe_csv}")

    df_delta = pd.DataFrame(delta_rows)
    df_delta.to_csv(delta_csv, index=False)
    print(f"Saved per-reaction ΔZPE and ΔΔZPE to {delta_csv}")

    # Print ΔΔZPE summary table
    print_deltadelta_latex(delta_csv)

    elapsed = time.perf_counter() - started
    print(f"Processed {cnt_done} reactions in {elapsed:.2f}s. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
