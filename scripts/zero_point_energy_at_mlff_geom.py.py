import argparse
import os
import time
import pathlib
import numpy as np
import pandas as pd
import torch

from pysisyphus.Geometry import Geometry
from pysisyphus.optimizers.RFOptimizer import RFOptimizer
from pysisyphus.constants import BOHR2ANG, AU2EV

# ReactBench MLFF wrapper
from pysisyphus.calculators.MLFF import MLFF
from ReactBench.Calculators import AVAILABLE_CALCULATORS

from hip.t1x_dft_dataloader import Dataloader as T1xDFTDataloader
from hip.path_config import ROOT_DIR

# Frequency/ZPE utilities
from hip.frequency_analysis import analyze_frequencies, eckart_projection_notmw  # noqa: F401
from hip.frequency_analysis import eigval_to_wavenumber

# Model modules used to compute Hessians
from hip.training_module import PotentialModule
from ocpmodels.common.relaxation.ase_utils import (
    coord_atoms_to_torch_geometric_hessian,
)
from nets.prediction_utils import compute_extra_props

# DFT (PySCF)
from pyscf import dft, gto
import scipy.constants as spc


def _write_xyz(path, atomssymbols, coords_ang):
    n = len(atomssymbols)
    lines = [str(n), "relaxed geometry (MLFF)"]
    for el, (x, y, z) in zip(atomssymbols, coords_ang):
        lines.append(f"{el} {x:.8f} {y:.8f} {z:.8f}")
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def _read_xyz(path):
    lines = pathlib.Path(path).read_text().strip().splitlines()
    n = int(lines[0].strip())
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


def _compute_hessian(coords, energy, forces=None):
    if forces is None:
        forces = -_get_derivatives(coords, energy, create_graph=True)
    n_comp = forces.reshape(-1).shape[0]
    hess = []
    for f in forces.reshape(-1):
        hess_row = _get_derivatives(coords, -f, retain_graph=True)
        hess.append(hess_row)
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


def _hessian_with_model(model, coords_ang, atomic_numbers, device, do_autograd=True):
    use_pbc = getattr(model, "use_pbc", False)
    batch = coord_atoms_to_torch_geometric_hessian(
        coords=coords_ang,
        atomic_nums=atomic_numbers,
        cutoff=getattr(model, "cutoff", 12.0),
        max_neighbors=getattr(model, "max_neighbors", None),
        use_pbc=use_pbc,
        with_grad=True,
        cutoff_hessian=getattr(model, "cutoff_hessian", 100.0),
    )
    batch.ae = torch.tensor(np.array([len(atomic_numbers)]), dtype=torch.int64)
    batch = batch.to(device)
    batch = compute_extra_props(batch, pos_require_grad=True)

    model_name = model.__class__.__name__.lower()
    n_atoms = batch.pos.shape[0]
    if "leftnet" in model_name:
        batch.pos.requires_grad_()
        energy_model, force_model = model.forward_autograd(batch)
        hessian_model = _compute_hessian(batch.pos, energy_model, force_model)
    elif "equiformer" in model_name:
        if do_autograd:
            batch.pos.requires_grad_()
            energy_model, force_model, _ = model.forward(
                batch, otf_graph=False, hessian=False
            )
            hessian_model = _compute_hessian(batch.pos, energy_model, force_model)
        else:
            with torch.no_grad():
                energy_model, force_model, out = model.forward(
                    batch, otf_graph=False, hessian=True, add_props=True
                )
            hessian_model = out["hessian"].reshape(n_atoms * 3, n_atoms * 3)
    else:
        batch.pos.requires_grad_()
        energy_model, force_model = model.forward(batch)
        hessian_model = _compute_hessian(batch.pos, energy_model, force_model)

    hessian_model = hessian_model.reshape(n_atoms * 3, n_atoms * 3)
    return hessian_model.detach().cpu().numpy()


def _zpe_from_hessian_au(h_au, coords_bohr, atomsymbols):
    proj = eckart_projection_notmw(h_au, coords_bohr.reshape(-1), atomsymbols)
    eigvals, _ = np.linalg.eigh(proj)
    eigvals = eigvals[eigvals > 0]
    wavenumbers_cm = eigval_to_wavenumber(eigvals)
    zpe_J = 0.5 * spc.h * spc.c * np.sum(wavenumbers_cm * 100.0)
    return zpe_J / spc.e


def relax_t1x_with_mlff(
    dataset_path,
    out_dir,
    *,
    coord="redund",
    max_samples=50,
    max_cycles=200,
    thresh="gau",
    device="cpu",
    redo=False,
):
    os.makedirs(out_dir, exist_ok=True)

    # Hard-coded relax models
    model_names = ["equiformer", "leftnet", "leftnet-d"]

    # Hard-coded checkpoints
    ckpt_left = "ckpt/left.ckpt"
    ckpt_left_df = "ckpt/left-df.ckpt"
    ckpt_eqv2_autograd = "ckpt/eqv2.ckpt"
    ckpt_eqv2_predict = "ckpt/eqv2.ckpt"

    # Load dataset (validation reactants)
    dataset = T1xDFTDataloader(dataset_path, datasplit="val", only_final=True)

    # Load models once
    def _load_model(ckpt):
        if ckpt is None:
            raise ValueError("Checkpoint path must be provided")
        m = PotentialModule.load_from_checkpoint(ckpt, strict=False).potential
        return m.to(device).eval()

    model_left = _load_model(ckpt_left)
    model_left_df = _load_model(ckpt_left_df)
    model_eqv2_predict = _load_model(ckpt_eqv2_predict)
    model_eqv2_autograd = _load_model(ckpt_eqv2_autograd)

    started = time.perf_counter()
    cnt_done = 0
    zpe_rows = []

    for idx, molecule in enumerate(dataset):
        if cnt_done >= max_samples:
            break
        reactant = molecule["reactant"]
        coords_ang = np.asarray(reactant["positions"], dtype=float)
        atomic_numbers = np.asarray(reactant["atomic_numbers"], dtype=int)
        atomssymbols = [
            {1: "H", 6: "C", 7: "N", 8: "O"}.get(int(z), "X") for z in atomic_numbers
        ]

        print("=" * 80)
        print(f"Sample {idx}: relaxing with models {model_names}")
        print("=" * 80)

        for model_name in model_names:
            model_dir = os.path.join(out_dir, model_name)
            xyz_dir = os.path.join(model_dir, "relaxed_xyz")
            dft_dir = os.path.join(model_dir, "dft")
            dft_grad_dir = os.path.join(dft_dir, "gradients")
            dft_hess_dir = os.path.join(dft_dir, "hessians")
            out_dir_sample = os.path.join(model_dir, f"sample_{idx:05d}")
            os.makedirs(xyz_dir, exist_ok=True)
            os.makedirs(dft_grad_dir, exist_ok=True)
            os.makedirs(dft_hess_dir, exist_ok=True)
            os.makedirs(out_dir_sample, exist_ok=True)
            xyz_path = os.path.join(xyz_dir, f"reactant_{idx:05d}.xyz")

            if os.path.isfile(xyz_path) and not redo:
                print(f"[skip] {model_name} sample {idx} already relaxed")
                # If already relaxed, we still may compute ZPEs if absent
                atoms_xyz, coords_xyz = _read_xyz(xyz_path)
                final_coords_ang = coords_xyz
                coords_bohr = final_coords_ang / BOHR2ANG
                geom = Geometry(atomssymbols, coords_bohr.reshape(-1), coord_type=coord)
                calc = MLFF(method=model_name, device=device)
                geom.set_calculator(calc)
            else:
                # Convert to Bohr for pysisyphus
                coords_bohr = coords_ang / BOHR2ANG

                # Build geometry + MLFF calculator
                geom = Geometry(atomssymbols, coords_bohr, coord_type=coord)
                calc = MLFF(method=model_name, device=device)
                geom.set_calculator(calc)

                # RFOptimizer with unit Hessian init and BFGS updates
                opt = RFOptimizer(
                    geom,
                    thresh=thresh,
                    trust_radius=0.3,
                    hessian_init="unit",
                    hessian_update="bfgs",
                    hessian_recalc=None,
                    line_search=True,
                    out_dir=out_dir_sample,
                    max_cycles=max_cycles,
                )
                opt.run()

                # Save relaxed structure as XYZ (Angstrom)
                final_coords_ang = (geom._coords).reshape(-1, 3) * BOHR2ANG
                _write_xyz(xyz_path, atomssymbols, final_coords_ang)
                print(f"[{model_name}] saved {xyz_path}")

            # ---- DFT reference at relaxed geometry ----
            hessian_path = os.path.join(
                dft_hess_dir, f"reactant_{idx:05d}.hessian_eV_A2.npy"
            )
            force_path = os.path.join(
                dft_grad_dir, f"reactant_{idx:05d}.forces_eV_A.npy"
            )
            if (
                (not os.path.isfile(hessian_path))
                or (not os.path.isfile(force_path))
                or redo
            ):
                atoms_bohr = [
                    (int(Z), (float(x), float(y), float(z)))
                    for Z, (x, y, z) in zip(atomic_numbers, geom._coords.reshape(-1, 3))
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
                    raise RuntimeError(f"PySCF SCF did not converge for sample {idx}")

                grad_au_bohr = mf.nuc_grad_method().kernel()
                forces_au_bohr = -grad_au_bohr
                forces_eV_A = forces_au_bohr * (AU2EV / BOHR2ANG)
                hobj = mf.Hessian()
                setattr(hobj, "conv_tol", 1e-10)
                setattr(hobj, "max_cycle", 100)
                hessian_au = hobj.kernel()
                N = mol.natm
                hess_cart_au = hessian_au.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
                hess_ev_ang2 = hess_cart_au * (AU2EV / (BOHR2ANG * BOHR2ANG))

                np.save(hessian_path, hess_ev_ang2)
                np.save(force_path, forces_eV_A)
            else:
                hess_ev_ang2 = np.load(hessian_path)

            # ---- Model Hessians at relaxed geometry ----
            def _to_au(h_ev_ang2):
                return h_ev_ang2 / AU2EV * (BOHR2ANG * BOHR2ANG)

            h_dft_au = hess_ev_ang2 / AU2EV * (BOHR2ANG * BOHR2ANG)

            h_left_ev = _hessian_with_model(
                model_left, final_coords_ang.copy(), atomic_numbers.copy(), device, True
            )
            h_leftdf_ev = _hessian_with_model(
                model_left_df,
                final_coords_ang.copy(),
                atomic_numbers.copy(),
                device,
                True,
            )
            h_eqv2_auto_ev = _hessian_with_model(
                model_eqv2_autograd,
                final_coords_ang.copy(),
                atomic_numbers.copy(),
                device,
                True,
            )
            h_eqv2_pred_ev = _hessian_with_model(
                model_eqv2_predict,
                final_coords_ang.copy(),
                atomic_numbers.copy(),
                device,
                False,
            )

            h_left_au = _to_au(h_left_ev)
            h_leftdf_au = _to_au(h_leftdf_ev)
            h_eqv2_auto_au = _to_au(h_eqv2_auto_ev)
            h_eqv2_pred_au = _to_au(h_eqv2_pred_ev)

            # ---- ZPEs ----
            zpe_rows.extend(
                [
                    {
                        "idx": int(idx),
                        "relax": model_name,
                        "model": "DFT",
                        "method": "DFT",
                        "zpe_eV": _zpe_from_hessian_au(
                            h_dft_au, geom._coords, atomssymbols
                        ),
                    },
                    {
                        "idx": int(idx),
                        "relax": model_name,
                        "model": "LeftNet",
                        "method": "autograd",
                        "zpe_eV": _zpe_from_hessian_au(
                            h_left_au, geom._coords, atomssymbols
                        ),
                    },
                    {
                        "idx": int(idx),
                        "relax": model_name,
                        "model": "LeftNet-DF",
                        "method": "autograd",
                        "zpe_eV": _zpe_from_hessian_au(
                            h_leftdf_au, geom._coords, atomssymbols
                        ),
                    },
                    {
                        "idx": int(idx),
                        "relax": model_name,
                        "model": "EquiformerV2",
                        "method": "autograd",
                        "zpe_eV": _zpe_from_hessian_au(
                            h_eqv2_auto_au, geom._coords, atomssymbols
                        ),
                    },
                    {
                        "idx": int(idx),
                        "relax": model_name,
                        "model": "EquiformerV2",
                        "method": "predict",
                        "zpe_eV": _zpe_from_hessian_au(
                            h_eqv2_pred_au, geom._coords, atomssymbols
                        ),
                    },
                ]
            )

        cnt_done += 1

    elapsed = time.perf_counter() - started
    print(f"Processed {cnt_done} geometries in {elapsed:.2f}s. Output root: {out_dir}")

    # ---- Persist ZPEs and print error tables vs DFT per relax method ----
    if len(zpe_rows) > 0:
        zpe_dir = os.path.join(out_dir, "zpe")
        os.makedirs(zpe_dir, exist_ok=True)
        df = pd.DataFrame(zpe_rows)
        csv_path = os.path.join(zpe_dir, "zpe_all_methods.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved ZPEs to {csv_path}")

        # Print simple LaTeX tables of errors vs DFT per relax method
        for relax_name in sorted(df["relax"].unique()):
            sub = df[df["relax"] == relax_name].copy()
            ref = sub[sub["method"] == "DFT"][["idx", "zpe_eV"]].rename(
                columns={"zpe_eV": "zpe_ref"}
            )
            if len(ref) == 0:
                continue
            merged = sub.merge(ref, on="idx", how="inner")
            merged = merged[merged["method"] != "DFT"].copy()
            merged["error"] = merged["zpe_eV"] - merged["zpe_ref"]
            merged["abs_error"] = merged["error"].abs()

            row_specs = [
                ("autograd", "LeftNet", "autograd"),
                ("autograd", "LeftNet-DF", "autograd"),
                ("autograd", "EquiformerV2", "autograd"),
                ("predict", "EquiformerV2", "predict"),
            ]
            rows = []
            for hess_kind, model_name, method_value in row_specs:
                subm = merged[
                    (merged["method"] == method_value) & (merged["model"] == model_name)
                ]
                if len(subm) == 0:
                    zpe_err_str = "-"
                else:
                    err = subm["error"].to_numpy()
                    ae = np.abs(err)
                    mae = float(ae.mean()) if len(ae) > 0 else np.nan
                    std = float(err.std(ddof=0)) if len(err) > 0 else np.nan
                    zpe_err_str = (
                        "-"
                        if (np.isnan(mae) or np.isnan(std))
                        else f"{mae:.3f} ({std:.3f})"
                    )
                rows.append(
                    {
                        "Hessian (autograd/predict)": hess_kind,
                        "Model": model_name,
                        "ZPE error": zpe_err_str,
                    }
                )
            table = pd.DataFrame(
                rows, columns=["Hessian (autograd/predict)", "Model", "ZPE error"]
            )
            latex = table.to_latex(index=False, escape=True)
            print(f"\nRelax={relax_name} ZPE error vs DFT (LaTeX):\n{latex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="data/t1x_val_reactant_hessian_100.h5",
        help="Path to T1x validation HDF5 file (reactant set)",
    )
    ap.add_argument(
        "--coord",
        type=str,
        default="redund",
        choices=["cart", "redund", "dlc", "tric"],
        help="Coordinate system",
    )
    ap.add_argument("--max_samples", type=int, default=10)
    ap.add_argument("--max_cycles", type=int, default=200)
    ap.add_argument("--thresh", type=str, default="gau")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--redo", type=bool, default=False)
    # Hard-coded checkpoints; no CLI flags for ckpts/models
    args = ap.parse_args()

    if args.out_dir is None:
        source_label = os.path.splitext(os.path.basename(args.dataset))[0]
        args.out_dir = os.path.join(
            ROOT_DIR,
            "runs_relax_mlff",
            f"{source_label}_{args.coord}_{args.thresh.replace('_', '')}_{args.max_samples}",
        )

    relax_t1x_with_mlff(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        coord=args.coord,
        max_samples=args.max_samples,
        max_cycles=args.max_cycles,
        thresh=args.thresh,
        device=args.device,
        redo=args.redo,
    )


if __name__ == "__main__":
    main()
