"""
Search for transition states (index-one saddle points) using Sella and predicted Hessians
- test different starting points:
        - starting from reactant
        - starting from geodesic interpolation
- verify found transition state using:
        - frequency analysis with mass weighting and Eckart projection
        - RMSD of found to true TS is under some threshold
"""

import numpy as np
import os
import random
import argparse
import json
import sys
import time
from datetime import datetime
import traceback
import pandas as pd
import torch
# from torch_geometric.data import Batch
# from torch_geometric.data import Data as TGData

# from rdkit import Chem
# from rdkit.Chem.Draw import rdMolDraw2D
# from rdkit.Chem.rdchem import Mol
# from rdkit.Chem.rdMolAlign import AlignMol
# from rdkit.Chem.rdMolAlign import GetBestRMS

from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path

# from hip.align_unordered_mols import rmsd
from hip.align_ordered_mols import align_ordered_and_get_rmsd
from hip.plot_molecules import (
    plot_molecule_mpl,
    plot_traj_mpl,
    clean_filename,
    save_trajectory_to_xyz,
)
from hip.trajectorysaver import MyTrajectory
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.equiformer_ase_calculator import EquiformerASECalculator
from hip.frequency_analysis import analyze_frequencies
from hip.geodesic_interpolate import geodesic_interpolate_wrapper

from ase import Atoms
from sella import Sella

# https://github.com/virtualzx-nad/geodesic-interpolate

###########################################################################
# Helper functions
###########################################################################


def copy_atoms(atoms: Atoms) -> Atoms:
    """
    Simple function to copy an atoms object to prevent mutability.
    """
    # try:
    #     atoms = copy.deepcopy(atoms)
    # except Exception:
    #     # Needed because of ASE issue #1084
    calc = atoms.calc
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def to_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.tensor(x).to(device)


def clean_dict(data):
    """
    Recursively clean a dictionary to contain only JSON-serializable types.

    - Keeps: int, float, str, bool, None
    - Converts single-element numpy arrays and torch tensors to scalars
    - Discards multi-element arrays/tensors and other non-serializable types
    - Recursively processes nested dictionaries and lists

    Args:
        data: Input data structure to clean

    Returns:
        Cleaned data structure with only serializable types
    """
    if data is None:
        return None
    elif isinstance(data, (int, float, str, bool)):
        return data
    elif isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            cleaned_value = clean_dict(value)
            if cleaned_value is not None or value is None:
                cleaned[str(key)] = cleaned_value
        return cleaned
    elif isinstance(data, (list, tuple)):
        cleaned = []
        for item in data:
            cleaned_item = clean_dict(item)
            if cleaned_item is not None or item is None:
                cleaned.append(cleaned_item)
        return cleaned
    elif torch.is_tensor(data):
        # Convert single-element tensors to scalars, discard multi-element
        if data.numel() == 1:
            return data.item()
        else:
            return None
    elif isinstance(data, np.ndarray):
        # Convert single-element arrays to scalars, discard multi-element
        if data.size == 1:
            return data.item()
        else:
            return None
    elif hasattr(data, "item") and hasattr(data, "size"):
        # Handle other array-like objects with single elements
        try:
            if data.size == 1:
                return data.item()
        except Exception:
            pass
        return None
    elif isinstance(data, type):
        # Return class name for classes
        return data.__class__.__name__
    elif hasattr(data, "__name__"):
        # Return function name for functions
        return data.__name__
    # elif callable(data):
    #     return
    else:
        # Discard other non-serializable types
        return None


def get_hessian_function(hessian_method, asecalc):
    """Return a callable that Sella uses to fetch the Cartesian Hessian (N*3, N*3).

    Uses the provided ASE calculator to compute either autodiff or predicted Hessian.
    """
    if hessian_method is None:
        return None

    if hessian_method not in {"autodiff", "predict"}:
        raise ValueError(f"Invalid method: {hessian_method}")

    def _hessian_function(atoms):
        # Map naming mismatch between callers and ASE wrapper
        hm = "autograd" if (hessian_method == "autodiff") else "predict"
        hessian = asecalc.get_hessian(atoms, hessian_method=hm)
        if isinstance(hessian, dict):
            hessian = hessian["hessian"]
        return hessian  # already shaped (N*3, N*3)

    return _hessian_function


###########################################################################
# Sella-specific functions
###########################################################################


sella_default_kwargs = dict(
    minimum=dict(
        delta0=1e-1,
        sigma_inc=1.15,
        sigma_dec=0.90,
        rho_inc=1.035,
        rho_dec=100,
        method="rfo",
        eig=False,
    ),
    saddle=dict(
        delta0=0.1,
        sigma_inc=1.15,
        sigma_dec=0.65,
        rho_inc=1.035,
        rho_dec=5.0,
        method="prfo",
        eig=True,
    ),
)


def run_sella_on_sample(
    start_pos,
    z,
    natoms,
    true_pos,
    title,
    sella_kwargs={},
    run_kwargs={},
    asecalc=None,
    hessian_method=None,
    rmsd_threshold: float = 0.5,
    plot_dir=None,
    plot_traj=False,
    plot_final_mol=False,
):
    default_run_kwargs = dict(
        fmax=1e-2,  # 0.05
        steps=1000,
    )
    if hessian_method is not None:
        diag_every_n = 1
    default_sella_kwargs = dict(
        order=1,  # 1 = first-order saddle point, 0 = minimum
        # eta=5e-5,  # Smaller finite difference step for higher accuracy
        # delta0=5e-3,  # Larger initial trust radius for TS search
        # gamma=0.1,  # Much tighter convergence for iterative diagonalization
        # rho_inc=1.05,  # More conservative trust radius adjustment
        # rho_dec=3.0,  # Allow larger trust radius changes
        # sigma_inc=1.3,  # Larger trust radius increases
        # sigma_dec=0.5,  # More aggressive trust radius decreases
        log_every_n=100,
        diag_every_n=diag_every_n,  # brute force. paper set this to 1. try 0
        # nsteps_per_diag=3,  # adaptive
        # internal=True,
    )
    default_sella_kwargs.update(sella_kwargs)
    sella_kwargs = default_sella_kwargs
    default_run_kwargs.update(run_kwargs)
    run_kwargs = default_run_kwargs

    title += f" | Hessian={hessian_method}"
    if sella_kwargs["internal"]:
        title += " | Internal"
    else:
        title += " | Cartesian"
    if sella_kwargs["diag_every_n"] is not None:
        title += f" | diag_every_n={sella_kwargs['diag_every_n']}"
    # if nsteps_per_diag != 3:
    #     title += f" | nsteps_per_diag={nsteps_per_diag}"
    print("\nRunning Sella TS search:\n", title)

    if plot_dir is None:
        # Auto-detect plot directory based on main script
        main_script = sys.argv[0]
        main_dir = os.path.dirname(os.path.abspath(main_script))
        plot_dir = os.path.join(main_dir, "plots_sella")
        print(f"Sella autodetected plot directory: {plot_dir}")

    # Create ASE Atoms object from the initial guess position
    start_pos = to_numpy(start_pos)
    atomic_numbers_np = to_numpy(z)
    mol_ase = Atoms(numbers=atomic_numbers_np, positions=start_pos)
    mol_ase.calc = asecalc
    calcname = asecalc.__class__.__name__

    summary = {
        "calcname": calcname,
        "natoms": natoms,
        "hessian_method": hessian_method,
        "internal": sella_kwargs.get("internal", False),
    }

    # Measure RMSD between start_pos and true_pos
    if true_pos is not None:
        true_pos = to_numpy(true_pos)
        rmsd_initial = align_ordered_and_get_rmsd(start_pos, true_pos)
        print(f"RMSD start: {rmsd_initial:.6f} Å")
        summary["rmsd_initial"] = rmsd_initial

    # Set up a Sella Dynamics object with improved parameters for TS search
    dyn = Sella(
        atoms=mol_ase,
        # constraints=cons,
        # trajectory=os.path.join(logfolder, "cu_sella_ts.traj"),
        trajectory=MyTrajectory(atoms=mol_ase),
        hessian_function=get_hessian_function(hessian_method, asecalc),
        **sella_kwargs,
    )

    summary.update(
        {
            "sella_kwargs": sella_kwargs,
            "run_kwargs": run_kwargs,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    # Run the optimization
    t1 = time.time()
    try:
        dyn.run(**run_kwargs)
    except Exception as e:
        error_traceback = traceback.print_exc()
        print(f"Error in Sella optimization: {e}")
        print(f"Sella kwargs: {sella_kwargs}")
        print(f"Run kwargs: {run_kwargs}")
        print("Full stack trace:")
        print(error_traceback)
        summary["error"] = str(e)
        summary["error_traceback"] = error_traceback
        return summary
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")
    dyn.print_total_time_per_function()
    # write timing to summary dict
    summary["time_stats"] = dyn._time_stats
    summary["time_stats_pes"] = dyn.pes._time_stats
    print("Sella optimization completed!")

    # Get the final structure, convert to np, compute RMSD to true transition state
    # final_positions_sella = mol_ase.get_positions()
    # traj = dyn.trajectory.trajectory # ASE
    traj = dyn.pes.traj.trajectory  # Sella PES trajectory

    summary.update(
        {
            "trajectory": traj,
            "nsteps": dyn.nsteps,
            "time_taken": t2 - t1,
        }
    )

    """Computes RMSD between predicted and true transition state, and plots trajectory"""
    trajectory = summary.get("trajectory", None)
    rmsd_initial = summary.get("rmsd_initial", float("inf"))
    calcname = summary.get("calcname", "")
    if (trajectory is not None) and (true_pos is not None):
        pred_pos = trajectory[-1]
        # Compute RMSD between Sella-optimized structure and true transition state
        rmsd_final = align_ordered_and_get_rmsd(pred_pos, true_pos)
        print(f"RMSD end: {rmsd_final:.6f} Å")
        print(f"Improvement: {rmsd_initial - rmsd_final:.6f} Å")
        summary["rmsd_final"] = rmsd_final
        summary["rmsd_improvement"] = rmsd_initial - rmsd_final

    # Frequency analysis at final geometry using predicted/autodiff Hessian
    final_atoms = Atoms(numbers=to_numpy(z), positions=to_numpy(pred_pos))
    final_atoms.calc = asecalc
    # Get Hessian from calculator for verification
    hm = hessian_method if hessian_method == "autograd" else "predict"
    hess_np = asecalc.get_hessian(final_atoms, hessian_method=hm)
    # Analyze with Eckart projection and mass weighting (torch routine)
    freq = analyze_frequencies(
        hessian=hess_np,
        cart_coords=final_atoms.get_positions(),
        atomsymbols=final_atoms.get_chemical_symbols(),
        # atomsymbols=[int(zi) for zi in to_numpy(z)],
    )
    neg_num = (
        int(freq["neg_num"])
        if hasattr(freq["neg_num"], "item")
        else int(freq["neg_num"])
    )
    summary.update(
        {
            "freq_neg_num": neg_num,
            "is_index_one_saddle": (neg_num == 1),
            "rmsd_within_threshold": bool(rmsd_final <= rmsd_threshold),
            "rmsd_threshold": rmsd_threshold,
        }
    )
    print(f"Frequency analysis: neg modes = {neg_num}")

    _this_plot_dir = os.path.join(plot_dir, clean_filename(title, None, None))
    os.makedirs(_this_plot_dir, exist_ok=True)

    # plot trajectory
    if len(trajectory) > 0:
        if plot_traj:
            plot_traj_mpl(
                coords_traj=trajectory,
                title=title,
                plot_dir=_this_plot_dir,
                atomic_numbers=z,
                save=True,
            )
        if plot_final_mol:
            plot_molecule_mpl(
                mol_ase.get_positions(),
                atomic_numbers=z,
                title=f"Sella {calcname} {title}",
                plot_dir=_this_plot_dir,
                save=True,
            )
        save_trajectory_to_xyz(trajectory, z, plotfolder=_this_plot_dir, filename=title)

    # save summary dict
    cleansummary = clean_dict(summary)
    with open(os.path.join(_this_plot_dir, "summary.json"), "w") as f:
        json.dump(cleansummary, f)
    print(f"Saved summary to {os.path.join(_this_plot_dir, 'summary.json')}")
    with open(os.path.join(_this_plot_dir, "summary.txt"), "w") as f:
        f.write(json.dumps(cleansummary, indent=4))
    print(f"Saved summary to {os.path.join(_this_plot_dir, 'summary.txt')}")

    return summary


###########################################################################
# CLI and experiment runner
###########################################################################


def _build_ase_atoms(
    positions: np.ndarray | torch.Tensor, z: np.ndarray | torch.Tensor
):
    pos_np = to_numpy(positions)
    z_np = to_numpy(z)
    atoms = Atoms(numbers=z_np, positions=pos_np)
    return atoms


def _init_calculators(device: torch.device):
    ckpt = "ckpt/hesspred_v1.ckpt"
    torchcalc = EquiformerTorchCalculator(device=device, checkpoint_path=ckpt)
    model = getattr(torchcalc, "model", None)
    if model is None:
        model = getattr(torchcalc, "potential", None)
    asecalc = EquiformerASECalculator(device=device, model=model)
    return torchcalc, asecalc


def launch_sella_tssearch_on_rgd1(
    dataset,
    idx: int = 104_000,
    hessian_method: str = "predict",
    rmsd_threshold: float = 0.5,
    steps: int = 1000,
    plot_traj: bool = True,
    plot_final_mol: bool = False,
    max_samples: int = 10,
    seed: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Dataset size: {len(dataset)}")

    # Calculators
    print("\nInitializing calculators")
    torchcalc, asecalc = _init_calculators(device)

    # Determine indices to run
    max_samples = min(max_samples, len(dataset))
    if max_samples is None or max_samples <= 1:
        indices = [idx]
    else:
        n = len(dataset)
        indices = random.sample(range(n), n)
    # print(f"Selected indices: {indices}")

    # Plot base dir
    this_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(this_dir, "plots_sella")
    os.makedirs(plot_dir, exist_ok=True)

    # Common kwargs
    run_kwargs = {"fmax": 1e-2, "steps": steps}

    out = {}
    # Aggregators per (start_point, coordinate_system)
    # keys: 'reactant_cart', 'reactant_internal', 'geodesic_cart', 'geodesic_internal'
    stats = {}

    def _update_stats(group_key: str, summary: dict):
        if group_key not in stats:
            stats[group_key] = {
                "count": 0,
                "sum_nsteps": 0,
                "count_nsteps": 0,
                "sum_time": 0.0,
                "count_time": 0,
                "freq_success": 0,  # neg modes == 1
                "freq_attempts": 0,
                "rmsd_success": 0,  # rmsd <= threshold
            }
        s = stats[group_key]
        s["count"] += 1
        nsteps = summary.get("nsteps", None)
        if isinstance(nsteps, int):
            s["sum_nsteps"] += nsteps
            s["count_nsteps"] += 1
        time_taken = summary.get("time_taken", None)
        if isinstance(time_taken, (int, float)):
            s["sum_time"] += float(time_taken)
            s["count_time"] += 1
        if "freq_neg_num" in summary:
            s["freq_attempts"] += 1
            if summary.get("freq_neg_num", None) == 1:
                s["freq_success"] += 1
        if summary.get("rmsd_within_threshold", False):
            s["rmsd_success"] += 1

    samples_done = 0
    for ii in indices:
        sample = dataset[ii]

        if samples_done >= max_samples:
            break
        if max_samples > 1 and sample.natoms > 10:
            continue

        print("", "=" * 80, f"Loading sample {ii}", "=" * 80, sep="\n")
        print(f"Number of atoms: {sample.natoms}")
        print(f"Elements (z): {sample.z}")
        print(f"Reactant SMILES: {sample.smiles_reactant}")

        # # Plot structures
        # plot_molecule_mpl(
        #     sample.pos_reactant,
        #     atomic_numbers=sample.z,
        #     title=f"Reactant idx{ii}",
        #     plot_dir=plot_dir,
        #     save=True,
        # )
        # plot_molecule_mpl(
        #     sample.pos_transition,
        #     atomic_numbers=sample.z,
        #     title=f"Transition state idx{ii}",
        #     plot_dir=plot_dir,
        #     save=True,
        # )
        # plot_molecule_mpl(
        #     sample.pos_product,
        #     atomic_numbers=sample.z,
        #     title=f"Product idx{ii}",
        #     plot_dir=plot_dir,
        #     save=True,
        # )

        # Geodesic interpolation midpoint
        reactant_atoms = _build_ase_atoms(sample.pos_reactant, sample.z)
        product_atoms = _build_ase_atoms(sample.pos_product, sample.z)
        x_geointer_atoms = geodesic_interpolate_wrapper(
            reactant=reactant_atoms,
            product=product_atoms,
            n_images=5,
            return_middle_image=True,
        )
        x_geointer_rp = x_geointer_atoms.get_positions()
        # plot_molecule_mpl(
        #     x_geointer_rp,
        #     atomic_numbers=sample.z,
        #     title=f"R-P geodesic interpolation idx{ii}",
        #     plot_dir=plot_dir,
        #     save=True,
        # )

        # Run Sella from two starts
        # for hessian_method in [None, "predict"]:
        for hessian_method in ["predict"]:
            for internal in [False, True]:
                # for internal in [True]:
                group = "internal" if internal else "cart"
                group += hessian_method
                summary_r = run_sella_on_sample(
                    start_pos=sample.pos_reactant,
                    z=sample.z,
                    natoms=sample.natoms,
                    true_pos=sample.pos_transition,
                    title=f"Sella TS from R idx{ii}",
                    asecalc=asecalc,
                    hessian_method=hessian_method,
                    rmsd_threshold=rmsd_threshold,
                    run_kwargs=run_kwargs,
                    plot_traj=plot_traj,
                    plot_final_mol=plot_final_mol,
                    sella_kwargs=dict(
                        internal=internal,
                    ),
                )
                _update_stats(group_key=f"reactant" + group, summary=summary_r)

                summary_geo = run_sella_on_sample(
                    start_pos=x_geointer_rp,
                    z=sample.z,
                    natoms=sample.natoms,
                    true_pos=sample.pos_transition,
                    title=f"Sella TS from geodesic R-P idx{ii}",
                    asecalc=asecalc,
                    hessian_method=hessian_method,
                    rmsd_threshold=rmsd_threshold,
                    run_kwargs=run_kwargs,
                    plot_traj=plot_traj,
                    plot_final_mol=plot_final_mol,
                    sella_kwargs=dict(
                        internal=internal,
                    ),
                )
                _update_stats(group_key=f"geodesic" + group, summary=summary_geo)

        out[ii] = {"reactant": summary_r, "geodesic": summary_geo}
        samples_done += 1

    out["__stats__"] = stats
    return out


if __name__ == "__main__":
    """
    # --max_samples 10 --seed 0 --hessian-method predict
    uv run /ssd/Code/hip/playground/run_tssearch_rgd1_sella.py 
    """
    parser = argparse.ArgumentParser(
        description="Run Sella TS search with predicted Hessians."
    )
    parser.add_argument(
        "--idx", type=int, default=104_000, help="Sample index in RGD1 dataset"
    )
    parser.add_argument(
        "--max_samples", type=int, default=10, help="Randomly sample this many indices"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to the LMDB dataset file.",
        default="rgd1",  # "ts1x-val.lmdb",
    )
    args = parser.parse_args()

    if args.dataset == "rgd1":
        print("Loading RGD1 dataset")
        dataset = LmdbDataset("data/rgd1/rgd1_minimal_train.lmdb")
        outdir = "results_sella/rgd1"
    else:
        # transform = HessianGraphTransform(
        #     cutoff=model.cutoff,
        #     cutoff_hessian=model.cutoff_hessian,
        #     max_neighbors=model.max_neighbors,
        #     use_pbc=model.use_pbc,
        # )
        outdir = f"results_sella/" + args.dataset.split("/")[-1].split(".")[0]
        dataset = LmdbDataset(fix_dataset_path(args.dataset), transform=None)
    os.makedirs(outdir, exist_ok=True)

    results = launch_sella_tssearch_on_rgd1(
        dataset=dataset,
        idx=args.idx,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    # save as df
    df = pd.DataFrame(results)
    df.to_csv(
        os.path.join(outdir, f"results_sella_{args.max_samples}.csv"), index=False
    )

    print("\nResults:")
    for ii, res in results.items():
        if ii == "__stats__":
            continue
        print(f"idx {ii}:")
        for start, v in res.items():
            rmsd = v.get("rmsd_final", None)
            neg = v.get("freq_neg_num", None)
            ok = v.get("is_index_one_saddle", None)
            print(f"  {start}: rmsd={rmsd}, neg_modes={neg}, is_ts={ok}")

    # Summary statistics
    stats = results.get("__stats__", {})
    if stats:
        print(
            "\nSummary (avg nsteps, avg time [s], freq success rate, rmsd success rate):"
        )
        for key in sorted(stats.keys()):
            s = stats[key]
            avg_nsteps = (
                (s["sum_nsteps"] / s["count_nsteps"]) if s["count_nsteps"] > 0 else None
            )
            avg_time = (
                (s["sum_time"] / s["count_time"]) if s["count_time"] > 0 else None
            )
            freq_rate = (
                (s["freq_success"] / s["freq_attempts"])
                if s["freq_attempts"] > 0
                else None
            )
            rmsd_rate = (s["rmsd_success"] / s["count"]) if s["count"] > 0 else None
            print(
                f"  {key}: avg_nsteps={avg_nsteps}, avg_time={avg_time}, freq_ok={freq_rate}, rmsd_ok={rmsd_rate}"
            )
