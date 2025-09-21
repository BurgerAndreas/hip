"""
GAD-RGD1: Gentlest Ascent Dynamics for Transition State Finding

This script implements Gentlest Ascent Dynamics (GAD) to find transition states
using different eigenvalue calculation methods for the Hessian matrix.

Available eigen methods:
- "qr": QR-based projector method (default)
- "svd": SVD-based projector method
- "inertia": Inertia tensor-based projector with auto-linearity detection
- "geo": Use Geometric library (external dependency)
- "ase": Use ASE library (external dependency)
- "eckart": Eckart frame alignment with principal axes

Example commands:

# Test all eigen methods
python playground/run_tssearch_rgd1.py --eigen-method qr --do-gad
python playground/run_tssearch_rgd1.py --eigen-method svd --do-gad
python playground/run_tssearch_rgd1.py --eigen-method svdforce --do-gad
python playground/run_tssearch_rgd1.py --eigen-method inertia --do-gad
python playground/run_tssearch_rgd1.py --eigen-method geo --do-gad
python playground/run_tssearch_rgd1.py --eigen-method ase --do-gad
python playground/run_tssearch_rgd1.py --eigen-method eckartsvd --do-gad
python playground/run_tssearch_rgd1.py --eigen-method eckartqr --do-gad
"""

import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdMolAlign import GetBestRMS
import io
import base64
from IPython.display import Image
import os
import argparse
import json

from hip.ff_lmdb import LmdbDataset
from hip.equiformer_torch_calculator import EquiformerTorchCalculator

# from hip.align_unordered_mols import rmsd
from hip.align_ordered_mols import align_ordered_and_get_rmsd
from hip.plot_molecules import (
    plot_molecule_mpl,
    plot_traj_mpl,
    clean_filename,
    save_to_xyz,
    save_trajectory_to_xyz,
)
from hip.align_ordered_mols import find_rigid_alignment
from hip.equiformer_ase_calculator import EquiformerASECalculator

from ase import Atoms
from ase.io import read
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.vibrations.data import VibrationsData
from sella import Sella, Constraints
from sella.peswrapper import InternalPES
from sella.internal import Internals

from recipes.ts_search import (
    integrate_dynamics,
    run_sella,
    before_ase_opt,
    after_ase_opt,
    run_irc,
    run_neb,
    run_relaxation,
    run_geodesic_interpolate,
    get_hessian_function,
    copy_atoms,
)
from hip.trajectorysaver import MyTrajectory

from hip.geodesic_interpolate import geodesic_interpolate_wrapper

this_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(this_dir, "plots_gad")
log_dir = os.path.join(this_dir, "logs_gad")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def test_gad_ts_search(sample, torchcalc, asecalc, x_lininter_rp, x_geointer_rp, idx):
    print("\n" + "=" * 60)
    print("Following GAD vector field to find transition state")
    tmp_eigen_dof_method = torchcalc.eigen_dof_method

    for eigen_method in [
        None,
        "qr",
        "svd",
        "svdforce",
        "inertia",
        "geo",
        "ase",
        "eckartsvd",
        "eckartqr",
    ]:
        torchcalc.eigen_dof_method = eigen_method

        results = {}

        # Test 1: is the transition state a fixed point of our GAD vector field?
        # Follow the GAD vector field from transition state
        summary = integrate_dynamics(
            sample.pos_transition,
            sample.z,
            sample.natoms,
            torchcalc,
            sample.pos_transition,
            force_field="gad",
            max_steps=1_000,
            dt=0.01,
            title=f"TS from TS {eigen_method} idx{idx}",
            n_patience_steps=1_000,
            patience_threshold=1.0,
            plot_dir=plot_dir,
            asecalc=asecalc,
        )
        traj = summary["trajectory"]
        _rmsd_ts = align_ordered_and_get_rmsd(
            traj[-1].detach().cpu().numpy(),
            sample.pos_transition.detach().cpu().numpy(),
        )
        results["ts_from_ts"] = _rmsd_ts

        # Follow the GAD vector field from perturbed transition state
        # RMSD ~ 1.2 Å
        _pos = torch.randn_like(sample.pos_transition) + sample.pos_transition
        summary = integrate_dynamics(
            _pos,
            sample.z,
            sample.natoms,
            torchcalc,
            sample.pos_transition,
            force_field="gad",
            max_steps=1000,
            dt=0.01,
            title=f"TS from perturbed TS {eigen_method} idx{idx}",
            n_patience_steps=1000,
            patience_threshold=1.0,
            plot_dir=plot_dir,
            asecalc=asecalc,
        )
        traj = summary["trajectory"]
        _rmsd_ts = align_ordered_and_get_rmsd(
            traj[-1].detach().cpu().numpy(),
            sample.pos_transition.detach().cpu().numpy(),
        )
        results["ts_from_perturbed_ts"] = _rmsd_ts

        # # Test run - start from reactant
        # summary = integrate_dynamics(
        #     sample.pos_reactant,
        #     sample.z,
        #     sample.natoms,
        #     calc,
        #     sample.pos_transition,
        #     force_field="gad",
        #     max_steps=100,
        #     dt=0.1,
        #     title=f"TS from R {eigen_method}",
        #     n_patience_steps=100,
        #     # patience_threshold=1.0,
        #     # center_around_com=True,
        #     plot_dir=plot_dir,
        # )
        # traj = summary["trajectory"]
        # _rmsd_ts = align_ordered_and_get_rmsd(
        #     traj[-1].detach().cpu().numpy(), sample.pos_transition.detach().cpu().numpy()
        # )
        # results["ts_from_r_dt0.1_s100"] = _rmsd_ts

        # Start from reactant
        summary = integrate_dynamics(
            sample.pos_reactant,
            sample.z,
            sample.natoms,
            torchcalc,
            sample.pos_transition,
            force_field="gad",
            max_steps=1_000,
            dt=0.01,
            title=f"TS from R {eigen_method} idx{idx}",
            n_patience_steps=1000,
            # patience_threshold=1.0,
            # center_around_com=True,
            plot_dir=plot_dir,
            asecalc=asecalc,
        )
        results["ts_from_r_dt0.01_s1000"] = summary["rmsd_final"]

        # large steps
        summary = integrate_dynamics(
            sample.pos_reactant,
            sample.z,
            sample.natoms,
            torchcalc,
            sample.pos_transition,
            force_field="gad",
            max_steps=1_000,
            dt=0.1,
            title=f"TS from R {eigen_method} idx{idx}",
            n_patience_steps=1000,
            # patience_threshold=1.0,
            # center_around_com=True,
            plot_dir=plot_dir,
            asecalc=asecalc,
        )
        traj = summary["trajectory"]
        _rmsd_ts = align_ordered_and_get_rmsd(
            traj[-1].detach().cpu().numpy(),
            sample.pos_transition.detach().cpu().numpy(),
        )
        results["ts_from_r_dt0.1_s1000"] = _rmsd_ts

        # very long
        summary = integrate_dynamics(
            sample.pos_reactant,
            sample.z,
            sample.natoms,
            torchcalc,
            sample.pos_transition,
            force_field="gad",
            max_steps=10_000,
            dt=0.01,
            title=f"TS from R (10k steps) {eigen_method} idx{idx}",
            n_patience_steps=10000,
            # patience_threshold=1.0,
            # center_around_com=True,
            plot_dir=plot_dir,
            asecalc=asecalc,
        )
        traj = summary["trajectory"]
        _rmsd_ts = align_ordered_and_get_rmsd(
            traj[-1].detach().cpu().numpy(),
            sample.pos_transition.detach().cpu().numpy(),
        )
        results["ts_from_r_dt0.1_s10000"] = _rmsd_ts

        # Follow the GAD vector field from R-P linear interpolation
        summary = integrate_dynamics(
            x_lininter_rp,
            sample.z,
            sample.natoms,
            torchcalc,
            sample.pos_transition,
            force_field="gad",
            max_steps=2_000,
            dt=0.01,
            title=f"TS from R-P interpolation {eigen_method} idx{idx}",
            n_patience_steps=500,
            # patience_threshold=1.0,
            plot_dir=plot_dir,
            asecalc=asecalc,
        )
        traj = summary["trajectory"]
        _rmsd_ts = align_ordered_and_get_rmsd(
            traj[-1].detach().cpu().numpy(),
            sample.pos_transition.detach().cpu().numpy(),
        )
        results["ts_from_r_p_dt0.01_s2000"] = _rmsd_ts

        # Follow the GAD vector field from R-P geodesic interpolation
        summary = integrate_dynamics(
            initial_pos=x_geointer_rp,
            z=sample.z,
            natoms=sample.natoms,
            torchcalc=torchcalc,
            true_pos=sample.pos_transition,
            force_field="gad",
            max_steps=2_000,
            dt=0.01,
            title=f"TS from R-P geodesic interpolation {eigen_method} idx{idx}",
            n_patience_steps=500,
            # patience_threshold=1.0,
            plot_dir=plot_dir,
            asecalc=asecalc,
        )
        traj = summary["trajectory"]
        _rmsd_ts = align_ordered_and_get_rmsd(
            traj[-1].detach().cpu().numpy(),
            sample.pos_transition.detach().cpu().numpy(),
        )
        results["ts_from_r_p_geo_dt0.01_s2000"] = _rmsd_ts

        # Save results to JSON
        with open(
            os.path.join(log_dir, f"results_{eigen_method}_idx{idx}.json"), "w"
        ) as f:
            json.dump(results, f)

        with open(
            os.path.join(log_dir, f"results_{eigen_method}_idx{idx}.txt"), "w"
        ) as f:
            f.write(f"# eigen_method: {eigen_method}, idx: {idx}\n")
            for k, v in results.items():
                f.write(f"{k}: {v:.6f}\n")

    torchcalc.eigen_dof_method = tmp_eigen_dof_method

    return results


def test_sella_ts_search(
    sample, torchcalc, x_lininter_rp, x_geointer_rp, idx, asecalc=None
):
    print("=" * 60)
    print("Following Sella to find transition state")

    # See if Sella can find the transition state

    # # Test run: Start from reactant, internal coordinates
    # hessian_method = None
    # mol_ase = run_sella(
    #     pos_reactant,
    #     z=sample.z,
    #     natoms=sample.natoms,
    #     true_pos=sample.pos_transition,
    #     title=f"Sella TS from R | Hessian={hessian_method} | Internal",
    #     calc=asecalc,
    #     hessian_function=get_hessian_function(hessian_method, asecalc),
    #     internal=True,
    #     run_kwargs={"steps": 100},
    # )

    # for hessian_method in [None, "autodiff", "predict"]:
    for hessian_method in [None, "autodiff"]:
        # for hessian_method in ["autodiff"]:

        for internal in [False, True]:
            hessian_function = get_hessian_function(hessian_method, asecalc)

            # title needs to be s.t. we can plot it later:
            # sella_ts_from_<starting_point>_hessian_<hessianmethod>_<coordinates>

            # Linear interpolation between reactant and product
            mol_ase = run_sella(
                x_lininter_rp,
                z=sample.z,
                natoms=sample.natoms,
                true_pos=sample.pos_transition,
                title=f"Sella TS from linear R-P idx{idx}",
                calc=asecalc,
                hessian_function=hessian_function,
                hessian_method=hessian_method,
                internal=internal,
            )

            # Geodesic interpolation between reactant and product
            mol_ase = run_sella(
                x_geointer_rp,
                z=sample.z,
                natoms=sample.natoms,
                true_pos=sample.pos_transition,
                title=f"Sella TS from geodesic R-P idx{idx}",
                calc=asecalc,
                hessian_function=hessian_function,
                hessian_method=hessian_method,
                internal=internal,
            )
            # Geodesic interpolation between reactant and product
            mol_ase = run_sella(
                x_geointer_rp,
                z=sample.z,
                natoms=sample.natoms,
                true_pos=sample.pos_transition,
                title=f"Sella TS from geodesic R-P idx{idx}",
                calc=asecalc,
                hessian_function=hessian_function,
                hessian_method=hessian_method,
                internal=internal,
                diag_every_n=0,
            )

            # Start from reactant
            mol_ase = run_sella(
                sample.pos_reactant,
                z=sample.z,
                natoms=sample.natoms,
                true_pos=sample.pos_transition,
                title=f"Sella TS from R idx{idx}",
                calc=asecalc,
                hessian_function=hessian_function,
                hessian_method=hessian_method,
                internal=internal,
            )

            # Start from reactant, diag every 1 step
            mol_ase = run_sella(
                sample.pos_reactant,
                z=sample.z,
                natoms=sample.natoms,
                true_pos=sample.pos_transition,
                title=f"Sella TS from R idx{idx}",
                calc=asecalc,
                hessian_function=hessian_function,
                hessian_method=hessian_method,
                internal=internal,
                diag_every_n=0,
            )
    return


def main(
    # eigen_method="qr",
    idx=104_000,
    do_gad=False,
    do_sella=False,
    do_irc_neb_geodesic=False,
    do_sella_hessian=False,
    do_forces=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(plot_dir, exist_ok=True)

    # Load the LMDB dataset
    print("Loading RGD1 dataset")
    dataset = LmdbDataset("data/rgd1/rgd1_minimal_train.lmdb")
    print(f"Dataset size: {len(dataset)}")

    # Get the sample by index
    print(f"\nLoading sample {idx}")
    sample = dataset[idx]
    print(sample)
    print(f"Sample keys: {sample.keys()}")
    print(f"Number of atoms: {sample.natoms}")
    print(f"Elements (z): {sample.z}")
    print(f"Reactant SMILES: {sample.smiles_reactant}")
    print(f"Product SMILES: {sample.smiles_product}")

    # Initialize equiformer calculator
    print("\n" + "-" * 6)
    print("Initializing EquiformerTorchCalculator")
    torchcalc = EquiformerTorchCalculator(device=device)

    print("\nASE EquiformerTorchCalculator")
    asecalc = EquiformerASECalculator(device=device, model=torchcalc.model)

    # # Example forward pass
    # # Create batch (equiformer expects batch format)
    # tg_data = convert_rgd1_to_tg_format(sample, state="transition")
    # batch = Batch.from_data_list([tg_data])
    # print(f"Batch shape - pos: {batch.pos.shape}, z: {batch.z.shape}")
    # # Run prediction
    # print("\n" + "-" * 6)
    # energy, forces, eigenpred = calc.predict(batch)
    # print(f"Energy: {energy.item():.6f}")
    # print(f"Forces shape: {forces.shape}")
    # print(f"Forces norm: {torch.norm(forces).item():.6f}")
    # print(f"Eigenprediction keys: {list(eigenpred.keys())}")

    # Plot the reactant and transition state
    print("\nPlotting molecular structures")
    plot_molecule_mpl(
        sample.pos_reactant,
        atomic_numbers=sample.z,
        title=f"Reactant idx{idx}",
        plot_dir=plot_dir,
        save=True,
    )
    plot_molecule_mpl(
        sample.pos_transition,
        atomic_numbers=sample.z,
        title=f"Transition state idx{idx}",
        plot_dir=plot_dir,
        save=True,
    )
    plot_molecule_mpl(
        sample.pos_product,
        atomic_numbers=sample.z,
        title=f"Product idx{idx}",
        plot_dir=plot_dir,
        save=True,
    )

    ###################################################################################
    # Use linear interpolation as initial guess (simpler than geodesic for now)
    print("\nCreating initial guess using interpolation")

    # Get reactant and transition state positions
    pos_reactant = sample.pos_reactant
    pos_transition = sample.pos_transition
    pos_product = sample.pos_product

    alpha = 0.5  # midpoint
    # Linear interpolation between reactant and product
    # pos_initial_guess = (1 - alpha) * pos_reactant + alpha * pos_product
    x_lininter_rts = (1 - alpha) * pos_reactant + alpha * pos_transition
    plot_molecule_mpl(
        x_lininter_rts,
        atomic_numbers=sample.z,
        title=f"R-TS linear interpolation idx{idx}",
        plot_dir=plot_dir,
        save=True,
    )

    x_lininter_rp = (1 - alpha) * pos_reactant + alpha * pos_product
    plot_molecule_mpl(
        x_lininter_rp,
        atomic_numbers=sample.z,
        title=f"R-P linear interpolation idx{idx}",
        plot_dir=plot_dir,
        save=True,
    )

    # geodesic interpolation
    geointer_atoms_list = run_geodesic_interpolate(
        pos_reactant, pos_product, z=sample.z, calc=asecalc, return_middle_image=True
    )
    x_geointer_rp = geointer_atoms_list.get_positions()
    plot_molecule_mpl(
        x_geointer_rp,
        atomic_numbers=sample.z,
        title=f"R-P geodesic interpolation idx{idx}",
        plot_dir=plot_dir,
        save=True,
    )
    _rmsd_lin_geo = align_ordered_and_get_rmsd(x_geointer_rp, pos_transition)
    print(f"Geodesic interpolation R-P vs TS RMSD: {_rmsd_lin_geo:.4f}")
    _rmsd_lin_geo = align_ordered_and_get_rmsd(x_geointer_rp, x_lininter_rp)
    print(f"Linear vs geodesic interpolation R-P RMSD: {_rmsd_lin_geo:.4f}")
    x_geointer_rp = torch.tensor(
        x_geointer_rp, device=device, dtype=sample.pos_reactant.dtype
    )

    ###################################################################################
    # Follow the GAD vector field to find the transition state

    if do_gad:
        test_gad_ts_search(
            sample=sample,
            torchcalc=torchcalc,
            x_lininter_rp=x_lininter_rp,
            x_geointer_rp=x_geointer_rp,
            idx=idx,
            asecalc=asecalc,
        )

    ###################################################################################
    # Use Sella internal coordinates for GAD

    # constraints = None
    # internal = Internals(atoms, cons=constraints)
    # pes = InternalPES(
    #     atoms: Atoms,
    #     internals: Internals,
    #     H0: np.ndarray = None,
    #     iterative_stepper: int = 0,
    #     auto_find_internals: bool = True,
    #     # passed to PES
    #     trajectory=trajectory,
    #     eta=eta,
    #     v0=v0,
    # )
    # print("InternalPES deduced internals (pes.int.internals):")
    # for k, v in pes.int.internals.items():
    #     print(f" {k}: {len(v)}")
    # print(f" pes.dim={pes.dim}, pes.ncart={pes.ncart}")

    # # get current internal coordinates
    # x = pes.int.calc()
    # # same as
    # x = pes.get_x()

    # # converts internal to cartesian
    # x = pes.get_cartesian_from_internal(internalcoords)

    # pes.int._convert_cartesian_hessian_to_internal()
    # # _convert_internal_hessian_to_cartesian

    ###################################################################################
    if do_sella:
        test_sella_ts_search(
            sample=sample,
            torchcalc=torchcalc,
            x_lininter_rp=x_lininter_rp,
            x_geointer_rp=x_geointer_rp,
            idx=idx,
            asecalc=asecalc,
        )

    ###################################################################################
    if do_irc_neb_geodesic:
        print("=" * 60)

        # print("\n# ts_job")
        # mol_ase = run_sella(
        #     x_lininter_rts,
        #     z=sample.z,
        #     natoms=sample.natoms,
        #     true_pos=sample.pos_transition,
        #     title="TS from R-TS | Hessian=autodiff",
        #     calc=asecalc,
        #     diag_every_n=1,
        #     # hessian_function=hessian_function,
        #     hessian_function=get_hessian_function(hessian_method="autodiff", asecalc=asecalc),
        #     do_freq=True,
        # )

        print("\n# irc_job")
        # build atoms object
        mol_ase, initsummary = before_ase_opt(
            start_pos=sample.pos_transition,
            z=sample.z,
            true_pos=sample.pos_transition,
            calc=asecalc,
        )
        # run function
        result = run_irc(
            atoms=mol_ase,
            direction="forward",  # forward or reverse
            run_freq=True,
            freq_job_kwargs=None,
            opt_params=None,
            additional_fields=None,
            calc=asecalc,
        )
        result.update(initsummary)
        # eval and plot
        endsummary = after_ase_opt(
            result,
            z=sample.z,
            title=f"Forward IRC QuAcc from R idx{idx}",
            true_pos=sample.pos_transition,
            plot_dir=plot_dir,
        )
        result.update(endsummary)

        print("\n# neb_job")
        atoms_r, _ = before_ase_opt(
            start_pos=sample.pos_reactant,
            z=sample.z,
            calc=asecalc,
        )
        atoms_p, _ = before_ase_opt(
            start_pos=sample.pos_product,
            z=sample.z,
            calc=asecalc,
        )
        result = run_neb(
            reactant_atoms=atoms_r,
            product_atoms=atoms_p,
            interpolation_method="linear",  # "linear", "idpp" and "geodesic"
            relax_job_kwargs=None,
            interpolate_kwargs=None,
            neb_kwargs=None,
        )

        atoms_list = geodesic_interpolate_wrapper(
            reactant=atoms_r,
            product=atoms_p,
            n_images=10,  # MEP guess for NEB
            perform_sweep="auto",
            redistribute_tol=1e-2,
            smoother_tol=2e-3,
            max_iterations=15,
            max_micro_iterations=20,
            morse_scaling=1.7,
            geometry_friction=1e-2,
            distance_cutoff=3.0,
            sweep_cutoff_size=35,
        )

    ###################################################################################
    # Follow the forces to find the reactant minimum
    if do_forces:
        print("=" * 60)
        print("\nFollowing forces to find reactant minimum")

        # start by interpolating true transition state and reactant
        alpha = 0.5
        pos_initial_guess = (1 - alpha) * pos_reactant + alpha * pos_transition

        # Follow the forces to find the reactant minimum
        trajectory_pos, _, _, _ = integrate_dynamics(
            pos_initial_guess,
            sample.z,
            sample.natoms,
            torchcalc,
            sample.pos_reactant,
            force_field="forces",
            dt=0.01,  # time step
            max_steps=100,  # maximum optimization steps
            convergence_threshold=1e-4,  # convergence criterion for forces
            title=f"R-TS interpolation idx{idx}",
            plot_dir=plot_dir,
            asecalc=asecalc,
        )
        plot_molecule_mpl(
            trajectory_pos[-1],
            atomic_numbers=sample.z,
            title=f"Optimized Minimum from R-TS interpolation idx{idx}",
            plot_dir=plot_dir,
            save=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GAD-RGD1 main.")
    # parser.add_argument(
    #     "--eigen-method",
    #     type=str,
    #     default="qr",
    #     help="Eigenvalue method for GAD (qr, svd, svdforce, inertia, geo, ase, eckartsvd, eckartqr)",
    # )
    parser.add_argument(
        "--do-gad",
        action="store_true",
        help="Run GAD",
    )
    parser.add_argument(
        "--do-sella",
        action="store_true",
        help="Run Sella",
    )
    parser.add_argument(
        "--do-irc-neb-geodesic",
        action="store_true",
        help="Run IRC, NEB, and geodesic interpolation",
    )
    parser.add_argument(
        "--do-sella-hessian",
        action="store_true",
        help="Run Sella with Equiformer Hessian",
    )
    parser.add_argument(
        "--do-forces",
        action="store_true",
        help="Run forces to find reactant minimum",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=104_000,
        help="Index of the sample to load from the dataset.",
    )
    args, unknown = parser.parse_known_args()

    # eigen_kwargs = {}
    # # Parse any additional arguments in format --key=value and add to eigen_kwargs
    # for arg in unknown:
    #     if arg.startswith('--') and '=' in arg:
    #         key, value = arg[2:].split('=', 1)
    #         # Try to convert to appropriate type
    #         try:
    #             # Try float first
    #             value = float(value)
    #         except ValueError:
    #             try:
    #                 # Try int
    #                 value = int(value)
    #             except ValueError:
    #                 # Keep as string
    #                 pass
    #         eigen_kwargs[key] = value
    #     elif arg.startswith('--'):
    #         # Boolean flag
    #         key = arg[2:]
    #         eigen_kwargs[key] = True

    results = main(
        # eigen_method=args.eigen_method,
        idx=args.idx,
        do_gad=args.do_gad,
        do_sella=args.do_sella,
        do_irc_neb_geodesic=args.do_irc_neb_geodesic,
        do_sella_hessian=args.do_sella_hessian,
        do_forces=args.do_forces,
    )
    if results is not None:
        print("Results:")
        for k, v in results.items():
            print(f"  {k}: {v:.6f}")

    print("\nDone ✅")
