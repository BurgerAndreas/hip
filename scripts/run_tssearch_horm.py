"""
GAD-RGD1: Gentlest Ascent Dynamics for Finding Transition State

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
python playground/run_tssearch_rgd1.py --do-gad
"""

import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData
import numpy as np
import matplotlib.pyplot as plt

# from rdkit import Chem
# from rdkit.Chem.Draw import rdMolDraw2D
# from rdkit.Chem.rdchem import Mol
# from rdkit.Chem.rdMolAlign import AlignMol
# from rdkit.Chem.rdMolAlign import GetBestRMS
# import io
# import base64
# from IPython.display import Image
import os
import argparse
import json
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from contextlib import contextmanager
import logging
import traceback

try:
    # pysisyphus imports for TS optimization and IRC
    from pysisyphus.Geometry import Geometry
    from pysisyphus.tsoptimizers.RSPRFOptimizer import RSPRFOptimizer
    from pysisyphus.irc.EulerPC import EulerPC
    from pysisyphus.calculators.FakeASE import FakeASE
    from pysisyphus.calculators.MLFF import MLFF
    from pysisyphus.constants import BOHR2ANG
    from pysisyphus.optimizers.RFOptimizer import RFOptimizer

    # PyGSM imports for Growing String Method
    from pyGSM.coordinate_systems.delocalized_coordinates import (
        DelocalizedInternalCoordinates,
    )
    from pyGSM.coordinate_systems.primitive_internals import (
        PrimitiveInternalCoordinates,
    )
    from pyGSM.coordinate_systems.topology import Topology
    from pyGSM.growing_string_methods import DE_GSM
    from pyGSM.level_of_theories.ase import ASELoT
    from pyGSM.optimizers.eigenvector_follow import eigenvector_follow
    from pyGSM.optimizers.lbfgs import lbfgs
    from pyGSM.potential_energy_surfaces import PES
    from pyGSM.utilities import nifty
    from pyGSM.utilities.elements import ElementData
    from pyGSM.molecule.molecule import Molecule
except ImportError:
    print()
    traceback.print_exc()
    print("\nFollow the instructions here: https://github.com/BurgerAndreas/ReactBench")
    exit()

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
from ase.io import write as ase_write
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.vibrations.data import VibrationsData

from transition1x import Dataloader as T1xDataloader

# from sella import Sella, Constraints
# from sella.peswrapper import InternalPES
# from sella.internal import Internals

from hip.geodesic_interpolate import geodesic_interpolate_wrapper
from hip.trajectorysaver import MyTrajectory
import traceback


@contextmanager
def suppress_stdout_stderr(_verbose=True):
    if _verbose:
        yield
        return
    else:
        with open(os.devnull, "w") as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = devnull, devnull
                yield
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr


this_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(this_dir, "plots_hormtssearch")
log_dir = os.path.join(this_dir, "logs_hormtssearch")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def run_gsm_horm(
    atoms_reactant, atoms_product, calculator, idx=0, display_log_level=logging.WARNING
):
    """
    Run Growing String Method (GSM) to find transition state initial guess.

    Args:
        atoms_reactant: ASE Atoms object for reactant
        atoms_product: ASE Atoms object for product
        calculator: ASE calculator (EquiformerASECalculator)
        idx: Sample index for naming/ID purposes

    Returns:
        ts_atoms: ASE Atoms object for the highest energy node (TS initial guess)
        gsm_path: List of ASE Atoms objects representing the entire GSM path
    """
    print("\n" + "=" * 60)
    print("Running Growing String Method (GSM)")
    print("=" * 60)

    # Configure logging to only show warnings and errors during GSM
    nifty_logger = logging.getLogger("NiftyLogger")
    original_level = nifty_logger.level
    nifty_logger.setLevel(display_log_level)

    # Also set handler levels to suppress verbose output
    original_handler_levels = []
    for handler in nifty_logger.handlers:
        original_handler_levels.append(handler.level)
        handler.setLevel(display_log_level)

    # GSM parameters following HORM workflow
    num_nodes = 9  # As specified in HORM paper
    coordinate_type = "TRIC"  # Translation-rotation-internal coordinates
    optimizer_method = "eigenvector_follow"
    line_search = "NoLineSearch"
    only_climb = True  # Climbing image enabled
    step_size_cap = 0.1  # DMAX

    # Convergence parameters
    add_node_tol = 0.1
    conv_tol = 0.0005
    conv_Ediff = 100.0
    conv_gmax = 100.0
    max_gsm_iterations = 100
    max_opt_steps = 3
    ID = idx

    print("GSM Parameters:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Coordinate type: {coordinate_type}")
    print(f"  Climbing image: {only_climb}")
    print(f"  Max iterations: {max_gsm_iterations}")

    # Setup GSM scratch directory
    scratch_dir = os.path.join(os.getcwd(), "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    # LOT (Level of Theory)
    nifty.printcool("Setting up Level of Theory", level=logging.INFO)
    lot = ASELoT.from_options(
        calculator,
        geom=[[x.symbol, *x.position] for x in atoms_reactant],
        ID=ID,
    )

    # PES (Potential Energy Surface)
    pes_obj = PES.from_options(lot=lot, ad_idx=0, multiplicity=1)

    # Build topologies
    nifty.printcool("Building topologies", level=logging.INFO)
    element_table = ElementData()
    elements = [
        element_table.from_symbol(sym) for sym in atoms_reactant.get_chemical_symbols()
    ]

    topology_reactant = Topology.build_topology(
        xyz=atoms_reactant.get_positions(), atoms=elements
    )
    topology_product = Topology.build_topology(
        xyz=atoms_product.get_positions(), atoms=elements
    )

    # Union of bonds (add product bonds to reactant topology)
    for bond in topology_product.edges():
        if (
            bond in topology_reactant.edges()
            or (bond[1], bond[0]) in topology_reactant.edges()
        ):
            continue
        logging.debug(f" Adding bond {bond} to reactant topology")
        if bond[0] > bond[1]:
            topology_reactant.add_edge(bond[0], bond[1])
        else:
            topology_reactant.add_edge(bond[1], bond[0])

    # Primitive internal coordinates
    nifty.printcool("Building Primitive Internal Coordinates", level=logging.INFO)
    # with suppress_stdout_stderr(_verbose=verbose):
    prim_reactant = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        topology=topology_reactant,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
    )

    prim_product = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_product.get_positions(),
        atoms=elements,
        topology=topology_product,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
    )

    # Add product coords to reactant coords
    prim_reactant.add_union_primitives(prim_product)

    # Delocalized internal coordinates
    nifty.printcool("Building Delocalized Internal Coordinates", level=logging.INFO)
    # with suppress_stdout_stderr(_verbose=verbose):
    deloc_coords_reactant = DelocalizedInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
        primitives=prim_reactant,
    )

    # Build molecules
    nifty.printcool(f"Building molecules with {coordinate_type}", level=logging.INFO)
    from_hessian = optimizer_method == "eigenvector_follow"
    # with suppress_stdout_stderr(_verbose=verbose):
    molecule_reactant = Molecule.from_options(
        geom=[[x.symbol, *x.position] for x in atoms_reactant],
        PES=pes_obj,
        coord_obj=deloc_coords_reactant,
        Form_Hessian=from_hessian,
    )

    molecule_product = Molecule.copy_from_options(
        molecule_reactant,
        xyz=atoms_product.get_positions(),
        new_node_id=num_nodes - 1,
        copy_wavefunction=False,
    )

    # Optimizer
    nifty.printcool("Building Optimizer", level=logging.DEBUG)
    opt_options = dict(
        print_level=0,  # Reduced from 1
        Linesearch=line_search,
        update_hess_in_bg=not (only_climb or optimizer_method == "lbfgs"),
        conv_Ediff=conv_Ediff,
        conv_gmax=conv_gmax,
        DMAX=step_size_cap,
        opt_climb=only_climb,
    )

    if optimizer_method == "eigenvector_follow":
        optimizer_object = eigenvector_follow.from_options(**opt_options)
    elif optimizer_method == "lbfgs":
        optimizer_object = lbfgs.from_options(**opt_options)
    else:
        raise NotImplementedError(f"Optimizer {optimizer_method} not implemented")

    # GSM object
    nifty.printcool("Building GSM object", level=logging.DEBUG)
    # with suppress_stdout_stderr(_verbose=verbose):
    gsm = DE_GSM.from_options(
        reactant=molecule_reactant,
        product=molecule_product,
        nnodes=num_nodes,
        CONV_TOL=conv_tol,
        CONV_gmax=conv_gmax,
        CONV_Ediff=conv_Ediff,
        ADD_NODE_TOL=add_node_tol,
        growth_direction=0,
        optimizer=optimizer_object,
        ID=ID,
        print_level=0,  # Reduced from 1
        mp_cores=1,
        interp_method="DLC",
    )

    # Run GSM calculation
    rtype = 1 if only_climb else 2
    logging.info(f"Starting GSM with {num_nodes} nodes...")

    try:
        # Comprehensive output suppression for pyGSM internal prints

        # Save original file descriptors
        original_stdout_fd = os.dup(sys.stdout.fileno())
        original_stderr_fd = os.dup(sys.stderr.fileno())

        # Open devnull for redirection
        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        try:
            # Redirect both Python-level and file descriptor level output
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull_fd, sys.stdout.fileno())
            os.dup2(devnull_fd, sys.stderr.fileno())

            # Run GSM calculation
            gsm.go_gsm(max_gsm_iterations, max_opt_steps, rtype=rtype)

        finally:
            # Restore original file descriptors
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(original_stdout_fd, sys.stdout.fileno())
            os.dup2(original_stderr_fd, sys.stderr.fileno())

            # Close file descriptors
            os.close(devnull_fd)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
        nifty.printcool("GSM completed successfully!", level=logging.INFO)

        # Extract results
        gsm_path = []
        energies = []

        for i, (energy, geom) in enumerate(zip(gsm.energies, gsm.geometries)):
            atoms = Atoms(
                symbols=[x[0] for x in geom], positions=[x[1:4] for x in geom]
            )
            atoms.info["energy"] = energy
            atoms.info["node_id"] = i
            gsm_path.append(atoms)
            energies.append(energy)

        # Find highest energy node (TS initial guess)
        max_energy_idx = np.argmax(energies)
        ts_geom = gsm.nodes[gsm.TSnode].geometry
        ts_atoms = Atoms(
            symbols=[x[0] for x in ts_geom], positions=[x[1:4] for x in ts_geom]
        )
        ts_atoms.info["energy"] = gsm.energies[gsm.TSnode]
        ts_atoms.info["node_id"] = gsm.TSnode

        logging.info(
            f"TS found at node {gsm.TSnode}, Energy: {ts_atoms.info['energy']:.3f}"
        )

        # Save results quietly
        gsm_path_file = os.path.join(plot_dir, f"gsm_path_idx{idx}.xyz")

        ase_write(gsm_path_file, gsm_path)

        ts_file = os.path.join(plot_dir, f"ts_initial_guess_idx{idx}.xyz")
        ase_write(ts_file, ts_atoms)

        return ts_atoms, gsm_path

    except Exception as e:
        logging.error(f"GSM calculation failed: {e}")
        logging.error(traceback.format_exc())
        return None, None

    finally:
        # Restore original logging levels
        try:
            nifty_logger.setLevel(original_level)
            for handler, orig_level in zip(
                nifty_logger.handlers, original_handler_levels
            ):
                handler.setLevel(orig_level)
        except:
            pass

        # Cleanup scratch directory
        try:
            import shutil

            scratch_path = os.path.join(os.getcwd(), "scratch", f"{ID:03d}")
            if os.path.exists(scratch_path):
                shutil.rmtree(scratch_path)
                logging.debug(f"Cleaned up scratch directory: {scratch_path}")
        except:
            pass


def run_pysisyphus_ts_optimization(
    ts_initial_guess, asecalc, idx=0, display_log_level=logging.WARNING
):
    """
    Optimize TS using pysisyphus RS-I-RFO method.

    Args:
        ts_initial_guess: ASE Atoms object (from GSM)
        asecalc: EquiformerASECalculator
        idx: Sample index for naming
        display_log_level: Logging level to display

    Returns:
        ts_optimized: ASE Atoms object (optimized TS)
        optimization_success: bool
        ts_opt_results: dict with optimization info
    """

    print("\n" + "=" * 60)
    print("Running pysisyphus RS-I-RFO TS Optimization")
    print("=" * 60)

    # Configure logging
    # pysis_logger = logging.getLogger("pysisyphus")
    pysis_logger = logging.getLogger("optimizer")
    original_level = pysis_logger.level
    pysis_logger.setLevel(display_log_level)

    # TS optimization parameters following HORM workflow
    trust_radius = 0.2  # Å
    max_cycles = 50
    hessian_recalc = 1  # Recalculate Hessian at every step for MLIP
    thresh = "gau"  # Gaussian default convergence thresholds

    try:
        # Convert ASE Atoms to pysisyphus Geometry
        # Coordinates need to be in Bohr (pysisyphus internal units)
        atoms_symbols = ts_initial_guess.get_chemical_symbols()
        coords_ang = ts_initial_guess.get_positions()
        coords_bohr = coords_ang / BOHR2ANG  # Convert Å → Bohr

        # Create pysisyphus Geometry object
        ts_geom = Geometry(
            atoms=atoms_symbols,
            coords=coords_bohr.flatten(),
            coord_type="cart",  # Use Cartesian coordinates
        )

        # Set up asecalc wrapper
        calculator = MLFF(
            method="equiformerv2",
            # model_kwargs={"model": asecalc.potential}
        )
        ts_geom.calc = calculator
        ts_geom.calculator = calculator

        # Initialize RS-I-RFO optimizer
        print("Initializing RS-I-RFO optimizer...")
        ts_optimizer = RSPRFOptimizer(
            ts_geom,
            trust_max=trust_radius,
            max_cycles=max_cycles,
            hessian_recalc=hessian_recalc,
            # gau_loose, gau, gau_tight, gau_vtight
            thresh=thresh,
            dump=True,
            prefix=f"ts_opt_idx{idx}",
            out_dir=log_dir,
            print_every=10,
            ## TSHessianOptimizer
            # root: int = 0,
            # hessian_ref: Optional[str] = None,
            # rx_modes=None,
            # prim_coord=None,
            # rx_coords=None,
            # hessian_init: HessInit = "calc",
            # hessian_update: HessUpdate = "bofill",
            # hessian_recalc_reset: bool = True,
            # max_micro_cycles: int = 50,
            # trust_radius: float = 0.3,
            # trust_max: float = 0.5,
            # augment_bonds: bool = False,
            # min_line_search: bool = False,
            # max_line_search: bool = False,
            # assert_neg_eigval: bool = False,
            # # HessianOptimizer
            # trust_radius: float = 0.5,
            # trust_update: bool = True,
            # trust_min: float = 0.1,
            # trust_max: float = 1,
            # max_energy_incr: Optional[float] = None,
            # hessian_update: HessUpdate = "bfgs",
            # hessian_init: HessInit = "fischer",
            # hessian_recalc: Optional[int] = None,
            # hessian_recalc_adapt: Optional[float] = None,
            # hessian_xtb: bool = False,
            # hessian_recalc_reset: bool = False,
            # small_eigval_thresh: float = 1e-8,
            # line_search: bool = False,
            # alpha0: float = 1.0,
            # max_micro_cycles: int = 25,
            # rfo_overlaps: bool = False,
            default_logging_level=logging.DEBUG,
        )

        # Run TS optimization
        print("Starting TS optimization...")
        initial_energy = ts_geom.energy
        print(f"Initial energy: {initial_energy:.6f} eV")

        # Suppress verbose output during optimization
        # logging happens through optimizer
        # self.logger = logging.getLogger("optimizer")
        # with suppress_stdout_stderr(_verbose=(display_log_level <= logging.INFO)):
        ts_optimizer.run()

        # Check convergence
        optimization_success = ts_optimizer.is_converged
        final_energy = ts_geom.energy
        num_cycles = ts_optimizer.cur_cycle

        print(f"TS optimization completed after {num_cycles} cycles")
        print(f"Final energy: {final_energy:.6f} eV")
        print(f"Energy change: {final_energy - initial_energy:.6f} eV")

        if optimization_success:
            print("✓ TS optimization converged successfully")
        else:
            print("⚠ TS optimization did not converge")

        # Convert optimized geometry back to ASE Atoms
        ts_optimized = ts_geom.as_ase_atoms()
        ts_optimized.info["energy"] = final_energy
        ts_optimized.info["optimization_converged"] = optimization_success
        ts_optimized.info["optimization_cycles"] = num_cycles

        # Save optimized TS structure
        ts_opt_file = os.path.join(plot_dir, f"ts_optimized_idx{idx}.xyz")
        ase_write(ts_opt_file, ts_optimized)

        # Prepare results summary
        ts_opt_results = {
            "converged": optimization_success,
            "cycles": num_cycles,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": final_energy - initial_energy,
            "trust_radius": trust_radius,
            "max_cycles": max_cycles,
        }

        return ts_optimized, optimization_success, ts_opt_results

    except Exception as e:
        print(f"TS optimization failed: {e}")
        print(traceback.format_exc())
        return None, False, {"error": str(e)}

    finally:
        # Restore original logging level
        try:
            pysis_logger.setLevel(original_level)
        except:
            pass


def run_pysisyphus_irc(
    ts_atoms,
    asecalc,
    reactant_atoms,
    product_atoms,
    idx=0,
    display_log_level=logging.WARNING,
):
    """
    Run IRC calculation to validate TS using pysisyphus EulerPC integrator.

    Args:
        ts_atoms: ASE Atoms object (optimized TS)
        asecalc: EquiformerASECalculator
        reactant_atoms: ASE Atoms object (original reactant for comparison)
        product_atoms: ASE Atoms object (original product for comparison)
        idx: Sample index
        display_log_level: Logging level to display

    Returns:
        irc_endpoints: Tuple of (reactant_like, product_like) ASE Atoms
        is_intended: bool (whether IRC connects expected reactant/product)
        irc_results: dict with IRC info
    """

    print("\n" + "=" * 60)
    print("Running pysisyphus IRC Validation")
    print("=" * 60)

    # Configure logging
    # pysis_logger = logging.getLogger("pysisyphus")
    pysis_logger = logging.getLogger("optimizer")
    original_level = pysis_logger.level
    pysis_logger.setLevel(display_log_level)

    # IRC parameters following HORM workflow
    rms_grad_thresh = 0.0005  # RMS gradient threshold for convergence
    hessian_recalc = 10  # Recalculate Hessian every 10 steps
    rmsd_threshold = 1.0  # RMSD threshold for "intended" classification

    print("IRC Parameters:")
    print(f"  RMS gradient threshold: {rms_grad_thresh}")
    print(f"  Hessian recalc: every {hessian_recalc} steps")
    print(f"  RMSD threshold for intended: {rmsd_threshold} Å")

    try:
        # Convert ASE Atoms to pysisyphus Geometry
        atoms_symbols = ts_atoms.get_chemical_symbols()
        coords_ang = ts_atoms.get_positions()
        coords_bohr = coords_ang / BOHR2ANG  # Convert Å → Bohr

        # Create pysisyphus Geometry object for TS
        irc_geom = Geometry(
            atoms=atoms_symbols, coords=coords_bohr.flatten(), coord_type="cart"
        )

        # Set up calculator wrapper
        irc_geom.calc = asecalc
        irc_geom.calculator = asecalc

        # Initialize EulerPC IRC integrator
        print("Initializing EulerPC IRC integrator...")
        irc_integrator = EulerPC(
            irc_geom,
            rms_grad_thresh=rms_grad_thresh,
            hessian_recalc=hessian_recalc,
            dump=True,
            prefix=f"irc_idx{idx}",
            out_dir=log_dir,
        )

        # Run IRC calculation
        print("Starting IRC integration...")
        ts_energy = irc_geom.energy
        print(f"TS energy: {ts_energy:.6f} eV")

        # Suppress verbose output during IRC
        # with suppress_stdout_stderr(_verbose=(display_log_level <= logging.INFO)):
        irc_integrator.run()

        # Extract IRC endpoints
        irc_coords = irc_integrator.all_coords
        forward_endpoint_coords = irc_coords[-1]  # Last point in forward direction
        backward_endpoint_coords = irc_coords[0]  # First point in backward direction

        # Create geometries for endpoints
        forward_geom = irc_geom.copy()
        forward_geom.coords = forward_endpoint_coords
        backward_geom = irc_geom.copy()
        backward_geom.coords = backward_endpoint_coords

        # Optimize IRC endpoints to stationary points
        print("Optimizing IRC endpoints...")

        # Forward endpoint optimization
        # from pysisyphus.optimizers.optimizers import Optimizer
        forward_optimizer = RFOptimizer(
            forward_geom,
            max_cycles=50,
            thresh="gau",
            check_eigval_structure=False,
            # max_step
            # max_cycles
            print_every=10,
            # prefix="",
        )
        with suppress_stdout_stderr(_verbose=(display_log_level <= logging.INFO)):
            forward_optimizer.run()

        # Backward endpoint optimization
        backward_optimizer = RFOptimizer(backward_geom, max_cycles=50, thresh="gau")
        with suppress_stdout_stderr(_verbose=(display_log_level <= logging.INFO)):
            backward_optimizer.run()

        # Convert optimized endpoints back to ASE Atoms
        forward_endpoint = forward_geom.as_ase_atoms()
        backward_endpoint = backward_geom.as_ase_atoms()

        forward_endpoint.info["energy"] = forward_geom.energy
        backward_endpoint.info["energy"] = backward_geom.energy

        # Compare endpoints with original reactant and product using RMSD
        # Test both possible assignments (forward/backward vs reactant/product)

        # Assignment 1: forward → product, backward → reactant
        rmsd_forward_product = align_ordered_and_get_rmsd(
            forward_endpoint.get_positions(), product_atoms.get_positions()
        )
        rmsd_backward_reactant = align_ordered_and_get_rmsd(
            backward_endpoint.get_positions(), reactant_atoms.get_positions()
        )
        assignment1_rmsd = max(rmsd_forward_product, rmsd_backward_reactant)

        # Assignment 2: forward → reactant, backward → product
        rmsd_forward_reactant = align_ordered_and_get_rmsd(
            forward_endpoint.get_positions(), reactant_atoms.get_positions()
        )
        rmsd_backward_product = align_ordered_and_get_rmsd(
            backward_endpoint.get_positions(), product_atoms.get_positions()
        )
        assignment2_rmsd = max(rmsd_forward_reactant, rmsd_backward_product)

        # Choose the better assignment
        if assignment1_rmsd <= assignment2_rmsd:
            reactant_like = backward_endpoint
            product_like = forward_endpoint
            max_rmsd = assignment1_rmsd
            print(f"Best assignment: backward→reactant, forward→product")
            print(f"  Backward vs reactant RMSD: {rmsd_backward_reactant:.4f} Å")
            print(f"  Forward vs product RMSD: {rmsd_forward_product:.4f} Å")
        else:
            reactant_like = forward_endpoint
            product_like = backward_endpoint
            max_rmsd = assignment2_rmsd
            print(f"Best assignment: forward→reactant, backward→product")
            print(f"  Forward vs reactant RMSD: {rmsd_forward_reactant:.4f} Å")
            print(f"  Backward vs product RMSD: {rmsd_backward_product:.4f} Å")

        # Determine if this is an "intended" reaction
        is_intended = max_rmsd <= rmsd_threshold
        print(f"Maximum RMSD: {max_rmsd:.4f} Å")

        if is_intended:
            print("✓ IRC validation successful - Intended reaction")
        else:
            print("⚠ IRC validation failed - Unintended reaction")

        # Save IRC endpoints
        reactant_like_file = os.path.join(plot_dir, f"irc_reactant_like_idx{idx}.xyz")
        product_like_file = os.path.join(plot_dir, f"irc_product_like_idx{idx}.xyz")
        ase_write(reactant_like_file, reactant_like)
        ase_write(product_like_file, product_like)

        # Save full IRC trajectory
        irc_trajectory = []
        for coords in irc_coords:
            irc_atoms = Atoms(
                symbols=atoms_symbols,
                positions=coords.reshape(-1, 3) * BOHR2ANG,  # Convert back to Å
            )
            irc_trajectory.append(irc_atoms)

        irc_traj_file = os.path.join(plot_dir, f"irc_trajectory_idx{idx}.xyz")
        ase_write(irc_traj_file, irc_trajectory)

        # Prepare results summary
        irc_results = {
            "is_intended": is_intended,
            "max_rmsd": max_rmsd,
            "rmsd_threshold": rmsd_threshold,
            "forward_vs_product": rmsd_forward_product,
            "backward_vs_reactant": rmsd_backward_reactant,
            "forward_vs_reactant": rmsd_forward_reactant,
            "backward_vs_product": rmsd_backward_product,
            "irc_steps": len(irc_coords),
            "ts_energy": ts_energy,
            "forward_energy": forward_geom.energy,
            "backward_energy": backward_geom.energy,
        }

        return (reactant_like, product_like), is_intended, irc_results

    except Exception as e:
        print(f"IRC calculation failed: {e}")
        print(traceback.format_exc())
        return None, False, {"error": str(e)}

    finally:
        # Restore original logging level
        try:
            pysis_logger.setLevel(original_level)
        except:
            pass


def run_horm_ts_search(
    idx=108000,  # 104_000,
    display_log_level=logging.WARNING,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(plot_dir, exist_ok=True)

    # Load the dataset
    # TODO: implement transition1x
    # dataloader = T1xDataloader(data/transition1x.h5)

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
    # Use interpolation as initial guess instead of Growing String Method (GSM)
    print("\nCreating initial guess using interpolation")

    # Get reactant and transition state positions
    pos_reactant = sample.pos_reactant
    pos_transition = sample.pos_transition
    pos_product = sample.pos_product

    alpha = 0.5  # midpoint
    # Linear interpolation between reactant and product
    x_lininter_rp = (1 - alpha) * pos_reactant + alpha * pos_product
    plot_molecule_mpl(
        x_lininter_rp,
        atomic_numbers=sample.z,
        title=f"R-P linear interpolation idx{idx}",
        plot_dir=plot_dir,
        save=True,
    )

    # Create ASE Atoms object
    reactant = Atoms(numbers=to_numpy(sample.z), positions=to_numpy(pos_reactant))
    reactant.calc = asecalc
    product = Atoms(numbers=to_numpy(sample.z), positions=to_numpy(pos_product))
    product.calc = asecalc

    # geodesic interpolation
    geointer_atoms_list = geodesic_interpolate_wrapper(
        reactant, product, n_images=3, return_middle_image=True
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
    # Run relaxation (skip for now)
    # uses energy and forces from the NN

    ###################################################################################
    # Run Growing String Method (GSM) using pyGSM to get initial guess
    # uses energy and forces from the NN

    # Convert tensor positions to ASE Atoms objects
    atoms_reactant = Atoms(
        symbols=[int(z) for z in sample.z], positions=pos_reactant.cpu().numpy()
    )
    atoms_product = Atoms(
        symbols=[int(z) for z in sample.z], positions=pos_product.cpu().numpy()
    )

    # Run GSM calculation
    ts_initial_guess, gsm_path = run_gsm_horm(
        atoms_reactant=atoms_reactant,
        atoms_product=atoms_product,
        calculator=asecalc,
        idx=idx,
        display_log_level=display_log_level,
    )

    if ts_initial_guess is not None and gsm_path is not None:
        # Extract TS initial guess position
        x_gsm_ts = ts_initial_guess.get_positions()

        # Plot GSM TS initial guess
        plot_molecule_mpl(
            x_gsm_ts,
            atomic_numbers=sample.z,
            title=f"GSM TS initial guess idx{idx}",
            plot_dir=plot_dir,
            save=True,
        )

        # Compare GSM TS guess with actual transition state
        rmsd_gsm_vs_actual = align_ordered_and_get_rmsd(
            x_gsm_ts, pos_transition.cpu().numpy()
        )
        rmsd_gsm_vs_linear = align_ordered_and_get_rmsd(
            x_gsm_ts, x_lininter_rp.cpu().numpy()
        )
        rmsd_gsm_vs_geodesic = align_ordered_and_get_rmsd(
            x_gsm_ts, x_geointer_rp.cpu().numpy()
        )

        print(f"RMSD vs actual TS: {rmsd_gsm_vs_actual:.4f} Å")

        # Plot energy profile along GSM path
        if len(gsm_path) > 0:
            energies = [atoms.info["energy"] for atoms in gsm_path]
            node_indices = list(range(len(energies)))

            plt.figure(figsize=(10, 6))
            plt.plot(node_indices, energies, "o-", linewidth=2, markersize=8)
            plt.axvline(
                x=ts_initial_guess.info["node_id"],
                color="red",
                linestyle="--",
                label=f"TS node {ts_initial_guess.info['node_id']}",
            )
            plt.xlabel("Node Index")
            plt.ylabel("Energy (eV)")
            plt.title(f"GSM Energy Profile - Sample {idx}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            energy_plot_file = os.path.join(
                plot_dir, f"gsm_energy_profile_idx{idx}.png"
            )
            plt.savefig(energy_plot_file, dpi=300, bbox_inches="tight")
            plt.close()

        # Save comparison results
        results_summary = {
            "idx": idx,
            "gsm_success": True,
            "gsm_nodes": len(gsm_path),
            "ts_node_id": ts_initial_guess.info["node_id"],
            "ts_energy": ts_initial_guess.info["energy"],
            "rmsd_gsm_vs_actual_ts": rmsd_gsm_vs_actual,
            "rmsd_gsm_vs_linear": rmsd_gsm_vs_linear,
            "rmsd_gsm_vs_geodesic": rmsd_gsm_vs_geodesic,
            "energy_range": max(energies) - min(energies) if len(gsm_path) > 0 else 0.0,
        }

        results_file = os.path.join(log_dir, f"gsm_results_idx{idx}.json")
        with open(results_file, "w") as f:
            json.dump(results_summary, f, indent=2)

        if rmsd_gsm_vs_actual < 1.0:
            print("✓ Good TS initial guess")
        else:
            print("⚠ TS guess differs significantly from actual TS")

    else:
        print("GSM calculation failed")
        results_summary = {
            "idx": idx,
            "gsm_success": False,
            "error": "GSM calculation failed",
        }
        results_file = os.path.join(log_dir, f"gsm_results_idx{idx}.json")
        with open(results_file, "w") as f:
            json.dump(results_summary, f, indent=2)

    ###################################################################################
    # Run local dimer-like search to find transition state
    # restricted step rational-function-optimization (RS-I-RFO) algorithm using pysisyphus
    # uses NN Hessian

    ts_optimized = None
    optimization_success = False
    ts_opt_results = {}

    if ts_initial_guess is not None:
        print("\n" + "=" * 80)
        print("Running RS-I-RFO TS Optimization with pysisyphus")
        print("=" * 80)

        ts_optimized, optimization_success, ts_opt_results = (
            run_pysisyphus_ts_optimization(
                ts_initial_guess=ts_initial_guess,
                asecalc=asecalc,
                idx=idx,
                display_log_level=display_log_level,
            )
        )

        if ts_optimized is not None and optimization_success:
            # Plot optimized TS
            plot_molecule_mpl(
                ts_optimized.get_positions(),
                atomic_numbers=sample.z,
                title=f"Optimized TS idx{idx}",
                plot_dir=plot_dir,
                save=True,
            )

            # Compare optimized TS with actual transition state
            rmsd_opt_vs_actual = align_ordered_and_get_rmsd(
                ts_optimized.get_positions(), pos_transition.cpu().numpy()
            )
            print(f"Optimized TS vs actual TS RMSD: {rmsd_opt_vs_actual:.4f} Å")

            # Update results summary
            if "results_summary" in locals():
                results_summary.update(
                    {
                        "ts_optimization_success": True,
                        "ts_opt_converged": optimization_success,
                        "ts_opt_cycles": ts_opt_results.get("cycles", 0),
                        "ts_energy_change": ts_opt_results.get("energy_change", 0.0),
                        "rmsd_opt_vs_actual_ts": rmsd_opt_vs_actual,
                    }
                )
            print("✓ TS optimization completed successfully")
        else:
            print("⚠ TS optimization failed")
            if "results_summary" in locals():
                results_summary.update(
                    {
                        "ts_optimization_success": False,
                        "ts_opt_error": ts_opt_results.get("error", "Unknown error"),
                    }
                )
    else:
        print("⚠ No TS initial guess available, skipping TS optimization")

    ###################################################################################
    # Run Intrinsic Reaction Coordinate (IRC)
    # compare both forward and backward structures to the starting reactant and product
    # uses energy and forces from the NN

    irc_endpoints = None
    is_intended = False
    irc_results = {}

    if ts_optimized is not None and optimization_success:
        print("\n" + "=" * 80)
        print("Running IRC Validation with pysisyphus")
        print("=" * 80)

        irc_endpoints, is_intended, irc_results = run_pysisyphus_irc(
            ts_atoms=ts_optimized,
            asecalc=asecalc,
            reactant_atoms=atoms_reactant,
            product_atoms=atoms_product,
            idx=idx,
            display_log_level=display_log_level,
        )

        if irc_endpoints is not None:
            reactant_like, product_like = irc_endpoints

            # Plot IRC endpoints
            plot_molecule_mpl(
                reactant_like.get_positions(),
                atomic_numbers=sample.z,
                title=f"IRC Reactant-like idx{idx}",
                plot_dir=plot_dir,
                save=True,
            )
            plot_molecule_mpl(
                product_like.get_positions(),
                atomic_numbers=sample.z,
                title=f"IRC Product-like idx{idx}",
                plot_dir=plot_dir,
                save=True,
            )

            # Update results summary
            if "results_summary" in locals():
                results_summary.update(
                    {
                        "irc_success": True,
                        "is_intended": is_intended,
                        "irc_max_rmsd": irc_results.get("max_rmsd", 0.0),
                        "irc_steps": irc_results.get("irc_steps", 0),
                        "forward_energy": irc_results.get("forward_energy", 0.0),
                        "backward_energy": irc_results.get("backward_energy", 0.0),
                    }
                )

            if is_intended:
                print("✓ IRC validation successful - Intended reaction")
            else:
                print("⚠ IRC validation shows unintended reaction")
        else:
            print("⚠ IRC calculation failed")
            if "results_summary" in locals():
                results_summary.update(
                    {
                        "irc_success": False,
                        "irc_error": irc_results.get("error", "Unknown error"),
                    }
                )
    else:
        print("⚠ No optimized TS available, skipping IRC validation")

    # Return results summary
    if "results_summary" in locals():
        # Save comprehensive results
        comprehensive_results_file = os.path.join(
            log_dir, f"horm_workflow_results_idx{idx}.json"
        )
        with open(comprehensive_results_file, "w") as f:
            json.dump(results_summary, f, indent=2)
        return results_summary
    else:
        default_results = {
            "idx": idx,
            "status": "completed",
            "gsm_success": False,
            "ts_optimization_success": False,
            "irc_success": False,
        }
        default_results_file = os.path.join(
            log_dir, f"horm_workflow_results_idx{idx}.json"
        )
        with open(default_results_file, "w") as f:
            json.dump(default_results, f, indent=2)
        return default_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HORM-TSsearch-workflow main.")
    parser.add_argument(
        "--idx",
        type=int,
        default=108000,  # 104_000,
        help="Index of the sample to load from the dataset.",
    )
    args, unknown = parser.parse_known_args()

    results = run_horm_ts_search(
        # eigen_method=args.eigen_method,
        idx=args.idx,
    )
    if results is not None:
        print("\n" + "=" * 80)
        print("HORM Transition State Search Results Summary")
        print("=" * 80)

        # Core workflow results
        print(f"Sample Index: {results.get('idx', 'N/A')}")
        print(f"GSM Success: {results.get('gsm_success', False)}")
        print(
            f"TS Optimization Success: {results.get('ts_optimization_success', False)}"
        )
        print(f"IRC Success: {results.get('irc_success', False)}")
        print(f"Intended Reaction: {results.get('is_intended', False)}")

        # GSM results
        if results.get("gsm_success", False):
            print(f"\nGSM Results:")
            print(f"  Nodes: {results.get('gsm_nodes', 'N/A')}")
            print(f"  TS Node ID: {results.get('ts_node_id', 'N/A')}")
            print(f"  TS Energy: {results.get('ts_energy', 0.0):.6f} eV")
            print(
                f"  RMSD vs Actual TS: {results.get('rmsd_gsm_vs_actual_ts', 0.0):.4f} Å"
            )

        # TS Optimization results
        if results.get("ts_optimization_success", False):
            print(f"\nTS Optimization Results:")
            print(f"  Converged: {results.get('ts_opt_converged', False)}")
            print(f"  Cycles: {results.get('ts_opt_cycles', 0)}")
            print(f"  Energy Change: {results.get('ts_energy_change', 0.0):.6f} eV")
            print(
                f"  RMSD Optimized vs Actual TS: {results.get('rmsd_opt_vs_actual_ts', 0.0):.4f} Å"
            )

        # IRC results
        if results.get("irc_success", False):
            print(f"\nIRC Validation Results:")
            print(f"  Intended: {results.get('is_intended', False)}")
            print(f"  Max RMSD: {results.get('irc_max_rmsd', 0.0):.4f} Å")
            print(f"  IRC Steps: {results.get('irc_steps', 0)}")
            print(f"  Forward Energy: {results.get('forward_energy', 0.0):.6f} eV")
            print(f"  Backward Energy: {results.get('backward_energy', 0.0):.6f} eV")

        # Success indicators
        print(f"\nWorkflow Success Indicators:")
        full_success = (
            results.get("gsm_success", False)
            and results.get("ts_optimization_success", False)
            and results.get("irc_success", False)
            and results.get("is_intended", False)
        )
        print(f"  Full HORM Workflow Success: {full_success}")

        if full_success:
            print(
                "🎉 Complete HORM workflow successful - found intended transition state!"
            )
        elif results.get("gsm_success", False) and results.get(
            "ts_optimization_success", False
        ):
            print("✅ TS found and optimized, but IRC validation needed")
        elif results.get("gsm_success", False):
            print("⚠️ GSM successful, but TS optimization failed")
        else:
            print("❌ GSM failed - no transition state found")

        print(
            "\nDetailed results saved to:",
            f"logs_hormtssearch/horm_workflow_results_idx{results.get('idx', 'unknown')}.json",
        )

    print("\nDone ✅")
