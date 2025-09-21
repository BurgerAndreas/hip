from typing import Callable, Literal

import numpy as np
import math
import logging

from ase import Atoms

# from hip.trajectorysaver import MyTrajectory
from hip.align_ordered_mols import align_ordered_and_get_rmsd
from hip.plot_molecules import (
    plot_molecule_mpl,
    plot_traj_mpl,
    clean_filename,
    save_to_xyz,
    save_trajectory_to_xyz,
)
from hip.align_ordered_mols import find_rigid_alignment

# https://github.com/virtualzx-nad/geodesic-interpolate
from geodesic_interpolate.geodesic import Geodesic
from geodesic_interpolate.interpolation import redistribute


def copy_atoms(atoms: Atoms) -> Atoms:
    """
    Simple function to copy an atoms object to prevent mutability.
    """
    # Needed because of ASE issue #1084
    calc = atoms.calc
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms


def geodesic_interpolate_wrapper(
    reactant: Atoms,
    product: Atoms,
    n_images: int = 3,  # 10,
    perform_sweep: bool | Literal["auto"] = "auto",
    redistribute_tol: float = 1e-2,
    smoother_tol: float = 2e-3,
    max_iterations: int = 15,
    max_micro_iterations: int = 20,
    morse_scaling: float = 1.7,
    geometry_friction: float = 1e-2,
    distance_cutoff: float = 3.0,
    sweep_cutoff_size: int = 35,
    return_middle_image: bool = True,
) -> list[Atoms]:
    """
    Interpolates between two geometries and optimizes the path with the geodesic method.

    Parameters
    ----------
    reactant
        The ASE Atoms object representing the initial geometry.
    product
        The ASE Atoms object representing the final geometry.
    n_images
        Number of images for interpolation. Default is 10.
    perform_sweep
        Whether to sweep across the path optimizing one image at a time.
        Default is to perform sweeping updates if there are more than 35 atoms.
    redistribute_tol
        the value passed to the tol keyword argument of
         geodesic_interpolate.interpolation.redistribute. Default is 1e-2.
    smoother_tol
        the value passed to the tol keyword argument of geodesic_smoother.smooth
        or geodesic_smoother.sweep. Default is 2e-3.
    max_iterations
        Maximum number of minimization iterations. Default is 15.
    max_micro_iterations
        Maximum number of micro iterations for the sweeping algorithm. Default is 20.
    morse_scaling
        Exponential parameter for the Morse potential. Default is 1.7.
    geometry_friction
        Size of friction term used to prevent very large changes in geometry. Default is 1e-2.
    distance_cutoff
        Cut-off value for the distance between a pair of atoms to be included in the coordinate system. Default is 3.0.
    sweep_cutoff_size
        Cut off system size that above which sweep function will be called instead of smooth
        in Geodesic.

    Returns
    -------
    list[Atoms]
        A list of ASE Atoms objects representing the smoothed path between the reactant and product geometries.
    """
    reactant = copy_atoms(reactant)
    product = copy_atoms(product)

    # Read the initial geometries.
    chemical_symbols = reactant.get_chemical_symbols()

    # First redistribute number of images.
    # Perform interpolation if too few and subsampling if too many images are given
    raw_interpolated_positions = redistribute(
        chemical_symbols,
        [reactant.positions, product.positions],
        n_images,
        tol=redistribute_tol,
    )

    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    geodesic_smoother = Geodesic(
        chemical_symbols,
        raw_interpolated_positions,
        morse_scaling,
        threshold=distance_cutoff,
        friction=geometry_friction,
        log_level=logging.DEBUG,
    )
    if perform_sweep == "auto":
        perform_sweep = len(chemical_symbols) > sweep_cutoff_size
    if perform_sweep:
        geodesic_smoother.sweep(
            tol=smoother_tol, max_iter=max_iterations, micro_iter=max_micro_iterations
        )
    else:
        geodesic_smoother.smooth(tol=smoother_tol, max_iter=max_iterations)
    atoms_list = [
        Atoms(symbols=chemical_symbols, positions=geom)
        for geom in geodesic_smoother.path
    ]
    if return_middle_image:
        assert n_images % 2 == 1, "n_images must be odd for return_middle_image"
        return atoms_list[math.floor(n_images / 2)]
    return atoms_list
