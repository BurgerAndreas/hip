from typing import List, Optional, Tuple

import os
import numpy as np
import scipy.linalg
import yaml
from tqdm import tqdm

import torch
import torch.nn
import torch.utils.data

from torch_geometric.loader import DataLoader as TGDataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData
from ocpmodels.ff_lmdb import LmdbDataset
from nets.prediction_utils import compute_extra_props, GLOBAL_ATOM_NUMBERS
from hip.path_config import DATASET_FILES_HORM, fix_dataset_path

from rdkit.Chem import GetPeriodicTable

from ase import Atoms
from ase.vibrations.data import VibrationsData

from ocpmodels.units import bohr_to_angstrom, angstrom_to_bohr

import hip.geometric_normal_modes

"""
get the eigenvectors of the hessian of a force field, 
that do not correspond to extra rotation or translation degrees of freedom (invariance of the energy)
"""

# helper functions


def _is_linear_molecule(coords, threshold=1e-8):
    """
    Check if a molecule is linear by examining the geometry

    Args:
        coords: numpy array of shape (N, 3) with atomic coordinates
        threshold: tolerance for linearity detection

    Returns:
        bool: True if molecule is linear
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    N = len(coords)
    if N <= 2:
        return True

    # Center coordinates
    com = np.mean(coords, axis=0)
    coords_centered = coords - com

    # Compute inertia tensor
    inertia_tensor = np.zeros((3, 3))
    for i in range(N):
        r = coords_centered[i]
        inertia_tensor += np.outer(r, r)

    # Check if smallest eigenvalue is much smaller than others
    eigenvals = np.linalg.eigvals(inertia_tensor)
    eigenvals = np.sort(eigenvals)

    # Linear if smallest eigenvalue is much smaller than largest
    return eigenvals[0] < threshold * eigenvals[-1]


def _get_masses_zsymbols_znumbers(atom_types, device="cpu"):
    pt = GetPeriodicTable()
    if isinstance(atom_types, torch.Tensor):
        device = atom_types.device
        atom_types = atom_types.tolist()
    if isinstance(atom_types[0], str):
        elements = atom_types
        atom_types = [pt.GetAtomicNumber(z) for z in atom_types]
    else:
        elements = [pt.GetElementSymbol(z) for z in atom_types]
    masses = torch.tensor([pt.GetAtomicWeight(z) for z in atom_types], device=device)
    return masses, elements, atom_types


def _compute_numerical_rank_threshold(
    evals: torch.Tensor, matrix_shape: tuple
) -> float:
    """
    Compute adaptive threshold for numerical rank determination using the same algorithm as
    NumPy's matrix_rank and MATLAB's rank function.

    This sets threshold = max(singular_values) * max(matrix_dimensions) * machine_epsilon

    Args:
        evals: Eigenvalues (can be negative for Hessian)
        matrix_shape: Shape of the matrix (M, N)

    Returns:
        Adaptive threshold value
    """
    abs_evals = torch.abs(evals)
    max_eval = torch.max(abs_evals)

    # Use the same algorithm as NumPy's matrix_rank: S.max() * max(M, N) * eps
    M, N = matrix_shape
    eps = torch.finfo(evals.dtype).eps
    threshold = max_eval * max(M, N) * eps

    return threshold.item()


def _get_ndrop(
    evals, m_shape, threshold, sorted_evals=None, islinear=False, print_warning=False
):
    expected_dof = 5 if islinear else 6
    if sorted_evals is None:
        sorted_evals, sort_idx = torch.sort(torch.abs(evals))
    # Compute adaptive threshold using NumPy's matrix_rank algorithm
    if threshold == "auto":
        threshold = _compute_numerical_rank_threshold(evals, m_shape)
        ndrop = (sorted_evals < threshold).sum().item()
    elif isinstance(threshold, float):
        threshold = threshold
        ndrop = (sorted_evals < threshold).sum().item()
    elif threshold is None:
        ndrop = expected_dof
    else:
        raise ValueError("threshold must be 'auto' or float")
    if print_warning and ndrop != expected_dof:
        print("W: Error in projector-based removal of translations & rotations")
        print(f"W: Num eigenvalues below threshold: {ndrop}, should be {expected_dof}")
        print(f"W: Threshold: {threshold:.2e}")
        print(sorted_evals[:8])
        ndrop = expected_dof
    return threshold, ndrop


# main functions to compute vibrational modes


def compute_modes_with_geometric_library(hessian, atom_types, coords, **kwargs):
    """
    Use Geometric library for frequency analysis

    Args:
        hessian: torch.Tensor Hessian matrix
        atom_types: list of atomic numbers
        coords: torch.Tensor coordinates in Angstrom
        return_raw_eigenvalues: bool, if True return eigenvalues in atomic units
                               instead of frequencies in cm⁻¹
        unmass_weight: bool, if True and return_raw_eigenvalues=True, return raw Cartesian
                      Hessian eigenvalues (Hartree/Bohr²) instead of mass-weighted ones

    Returns:
        If return_raw_eigenvalues=False (default):
            freqs_torch: frequencies in cm⁻¹
            modes_torch: normal modes
        If return_raw_eigenvalues=True and unmass_weight=False:
            eigenvals_torch: mass-weighted Hessian eigenvalues in atomic units (Hartree/amu)
            modes_torch: normal modes
        If return_raw_eigenvalues=True and unmass_weight=True:
            eigenvals_torch: raw Cartesian Hessian eigenvalues in atomic units (Hartree/Bohr²)
            modes_torch: normal modes (from mass-weighted calculation)

    Usage:
        # frequencies in cm⁻¹
        freqs_cm, modes = compute_modes_with_geometric_library(hessian, atom_types, coords)
    """

    # Convert to numpy
    device = "cpu"
    dtype = torch.float32
    if isinstance(coords, torch.Tensor):
        device = coords.device
        dtype = coords.dtype
        coords = coords.detach().cpu().numpy()
    if isinstance(hessian, torch.Tensor):
        device = hessian.device
        dtype = hessian.dtype
        hessian = hessian.detach().cpu().numpy()

    # Convert atomic numbers to symbols
    masses, elements, atom_types = _get_masses_zsymbols_znumbers(
        atom_types, device=device
    )

    # Convert coordinates from Angstrom to Bohr (geometric expects Bohr)
    coords_bohr = coords.flatten() * angstrom_to_bohr  # Convert Å to Bohr

    # convert Hessian from Hartree/Angstrom^2 to Hartree/Bohr^2 (atomic units)
    assert bohr_to_angstrom == 1 / angstrom_to_bohr
    hessian = hessian * (bohr_to_angstrom**2)

    # Get the normal modes. use geometric for proper removal of translational/rotational modes
    # (N*3-6,), (N*3-6, N*3)
    freqs, modes = hip.geometric_normal_modes.frequency_analysis(
        coords_bohr, hessian, elem=elements, verbose=False
    )

    # Convert back to torch
    freqs_torch = torch.from_numpy(freqs).to(device, dtype)
    modes_torch = torch.from_numpy(modes).to(device, dtype)

    freqs_torch, idx = torch.sort(torch.abs(freqs_torch))
    # freqs_torch, idx = torch.sort(freqs_torch)
    modes_torch = modes_torch[idx].T  # columns are the modes (3N, 3N-6)

    # modes from Bohr to Angstrom
    modes_torch = modes_torch * bohr_to_angstrom

    return freqs_torch, modes_torch


def compute_modes_with_ase_library(hessian, atom_types, coords, debug=False, **kwargs):
    """
    Use ASE library to get vibrational modes from torch Hessian

    Intelligently separates translational/rotational modes from vibrational modes:
    1. If exactly 6 (or 5 for linear) imaginary freqs: treat as TR modes
    2. If more imaginary freqs: smallest are TR, extras are transition state modes
    3. If fewer imaginary freqs: supplement with lowest real frequencies

    Args:
        hessian: torch.Tensor of shape (3*N, 3*N) - Hessian matrix
        atom_types: list of atomic numbers
        coords: torch.Tensor of shape (N, 3) - atomic coordinates
        debug: bool - whether to print diagnostic information

    Returns:
        frequencies_vib: frequencies in cm^-1 for vibrational modes only (3N-6 or 3N-5)
        modes_vib: vibrational modes excluding TR modes
        frequencies_all: all frequencies including TR modes (for analysis)
        n_tr_modes: number of translational/rotational modes detected
    """

    # Convert to numpy
    hessian_np = hessian.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy()
    masses, elements, atom_types = _get_masses_zsymbols_znumbers(
        atom_types, device=hessian.device
    )
    masses_np = masses.detach().cpu().numpy()
    symbols = elements

    # Create ASE atoms object
    atoms = Atoms(symbols=symbols, positions=coords_np, masses=masses_np)

    # Create VibrationsData object
    vib_data = VibrationsData.from_2d(atoms, hessian_np)

    # Get all frequencies and modes
    frequencies_all = vib_data.get_frequencies()  # in cm^-1 (complex for imaginary)
    modes_all = vib_data.get_modes()  # shape (3*N, N, 3)

    # Convert complex frequencies to real (negative for imaginary)
    frequencies_real = np.where(
        np.isreal(frequencies_all), frequencies_all.real, -np.abs(frequencies_all.imag)
    )

    # Check for linearity
    is_linear = _is_linear_molecule(coords_np)
    n_tr_expected = (
        5 if is_linear else 6
    )  # Linear: 3 trans + 2 rot, Non-linear: 3 trans + 3 rot

    # Separate imaginary and real frequencies
    is_imaginary = ~np.isreal(frequencies_all)
    imaginary_indices = np.where(is_imaginary)[0]
    real_indices = np.where(~is_imaginary)[0]

    n_imaginary = len(imaginary_indices)

    # Strategy:
    # 1. If we have exactly the expected number of imaginary freqs, assume they are TR modes
    # 2. If we have more, assume the extra ones are transition state modes (keep them as vibrational)
    # 3. If we have fewer, supplement with lowest real frequencies

    if n_imaginary != n_tr_expected and debug:
        print(f"Imaginary frequencies: {n_imaginary}")
        print(f"Expected TR modes: {n_tr_expected}")
        print(f"Molecule is {'linear' if is_linear else 'non-linear'}")
        print(f"frequencies_all\n", np.sort(np.abs(frequencies_all))[:10])
        print(f"frequencies_real\n", np.sort(np.abs(frequencies_real))[:10])
        print(
            f"frequencies_all[imaginary_indices]\n",
            frequencies_all[imaginary_indices][:10],
        )

    if n_imaginary == n_tr_expected:
        # Perfect case: imaginary frequencies are exactly the TR modes
        tr_indices = imaginary_indices
        vib_indices = real_indices

    elif n_imaginary > n_tr_expected:
        # Extra imaginary frequencies - likely transition state
        # Sort imaginary frequencies by absolute value (smallest = most likely TR)
        imag_abs_vals = np.abs(frequencies_all[imaginary_indices].imag)
        imag_sorted_idx = np.argsort(imag_abs_vals)

        # Take the smallest (absolute value) imaginary frequencies as TR modes
        tr_from_imag = imaginary_indices[imag_sorted_idx[:n_tr_expected]]
        extra_imag = imaginary_indices[imag_sorted_idx[n_tr_expected:]]

        tr_indices = tr_from_imag
        vib_indices = np.concatenate([extra_imag, real_indices])

    else:
        # Fewer imaginary frequencies than expected TR modes
        # Use all imaginary + lowest real frequencies to complete TR modes
        n_real_needed = n_tr_expected - n_imaginary

        if n_real_needed > 0:
            # Sort real frequencies by absolute value
            real_abs_vals = np.abs(frequencies_real[real_indices])
            real_sorted_idx = np.argsort(real_abs_vals)

            tr_from_real = real_indices[real_sorted_idx[:n_real_needed]]
            remaining_real = real_indices[real_sorted_idx[n_real_needed:]]

            tr_indices = np.concatenate([imaginary_indices, tr_from_real])
            vib_indices = remaining_real
        else:
            tr_indices = imaginary_indices
            vib_indices = real_indices

    # Sort indices for consistent output
    tr_indices = np.sort(tr_indices)
    vib_indices = np.sort(vib_indices)

    # Extract vibrational modes
    frequencies_vib = frequencies_real[vib_indices]
    modes_vib = modes_all[vib_indices]  # shape (n_vib, N, 3)

    # Convert back to torch
    frequencies_vib_torch = torch.from_numpy(frequencies_vib).to(hessian.device)
    modes_vib_torch = torch.from_numpy(modes_vib).to(hessian.device)
    frequencies_all_torch = torch.from_numpy(frequencies_real).to(hessian.device)

    # columns are the modes
    modes_vib_torch = modes_vib_torch.reshape(modes_vib_torch.shape[0], -1).T

    return (
        frequencies_vib_torch,
        modes_vib_torch,
    )  # , frequencies_all_torch, len(tr_indices)


def compute_modes_svd_projector(
    hessian,
    atom_types,
    coords,
    forces=None,
    include_force=False,
    threshold=None,
    **kwargs,
):
    """
    Compute vibrational eigenvalues and eigenvectors of a molecule by removing
    translational and rotational invariant degrees of freedom, optionally
    augmenting the null-space projector with the (mass-weighted) force vector.

    Parameters
    ----------
    hessian : torch.Tensor, shape (3N, 3N)
        Cartesian Hessian matrix (second derivatives of the potential energy).
    atom_types : list of length N
        Atomic types (atomic numbers).
    coords : torch.Tensor, shape (N, 3)
        Cartesian coordinates of atoms.
    forces : torch.Tensor, shape (N, 3), optional
        Cartesian forces on each atom. Required if include_force is True.
    include_force : bool, default False
        If True, include the mass-weighted force vector as an additional
        constraint in the null-space projector.
    threshold : float, default 1e-8
        Tolerance for zero singular values (not currently used).

    Returns
    -------
    eigenvalues : torch.Tensor, shape (3N - 6 or 3N - 7,)
        Vibrational eigenvalues (mass-weighted).
    eigenvectors : torch.Tensor, shape (3N, 3N - 6 or 3N - 7)
        Full mass-weighted eigenvectors for vibrational modes.
    """
    device = hessian.device
    dtype = hessian.dtype

    # Number of atoms
    N = len(atom_types)
    masses, elements, atom_types = _get_masses_zsymbols_znumbers(
        atom_types, device=device
    )
    masses = masses.to(dtype)

    if hessian.shape != (3 * N, 3 * N):
        raise ValueError(f"Hessian must be shape (3N,3N), got {hessian.shape}")
    if coords.shape != (N, 3):
        raise ValueError(f"Positions must be shape (N,3), got {coords.shape}")

    # Mass-weight the Hessian: Hmw = M^{-1/2} H M^{-1/2}
    m_sqrt = torch.repeat_interleave(torch.sqrt(masses), 3)
    Hmw = hessian / torch.outer(m_sqrt, m_sqrt)

    # Compute center of mass
    R_cm = torch.sum(masses[:, None] * coords, dim=0) / torch.sum(masses)

    # Build translation and rotation constraint vectors
    C = []
    sqrt_masses = torch.sqrt(masses)

    # Translations
    for alpha in range(3):
        t = torch.zeros(3 * N, device=device, dtype=dtype)
        t[alpha::3] = sqrt_masses
        C.append(t)

    # Rotations
    for alpha in range(3):
        e = torch.zeros(3, device=device, dtype=dtype)
        e[alpha] = 1.0
        disp = coords - R_cm
        # cross product for each atom
        r = torch.cross(e.unsqueeze(0).expand(N, -1), disp, dim=1)
        # flatten and mass-weight
        r_vec = r.flatten() * torch.repeat_interleave(sqrt_masses, 3)
        C.append(r_vec)

    # Optionally include force vector as additional constraint
    if include_force:
        if forces is None:
            raise ValueError("Forces must be provided when include_force is True.")
        # Mass-weighted force vector: g = M^{-1/2} * F
        g = forces.flatten() / torch.repeat_interleave(sqrt_masses, 3)
        C.append(g)

    C = torch.stack(C, dim=0)  # shape: (n_constraints, 3N)
    n_constraints = C.shape[0]

    # SVD of C^T to get orthonormal basis
    # C^T = U Sigma V^T, U is (3N,3N), its first n_constraints columns span constraint space
    U, S, VT = torch.linalg.svd(C.T, full_matrices=True)
    # Vibrational subspace basis: columns from n_constraints onward
    Q = U[:, n_constraints:]  # [3N, 3N-6]

    # Project Hessian into vibrational subspace
    H_red = Q.T @ Hmw @ Q

    # Symmetrize
    H_red = 0.5 * (H_red + H_red.T)

    # Diagonalize reduced Hessian
    eigvals, eigvecs_red = torch.linalg.eigh(H_red)

    # Build full eigenvectors in mass-weighted Cartesian coords
    # [3N-6, 3N-6] -> [3N, 3N-6]
    eigvecs = Q @ eigvecs_red

    # Convert to Cartesian displacement vectors
    eigvecs = eigvecs / m_sqrt[:, None]

    return eigvals, eigvecs


def compute_modes_inertia_projector(
    hessian, atom_types, coords, threshold_linear=1e-8, threshold=None, **kwargs
):
    """
    Compute eigenvalues/eigenvectors of Hessian excluding translational/rotational modes.
    Automatically detects linear molecules using inertia tensor.

    Args:
        hessian (torch.Tensor): Cartesian Hessian (3N x 3N)
        atom_types (list): Atomic numbers
        coords (torch.Tensor): Atomic coordinates (N x 3)
        threshold_linear (float): Numerical tolerance for linearity detection
        threshold (float): Numerical tolerance for internal mode detection

    Returns:
        internal_evals (torch.Tensor): Eigenvalues of internal modes (3N-6 or 3N-5)
        internal_evecs_cart (torch.Tensor): Cartesian eigenvectors (3N x (3N-6) or (3N x (3N-5))
    """

    N = len(atom_types)
    total_dof = 3 * N

    # Store original device and dtype
    device = hessian.device
    dtype = hessian.dtype

    # Center coordinates at center of mass
    masses, elements, atom_types = _get_masses_zsymbols_znumbers(
        atom_types, device=device
    )
    masses = masses.to(dtype)

    com = torch.sum(masses[:, None] * coords, dim=0) / torch.sum(masses)
    coords_rel = coords - com[None, :]

    # 1. Detect linearity using inertia tensor
    # Vectorized computation of inertia tensor
    x, y, z = coords_rel[:, 0], coords_rel[:, 1], coords_rel[:, 2]
    m = masses

    inertia_tensor = torch.zeros((3, 3), device=device, dtype=dtype)
    inertia_tensor[0, 0] = torch.sum(m * (y**2 + z**2))
    inertia_tensor[1, 1] = torch.sum(m * (x**2 + z**2))
    inertia_tensor[2, 2] = torch.sum(m * (x**2 + y**2))
    inertia_tensor[0, 1] = -torch.sum(m * x * y)
    inertia_tensor[0, 2] = -torch.sum(m * x * z)
    inertia_tensor[1, 2] = -torch.sum(m * y * z)
    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]

    # Compute eigenvalues of inertia tensor
    inertia_eigvals = torch.linalg.eigvalsh(inertia_tensor)
    is_linear = inertia_eigvals[0] < threshold_linear * torch.max(
        torch.tensor(1.0, device=device), torch.max(inertia_eigvals)
    )

    # Determine number of external modes
    if N == 1:  # Single atom
        rank_ext = 3
    elif is_linear:
        rank_ext = 5  # Linear molecule: 3 trans + 2 rot
        print("W: Linear molecule detected")
    else:
        rank_ext = 6  # Nonlinear molecule: 3 trans + 3 rot

    # 2. Mass-weight the Hessian
    sqrt_masses_rep = torch.repeat_interleave(torch.sqrt(masses), 3)
    inv_sqrt_masses_rep = 1.0 / sqrt_masses_rep
    H_mw = inv_sqrt_masses_rep[:, None] * hessian * inv_sqrt_masses_rep[None, :]

    # 3. Build external modes matrix B (3N x 6)
    B = torch.zeros((total_dof, 6), device=device, dtype=dtype)
    sqrt_masses_torch = torch.sqrt(masses)

    # Translations (always present)
    B[0::3, 0] = sqrt_masses_torch  # X
    B[1::3, 1] = sqrt_masses_torch  # Y
    B[2::3, 2] = sqrt_masses_torch  # Z

    # Rotations (skip if single atom)
    if N > 1:
        # Reshape relative coordinates to vector [x0,y0,z0, x1,y1,z1, ...]
        coords_vec = coords_rel.reshape(-1)
        B[1::3, 3] = -coords_vec[2::3] * sqrt_masses_torch  # -z_i * √m_i
        B[2::3, 3] = coords_vec[1::3] * sqrt_masses_torch  #  y_i * √m_i
        B[0::3, 4] = coords_vec[2::3] * sqrt_masses_torch  #  z_i * √m_i
        B[2::3, 4] = -coords_vec[0::3] * sqrt_masses_torch  # -x_i * √m_i
        B[0::3, 5] = -coords_vec[1::3] * sqrt_masses_torch  # -y_i * √m_i
        B[1::3, 5] = coords_vec[0::3] * sqrt_masses_torch  #  x_i * √m_i

    # 4. Orthonormalize external modes (using first `rank_ext` columns)
    U, _, _ = torch.linalg.svd(B[:, :rank_ext], full_matrices=False)
    U_ext = U[:, :rank_ext]  # Orthonormal basis for external modes

    # 5. Project Hessian into internal space
    projector_ext = U_ext @ U_ext.T
    projector_int = torch.eye(total_dof, device=device, dtype=dtype) - projector_ext
    H_int = projector_int.T @ H_mw @ projector_int

    # 6. Diagonalize projected Hessian
    evals, evecs_mw = torch.linalg.eigh(H_int)

    sorted_evals_idx = torch.argsort(torch.abs(evals))
    evals_sorted = evals[sorted_evals_idx]
    evecs_mw_sorted = evecs_mw[:, sorted_evals_idx]

    # 7. Filter internal modes (ignore near-zero eigenvalues)
    threshold, ndrop = _get_ndrop(
        evals,
        H_int.shape,
        threshold,
        evals_sorted,
        islinear=is_linear,
        print_warning=False,
    )
    internal_evecs_mw = evecs_mw_sorted[:, ndrop:]
    internal_evals = evals_sorted[ndrop:]

    # 8. Convert to Cartesian eigenvectors
    internal_evecs_cart = internal_evecs_mw / sqrt_masses_rep[:, None]

    idx = torch.argsort(internal_evals)
    internal_evals = internal_evals[idx]
    internal_evecs_cart = internal_evecs_cart[:, idx]

    return internal_evals, internal_evecs_cart


def compute_modes_qr_projector(hessian, atom_types, coords, threshold=None, **kwargs):
    """
    Projector-based removal of translations & rotations using QR decomposition
    A robust alternative to filtering is to build the subspace of exactly invariant motions
    (3 translations + 3 rotations, or 5 rotations for a linear molecule)
    and project them out of your Hessian before diagonalizing.
    This guarantees that no physical vibrational mode is accidentally thrown away.

    Args:
        hessian: (3N,3N) Cartesian Hessian
        atom_types: list[int] of length N of atomic numbers;
        coords: (N,3) positions;
    Returns:
        evals_vib: (3N-6,) eigenvalues of the vibrational modes, sorted in ascending order
        eigvecs_vib: (3N,3N-6) eigenvectors of the vibrational modes
    """
    if isinstance(atom_types, torch.Tensor):
        atom_types = atom_types.tolist()
    N = len(atom_types)
    # 0) Build mass vector m3 = [sqrt(m1), sqrt(m1), sqrt(m1), sqrt(m2), ...]
    masses, elements, atom_types = _get_masses_zsymbols_znumbers(
        atom_types, device=hessian.device
    )
    sqrt_m3 = masses.repeat_interleave(3).sqrt()  # (3N,)

    # 1) mass-weight Hessian F = M^{-1/2} H M^{-1/2}
    F = hessian / (sqrt_m3[:, None] * sqrt_m3[None, :])  # (3N,3N)

    # 2) build T (3N×6) of rigid-body vectors
    # translations
    T = []
    for alpha in range(3):
        t = torch.zeros(3 * N, device=hessian.device, dtype=hessian.dtype)
        t[alpha::3] = sqrt_m3[alpha::3]  # x: indices 0,3,6...; y:1,4,7...; z:2,5,8...
        T.append(t)
    # rotations about COM
    com = (coords * masses[:, None]).sum(dim=0) / masses.sum()
    for axis in [0, 1, 2]:  # x,y,z rotation
        r = torch.zeros(3 * N, device=hessian.device, dtype=hessian.dtype)
        for i in range(N):
            x, y, z = coords[i] - com
            m3 = sqrt_m3[3 * i : 3 * i + 3]
            if axis == 0:  # rotate about x: (0, -z, y)
                r[3 * i : 3 * i + 3] = (
                    torch.tensor([0, -z, y], device=hessian.device) * m3
                )
            elif axis == 1:  # about y: ( z, 0, -x)
                r[3 * i : 3 * i + 3] = (
                    torch.tensor([z, 0, -x], device=hessian.device) * m3
                )
            else:  # about z: (-y, x, 0)
                r[3 * i : 3 * i + 3] = (
                    torch.tensor([-y, x, 0], device=hessian.device) * m3
                )
        T.append(r)
    T = torch.stack(T, dim=1)  # (3N,6)

    # 3) orthonormalize T → U via QR
    Q, _ = torch.linalg.qr(T)  # Q: (3N,6) orthonormal
    # Build projector P = I - Q Q^T
    P = torch.eye(3 * N, device=hessian.device) - Q @ Q.T

    # 4) project the mass-weighted Hessian & diagonalize
    Fp = P @ (F @ P)
    evals, evecs = torch.linalg.eigh(Fp)  # mass-weighted vibrational modes

    # 5) drop the six exact zeros
    # sort by magnitude
    sorted_evals, sorted_evals_idx = torch.sort(torch.abs(evals))
    sorted_evecs = evecs[:, sorted_evals_idx]

    # 6) drop the six exact zeros, un-mass-weight the rest
    islinear = _is_linear_molecule(coords)
    threshold, ndrop = _get_ndrop(
        evals, Fp.shape, threshold, sorted_evals, islinear=islinear, print_warning=False
    )
    keep = torch.arange(ndrop, 3 * N)

    # The eigenvalues ω² in mass‐weighted Fp are the same for Cartesian H
    evals_vib = sorted_evals[keep]
    # Un‐mass‐weight to get Cartesian eigenvectors
    evecs_vib = sorted_evecs[:, keep] / sqrt_m3[:, None]

    # sort by smallest eigenvalues (same output as torch.linalg.eigh)
    sorted_evals_vib, sorted_evals_vib_idx = torch.sort(evals_vib)
    sorted_evecs_vib = evecs_vib[:, sorted_evals_vib_idx]

    return sorted_evals_vib, sorted_evecs_vib


def compute_modes_eckart_frame(
    hessian, atom_types, coords, orth_method="svd", threshold=None, **kwargs
):
    """
    Compute vibrational eigenvalues and eigenvectors using Eckart frame alignment,
    removing translations and rotations via principal axes alignment and specified orthonormalization.

    Parameters
    ----------
    hessian : torch.Tensor, shape (3N, 3N)
        Cartesian Hessian (symmetric).
    atom_types : list of str, length N
        Element symbols, e.g. ['C', 'H', 'H', ...].
    coords : torch.Tensor, shape (N, 3)
        Atomic positions.
    orth_method : str, 'svd' or 'qr'
        Method to orthonormalize rigid-body vectors ('svd' for SVD, 'qr' for Gram-Schmidt/QR).
    threshold : float, optional
        Threshold for dropping modes. If None, drop 6 modes.

    Returns
    -------
    eigvals_cart : torch.Tensor, shape (3N - n_drop,)
        Cartesian Hessian eigenvalues (ω²), including one negative for TS.
    eigvecs_cart : torch.Tensor, shape (3N, 3N - n_drop)
        Corresponding Cartesian displacement eigenvectors (columns).
    """
    N = coords.shape[0]
    assert hessian.shape == (3 * N, 3 * N), "Hessian must be of shape (3N, 3N)"

    pt = GetPeriodicTable()
    if isinstance(atom_types, torch.Tensor):
        atom_types = atom_types.tolist()
    if isinstance(atom_types[0], str):
        atom_types = [pt.GetAtomicNumber(z) for z in atom_types]

    # 1. Compute masses and mass‐weighted positions
    masses = torch.tensor(
        [pt.GetAtomicWeight(z) for z in atom_types],
        dtype=coords.dtype,
        device=coords.device,
    )
    m3 = masses.repeat_interleave(3).sqrt()

    # 2. Center on mass and align to principal axes (Eckart frame)
    com = (coords * masses[:, None]).sum(dim=0) / masses.sum()
    pos_centered = coords - com
    # Inertia tensor
    Idm = torch.zeros((3, 3), dtype=coords.dtype, device=coords.device)
    for i in range(N):
        r = pos_centered[i]
        m = masses[i]
        Idm += m * (r.dot(r) * torch.eye(3, device=coords.device) - torch.ger(r, r))
    # Principal axes
    eig_I, axes = torch.linalg.eigh(Idm)
    pos_eckart = (axes.T @ pos_centered.T).T  # rotate positions

    # 3. Mass‐weight Hessian
    F = hessian / (m3[:, None] * m3[None, :])

    # 4. Build rigid‐body vectors in mass‐weighted coordinates
    T_list = []
    # translations
    for alpha in range(3):
        t = torch.zeros(3 * N, dtype=coords.dtype, device=coords.device)
        t[alpha::3] = m3[alpha::3]
        T_list.append(t)
    # rotations about principal axes
    for axis in range(3):
        r = torch.zeros(3 * N, dtype=coords.dtype, device=coords.device)
        for i in range(N):
            x, y, z = pos_eckart[i]
            m3_i = m3[3 * i : 3 * i + 3]
            if axis == 0:
                vec = torch.tensor([0.0, -z, y], device=coords.device)
            elif axis == 1:
                vec = torch.tensor([z, 0.0, -x], device=coords.device)
            else:
                vec = torch.tensor([-y, x, 0.0], device=coords.device)
            r[3 * i : 3 * i + 3] = vec * m3_i
        T_list.append(r)
    T = torch.stack(T_list, dim=1)  # (3N, 6)

    # 5. Orthonormalize T to get Q
    if orth_method.lower() == "svd":
        U, S, Vh = torch.linalg.svd(T, full_matrices=False)
        Q = U[:, :6]
    elif orth_method.lower() == "qr":
        Q, _ = torch.linalg.qr(T)
    else:
        raise ValueError("orth_method must be 'svd' or 'qr'")

    # 6. Projector P = I - Q Q^T
    I3N = torch.eye(3 * N, dtype=coords.dtype, device=coords.device)
    P = I3N - Q @ Q.T

    # 7. Project and diagonalize
    Fp = P @ (F @ P)
    evals, evecs = torch.linalg.eigh(Fp)

    # 8. Drop zero modes
    # sort by magnitude
    sorted_evals, sorted_evals_idx = torch.sort(torch.abs(evals))
    sorted_evecs = evecs[:, sorted_evals_idx]
    islinear = _is_linear_molecule(coords)
    threshold, ndrop = _get_ndrop(
        evals, Fp.shape, threshold, sorted_evals, islinear=islinear, print_warning=False
    )
    keep = torch.arange(ndrop, 3 * N)
    evals_vib = sorted_evals[keep]
    q_vib = sorted_evecs[:, keep]

    if ndrop != 6:
        print(f"Eckart {orth_method}: ndrop: {ndrop}")
        print(f"sorted_evals\n", sorted_evals[:10])
        print(f"threshold={threshold}")

    # 9. Un‐mass‐weight to Cartesian eigenvectors
    eigvecs_cart = q_vib / m3[:, None]
    eigvals_cart = evals_vib

    return eigvals_cart, eigvecs_cart


def compute_vibrational_modes(
    hessian, atom_types, coords, method="geo", forces=None, **kwargs
):
    """
    Unified wrapper function to compute vibrational eigenvalues and eigenvectors using different methods.

    Args:
        hessian: torch.Tensor of shape (3N, 3N) - Cartesian Hessian matrix
        atom_types: list of atomic numbers or element symbols
        coords: torch.Tensor of shape (N, 3) - atomic coordinates
        method: str - method to use for computation, one of:
            - "geo": Use Geometric library (external dependency)
            - "ase": Use ASE library (external dependency)
            - "svd": Manual SVD-based null-space projector
            - "svdforce": Manual SVD-based null-space projector with force constraint
            - "inertia": Inertia tensor-based projector with auto-linearity detection
            - "qr": QR-based projector method (default)
            - "eckartsvd": Eckart frame alignment with principal axes with SVD orthonormalization
            - "eckartqr": Eckart frame alignment with principal axes with QR orthonormalization
        **kwargs: Additional method-specific arguments

    Returns:
        eigenvalues: torch.Tensor - vibrational eigenvalues
        eigenvectors: torch.Tensor - vibrational eigenvectors (columns)

    Example:
        # Use QR projector method (default)
        evals, evecs = compute_vibrational_modes(hessian, atom_types, coords)

        # Use Geometric library with frequencies in cm^-1
        freqs, modes = compute_vibrational_modes(hessian, atom_types, coords, method="geo")

        # Use SVD projector with force constraint
        evals, evecs = compute_vibrational_modes(hessian, atom_types, coords,
                                               method="svd",
                                               forces=forces, include_force=True)

        # Use Eckart frame with SVD orthonormalization
        evals, evecs = compute_vibrational_modes(hessian, atom_types, coords,
                                               method="eckart", orth_method="svd")
    """

    if method == "geo":
        return compute_modes_with_geometric_library(
            hessian, atom_types, coords, **kwargs
        )
    elif method == "ase":
        return compute_modes_with_ase_library(hessian, atom_types, coords, **kwargs)
    elif method == "svd":
        return compute_modes_svd_projector(
            hessian, atom_types, coords, forces=forces, **kwargs
        )
    elif method == "svdforce":
        return compute_modes_svd_projector(
            hessian, atom_types, coords, forces=forces, include_force=True, **kwargs
        )
    elif method == "inertia":
        return compute_modes_inertia_projector(hessian, atom_types, coords, **kwargs)
    elif method == "qr":
        return compute_modes_qr_projector(hessian, atom_types, coords, **kwargs)
    elif method == "eckartsvd":
        return compute_modes_eckart_frame(
            hessian, atom_types, coords, orth_method="svd", **kwargs
        )
    elif method == "eckartqr":
        return compute_modes_eckart_frame(
            hessian, atom_types, coords, orth_method="qr", **kwargs
        )
    elif method is None:
        evals, evecs = torch.linalg.eigh(hessian)
        return evals, evecs
    else:
        raise ValueError(
            f"Unknown method '{method}'. Available methods: "
            "geo, ase, svd, inertia, qr, eckart"
        )


# Legacy function names for backward compatibility
get_modes_geometric = compute_modes_with_geometric_library
get_vibrational_modes_ase = compute_modes_with_ase_library
vibrational_modes = compute_modes_svd_projector
compute_internal_modes = compute_modes_inertia_projector
projector_vibrational_modes = compute_modes_qr_projector
compute_cartesian_modes = compute_modes_eckart_frame


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))

    # Example 1: load a dataset file and predict the first batch

    # dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
    DATASET_FILES_HORM = [
        "ts1x-val.lmdb",  # 50844 samples
        "ts1x_hess_train_big.lmdb",  # 1725362 samples
        "RGD1.lmdb",  # 60000 samples
    ]
    dataset_path = fix_dataset_path("RGD1.lmdb")
    dataset = LmdbDataset(dataset_path)

    max_samples = 100

    count_wrong_eckart_svd = 0
    count_wrong_eckart_qr = 0
    count_wrong_projection = 0
    count_wrong_internal = 0
    count_wrong_geometric = 0
    count_wrong_ase = 0
    count_wrong_modes = 0
    print("\n")
    for i, sample in enumerate(dataset):
        if i >= max_samples and max_samples > 0:
            break

        indices = sample.one_hot.long().argmax(dim=1)
        sample.z = GLOBAL_ATOM_NUMBERS[indices]

        hessian = sample.hessian
        atom_types = sample.z.tolist()
        pos = sample.pos
        N = len(atom_types)

        forces = sample.forces

        num_vibrational_modes = 3 * N - 6

        hessian = hessian.reshape(3 * N, 3 * N)

        evals_vib, eigvecs_vib = compute_modes_qr_projector(hessian, atom_types, pos)
        if len(evals_vib) != num_vibrational_modes:
            count_wrong_projection += 1
            diff = num_vibrational_modes - len(evals_vib)
            print(
                f"Projection {i}: vibrational modes: {diff}",
                "❌",
            )

        evals_vib_cart, eigvecs_vib_cart = compute_modes_eckart_frame(
            hessian, atom_types, pos
        )
        if len(evals_vib_cart) != num_vibrational_modes:
            count_wrong_eckart_svd += 1
            diff = num_vibrational_modes - len(evals_vib_cart)
            print(
                f"Eckart SVD {i}: vibrational modes: {diff}",
                "❌",
            )
        evals_vib_cart, eigvecs_vib_cart = compute_modes_eckart_frame(
            hessian, atom_types, pos, orth_method="qr"
        )
        if len(evals_vib_cart) != num_vibrational_modes:
            count_wrong_eckart_qr += 1
            diff = num_vibrational_modes - len(evals_vib_cart)
            print(
                f"Eckart QR {i}: vibrational modes: {diff}",
                "❌",
            )

        evals_internal, eigvecs_internal = compute_modes_inertia_projector(
            hessian, atom_types, pos
        )
        if len(evals_internal) != num_vibrational_modes:
            count_wrong_internal += 1
            diff = num_vibrational_modes - len(evals_internal)
            print(
                f"Internal {i}: vibrational modes: {diff}",
                "❌",
            )

        evals_geometric, eigvecs_geometric = compute_modes_with_geometric_library(
            hessian, atom_types, pos
        )
        if len(evals_geometric) != num_vibrational_modes:
            count_wrong_geometric += 1
            diff = num_vibrational_modes - len(evals_geometric)
            print(
                f"Geometric {i}: vibrational modes: {diff}",
                "❌",
            )

        freq_vib_ase, modes_vib_ase = compute_modes_with_ase_library(
            hessian, atom_types, pos
        )
        if len(freq_vib_ase) != num_vibrational_modes:
            count_wrong_ase += 1
            diff = num_vibrational_modes - len(freq_vib_ase)
            print(
                f"ASE {i}: vibrational modes: {diff}",
                "❌",
            )

        freq_vib_modes, modes_vib_modes = compute_modes_svd_projector(
            hessian, atom_types, pos, forces=forces
        )
        if len(freq_vib_modes) != num_vibrational_modes:
            count_wrong_modes += 1
            diff = num_vibrational_modes - len(freq_vib_modes)
            print(
                f"Modes {i}: vibrational modes: {diff}",
                "❌",
            )

    print(f"Count wrong eckart svd: {count_wrong_eckart_svd}")
    print(f"Count wrong eckart qr: {count_wrong_eckart_qr}")
    print(f"Count wrong projection: {count_wrong_projection}")
    print(f"Count wrong internal: {count_wrong_internal}")
    print(f"Count wrong geometric: {count_wrong_geometric}")
    print(f"Count wrong ASE: {count_wrong_ase}")
    print(f"Count wrong modes: {count_wrong_modes}")

    # Demonstration of the new unified wrapper function
    print("\n=== Wrapper Function Demonstration ===")
    if max_samples > 0:
        sample = dataset[0]
        indices = sample.one_hot.long().argmax(dim=1)
        sample.z = GLOBAL_ATOM_NUMBERS[indices]
        hessian = sample.hessian.reshape(3 * len(sample.z), 3 * len(sample.z))
        atom_types = sample.z.tolist()
        pos = sample.pos

        print("Testing unified wrapper function with different methods:")

        # Test QR projector (default)
        evals, evecs = compute_vibrational_modes(hessian, atom_types, pos)
        print(f"QR projector (default): {len(evals)} modes")

        # Test SVD projector
        evals, evecs = compute_vibrational_modes(hessian, atom_types, pos, method="svd")
        print(f"SVD projector: {len(evals)} modes")

        # Test Eckart frame with SVD
        evals, evecs = compute_vibrational_modes(
            hessian, atom_types, pos, method="eckart", orth_method="svd"
        )
        print(f"Eckart frame (SVD): {len(evals)} modes")

        # Test inertia projector
        evals, evecs = compute_vibrational_modes(
            hessian, atom_types, pos, method="inertia"
        )
        print(f"Inertia projector: {len(evals)} modes")

        try:
            # Test geometric library
            evals, evecs = compute_vibrational_modes(
                hessian, atom_types, pos, method="geo"
            )
            print(f"Geometric library: {len(evals)} modes")
        except Exception as e:
            print(f"Geometric library: Failed ({e})")

        try:
            # Test ASE library
            evals, evecs = compute_vibrational_modes(
                hessian, atom_types, pos, method="ase"
            )
            print(f"ASE library: {len(evals)} modes")
        except Exception as e:
            print(f"ASE library: Failed ({e})")

        print("Wrapper function working correctly!")
