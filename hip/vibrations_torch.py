"""Compute Hessian matrices using finite differences with PyTorch Geometric.

This module provides a function-based API for computing Hessian matrices
using finite differences, working directly with PyTorch models and tensors.
All computation stays in-memory using torch tensors.
"""

from typing import Optional, Iterator, Tuple, List
import torch
from pathlib import Path
import fcntl
from torch_geometric.data import Data as TGData
from torch_geometric.data import Batch as TGBatch


def _create_batch(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    device: torch.device,
) -> TGBatch:
    """Create torch_geometric Batch from positions and atomic numbers.

    Args:
        positions: Tensor of shape [n_atoms, 3]
        atomic_numbers: Tensor of shape [n_atoms]
        device: torch device

    Returns:
        Batch: torch_geometric Batch object on the specified device
    """
    data = TGData(
        pos=positions.to(device),
        z=atomic_numbers.to(device),
        charges=atomic_numbers.to(device),
        natoms=torch.tensor([len(atomic_numbers)], dtype=torch.int64, device=device),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool, device=device),
    )
    return TGBatch.from_data_list([data])


def _get_forces(
    model: torch.nn.Module,
    batch: TGBatch,
) -> torch.Tensor:
    """Call model.forward() and extract forces.

    Args:
        model: PyTorch model with forward() method
        batch: torch_geometric Batch object

    Returns:
        forces: Tensor of shape [n_atoms, 3]
    """
    with torch.no_grad():
        _, forces, _ = model.forward(batch, otf_graph=True)
    return forces


def _get_forces_batched(
    model: torch.nn.Module,
    batch: TGBatch,
) -> List[torch.Tensor]:
    """Call model.forward() on a batched input and extract forces for each graph.

    Args:
        model: PyTorch model with forward() method
        batch: torch_geometric Batch object containing multiple graphs

    Returns:
        forces_list: List of force tensors, one per graph in the batch
    """
    with torch.no_grad():
        _, forces, _ = model.forward(batch, otf_graph=True)

    # Extract forces for each graph in the batch
    # batch.batch contains the graph index for each atom
    forces_list = []
    for i in range(batch.num_graphs):
        mask = batch.batch == i
        forces_list.append(forces[mask])

    return forces_list


def compute_hessian_finite_difference(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    indices: Optional[torch.Tensor] = None,
    delta: float = 0.01,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute Hessian matrix using finite differences.

    Uses central difference method to compute the Hessian matrix by
    displacing atoms and computing force differences. All computation
    is performed in-memory using torch tensors.

    Args:
        positions: Tensor of shape [n_atoms, 3] with atomic positions
        atomic_numbers: Tensor of shape [n_atoms] with atomic numbers
        model: PyTorch model with forward() method that returns (energy, forces, out)
        device: torch device to perform computation on
        indices: Optional tensor of atom indices to include in Hessian.
            If None, all atoms are included. Shape: [n_selected_atoms]
        delta: Displacement magnitude for finite differences (default: 0.01)
        dtype: torch.dtype to use for the computation

    Returns:
        hessian: Hessian matrix tensor of shape [n_dof, n_dof] where n_dof = 3 * n_selected_atoms.
    """

    positions = positions.to(device)
    atomic_numbers = atomic_numbers.long().to(device)

    n_atoms = len(positions)

    # Determine indices (all atoms if None)
    if indices is None:
        indices = torch.arange(n_atoms, device=device)
    else:
        if not isinstance(indices, torch.Tensor):
            raise TypeError(f"indices must be torch.Tensor, got {type(indices)}")
        indices = indices.long().to(device)

    n_dof = 3 * len(indices)

    # Initialize Hessian tensor on device
    H = torch.zeros((n_dof, n_dof), dtype=dtype, device=device)

    # Loop over atoms and coordinates
    r = 0
    for a in indices:
        a = a.item()  # Convert to Python int for indexing
        for i in range(3):  # x, y, z
            # Create displaced positions for -delta
            positions_minus = positions.clone()
            positions_minus[a, i] -= delta
            batch_minus = _create_batch(positions_minus, atomic_numbers, device)
            fminus = _get_forces(model, batch_minus)

            # Create displaced positions for +delta
            positions_plus = positions.clone()
            positions_plus[a, i] += delta
            batch_plus = _create_batch(positions_plus, atomic_numbers, device)
            fplus = _get_forces(model, batch_plus)

            # Central difference formula matching vibrations.py line 392:
            # H[r] = 0.5 * (fminus - fplus)[indices].ravel() / (2 * delta)
            force_diff = fminus - fplus
            # Select forces for the indices we care about, then flatten
            selected_forces = force_diff[indices]
            H[r] = 0.5 * selected_forces.ravel() / (2 * delta)
            r += 1

    # Symmetrize the Hessian (matching vibrations.py line 406: H += H.copy().T)
    H = H + H.T

    return H


def compute_hessian_finite_difference_batched(
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    indices: Optional[torch.Tensor] = None,
    delta: float = 0.01,
    batch_size: int = 32,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute Hessian matrix using finite differences with torch_geometric batching.

    Uses central difference method to compute the Hessian matrix by
    displacing atoms and computing force differences. All computation
    is performed in-memory using torch tensors, with multiple displacements
    processed simultaneously using torch_geometric batching for efficiency.

    Args:
        positions: Tensor of shape [n_atoms, 3] with atomic positions
        atomic_numbers: Tensor of shape [n_atoms] with atomic numbers
        model: PyTorch model with forward() method that returns (energy, forces, out)
        device: torch device to perform computation on
        indices: Optional tensor of atom indices to include in Hessian.
            If None, all atoms are included. Shape: [n_selected_atoms]
        delta: Displacement magnitude for finite differences (default: 0.01)
        batch_size: Number of displacements to process in each batch (default: 32)

    Returns:
        hessian: Hessian matrix tensor of shape [n_dof, n_dof] where n_dof = 3 * n_selected_atoms.
    """
    positions = positions.to(device)
    atomic_numbers = atomic_numbers.long().to(device)

    n_atoms = len(positions)

    # Determine indices (all atoms if None)
    if indices is None:
        indices = torch.arange(n_atoms, device=device)
    else:
        if not isinstance(indices, torch.Tensor):
            raise TypeError(f"indices must be torch.Tensor, got {type(indices)}")
        indices = indices.long().to(device)

    n_dof = 3 * len(indices)

    # Collect all displacement configurations
    displacement_configs = []
    for a in indices:
        a = a.item()
        for i in range(3):
            for sign in [-1, 1]:
                displacement_configs.append((a, i, sign))

    # Dictionary to store forces for each displacement
    forces_dict = {}

    # Process displacements in batches
    for batch_start in range(0, len(displacement_configs), batch_size):
        batch_end = min(batch_start + batch_size, len(displacement_configs))
        batch_configs = displacement_configs[batch_start:batch_end]

        # Create batched data for this batch of displacements
        batch_data_list = []
        for atom_idx, coord_idx, sign in batch_configs:
            positions_disp = positions.clone()
            positions_disp[atom_idx, coord_idx] += sign * delta
            data = TGData(
                pos=positions_disp.to(device),
                z=atomic_numbers.to(device),
                charges=atomic_numbers.to(device),
                natoms=torch.tensor(
                    [len(atomic_numbers)], dtype=torch.int64, device=device
                ),
                cell=None,
                pbc=torch.tensor(False, dtype=torch.bool, device=device),
            )
            batch_data_list.append(data)

        # Create batch and compute forces
        batch = TGBatch.from_data_list(batch_data_list)
        forces_list = _get_forces_batched(model, batch)

        # Store forces in dictionary
        for (atom_idx, coord_idx, sign), forces in zip(batch_configs, forces_list):
            forces_dict[(atom_idx, coord_idx, sign)] = forces

    # Compute Hessian using central differences
    H = torch.zeros((n_dof, n_dof), dtype=torch.float32, device=device)

    # Map from (atom_idx, coord_idx) to row index in Hessian
    row_map = {}
    r = 0
    for a in indices:
        a = a.item()
        for i in range(3):
            row_map[(a, i)] = r
            r += 1

    # Compute Hessian rows using central differences
    for a in indices:
        a = a.item()
        for i in range(3):
            fminus = forces_dict[(a, i, -1)]
            fplus = forces_dict[(a, i, 1)]

            # Central difference formula
            force_diff = fminus - fplus
            selected_forces = force_diff[indices]
            row_idx = row_map[(a, i)]
            H[row_idx] = 0.5 * selected_forces.ravel() / (2 * delta)

    # Symmetrize the Hessian (matching vibrations.py line 406: H += H.copy().T)
    H = H + H.T

    return H


if __name__ == "__main__":
    from hip.ff_lmdb import LmdbDataset, Z_TO_ATOM_SYMBOL
    from hip.path_config import DATA_PATH_HORM_SAMPLE
    from hip.inference_utils import get_model_from_checkpoint
    import os
    import time
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = os.path.dirname(os.path.dirname(__file__))
    # project_root = os.path.dirname(__file__)
    checkpoint_path = os.path.join(project_root, "ckpt/hip_v2.ckpt")
    model = get_model_from_checkpoint(checkpoint_path, device)

    dtype = torch.float32

    # Load dataset and sample
    dataset = LmdbDataset(DATA_PATH_HORM_SAMPLE)
    sample = dataset[0]

    # Extract positions and atomic numbers as torch tensors
    if torch.is_tensor(sample["pos"]):
        positions = sample["pos"]
    else:
        positions = torch.tensor(sample["pos"], dtype=dtype)

    if torch.is_tensor(sample["z"]):
        atomic_numbers = sample["z"]
    else:
        atomic_numbers = torch.tensor(sample["z"], dtype=torch.int64)

    positions_ase = sample["pos"].numpy().copy()
    atomic_numbers_ase = sample["z"].numpy().copy()

    print(f"Number of atoms: {len(atomic_numbers)}")
    print(f"Number of degrees of freedom: {3 * len(atomic_numbers)}")
    print(f"Device: {device}")
    print()

    repeats = 5

    # do 10 forward passes to warm up the model
    print("Warming up the model...")
    for _ in range(repeats):
        model.forward(_create_batch(positions, atomic_numbers, device))
    print("Model warmed up")
    print()

    # Example 1: Sequential finite difference
    print("=" * 60)
    print("Example 1: Sequential finite difference")
    print("=" * 60)
    times1 = []
    rel_diffs1 = []
    abs_diffs1 = []
    prev_hessian = None
    for _ in range(repeats):
        positions_clone = positions.clone()
        atomic_numbers_clone = atomic_numbers.clone()
        start_time = time.time()
        hessian1 = compute_hessian_finite_difference(
            positions=positions_clone,
            atomic_numbers=atomic_numbers_clone,
            model=model,
            device=device,
            indices=None,  # All atoms
            delta=0.01,
        )
        times1.append(time.time() - start_time)
        if prev_hessian is not None:
            mean_diff = (hessian1 - prev_hessian).abs().mean().item()
            print(f"Abs difference: {mean_diff:.2e}")
            print(
                f"Relative difference: {((hessian1 - prev_hessian) / prev_hessian).abs().mean().item():.2e}"
            )
            rel_diffs1.append(
                ((hessian1 - prev_hessian) / prev_hessian).abs().mean().item()
            )
            abs_diffs1.append((hessian1 - prev_hessian).abs().mean().item())
        prev_hessian = hessian1

    print(f"Hessian shape: {hessian1.shape}")
    print(f"Hessian min/max: {hessian1.min().item():.6f} / {hessian1.max().item():.6f}")
    print(f"Time elapsed: {np.mean(times1):.2f} seconds")
    print(f"Abs difference: {np.mean(abs_diffs1):.2e}")
    print(f"Relative difference: {np.mean(rel_diffs1):.2e}")
    print()

    # Example 2: Batched finite difference
    print("=" * 60)
    print("Example 2: Batched finite difference (batch_size=32)")
    print("=" * 60)
    times2 = []
    rel_diffs2 = []
    abs_diffs2 = []
    prev_hessian = None
    for _ in range(repeats):
        positions_clone = positions.clone()
        atomic_numbers_clone = atomic_numbers.clone()
        start_time = time.time()
        hessian2 = compute_hessian_finite_difference_batched(
            positions=positions_clone,
            atomic_numbers=atomic_numbers_clone,
            model=model,
            device=device,
            indices=None,  # All atoms
            delta=0.01,
            batch_size=32,
        )
        elapsed_time2 = time.time() - start_time
        times2.append(elapsed_time2)
        if prev_hessian is not None:
            mean_diff = (hessian2 - prev_hessian).abs().mean().item()
            print(f"Abs difference: {mean_diff:.2e}")
            print(
                f"Relative difference: {((hessian2 - prev_hessian) / prev_hessian).abs().mean().item():.2e}"
            )
            rel_diffs2.append(
                ((hessian2 - prev_hessian) / prev_hessian).abs().mean().item()
            )
            abs_diffs2.append((hessian2 - prev_hessian).abs().mean().item())
        prev_hessian = hessian2

    print(f"Hessian min/max: {hessian2.min().item():.6f} / {hessian2.max().item():.6f}")
    print(f"Time elapsed: {np.mean(times2):.2f} seconds")
    print(f"Abs difference: {np.mean(abs_diffs2):.2e}")
    print(f"Relative difference: {np.mean(rel_diffs2):.2e}")
    print()

    # Comparison
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Sequential time: {np.mean(times1):.2f} seconds")
    print(f"Batched time:    {np.mean(times2):.2f} seconds")
    speedup = np.mean(times1) / np.mean(times2) if np.mean(times2) > 0 else float("inf")
    print(f"Speedup:         {speedup:.2f}x")
    print(f"Time difference: {np.mean(times1) - np.mean(times2):.2f} seconds")

    # Verify results match
    max_diff = (hessian1 - hessian2).abs().max().item()
    mean_diff = (hessian1 - hessian2).abs().mean().item()
    print(f"Max difference:  {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    are_close = torch.allclose(hessian1, hessian2, rtol=1e-4, atol=1e-5)
    print(f"Results match:   {are_close}")

    # ground truth
    print("\n")

    from hip.equiformer_ase_calculator import EquiformerASECalculator

    calc = EquiformerASECalculator(
        # checkpoint_path=checkpoint_path,
        # hessian_method="predict",
    )

    from hip.finite_difference_ase import get_finite_difference_hessian
    from ase import Atoms

    times_ase = []
    rel_diffs_ase = []
    abs_diffs_ase = []
    prev_hessian = None
    for _ in range(repeats):
        atoms = Atoms(
            symbols=[Z_TO_ATOM_SYMBOL[int(z)] for z in atomic_numbers_ase],
            positions=positions_ase.copy(),
        )
        atoms.calc = calc
        start_time = time.time()
        hessian_ase = get_finite_difference_hessian(atoms)
        elapsed_time_ase = time.time() - start_time
        times_ase.append(elapsed_time_ase)
        if prev_hessian is not None:
            mean_diff = np.abs((hessian_ase - prev_hessian)).mean().item()
            print(f"Abs difference: {mean_diff:.2e}")
            print(
                f"Relative difference: {np.abs((hessian_ase - prev_hessian) / prev_hessian).mean().item():.2e}"
            )
            rel_diffs_ase.append(
                np.abs((hessian_ase - prev_hessian) / prev_hessian).mean().item()
            )
            abs_diffs_ase.append(mean_diff)
        prev_hessian = hessian_ase

    print(f"Hessian shape: {hessian_ase.shape}")
    print(
        f"Hessian min/max: {hessian_ase.min().item():.6f} / {hessian_ase.max().item():.6f}"
    )
    print(f"Time elapsed: {np.mean(times_ase):.2f} seconds")
    print(f"Abs difference: {np.mean(abs_diffs_ase):.2e}")
    print(f"Relative difference: {np.mean(rel_diffs_ase):.2e}")
    print()

    # print the first 6x6 of the hessian
    print(
        f"First 6x6 of the ground truth Hessian:\n{torch.from_numpy(hessian_ase[:6, :6])}"
    )
    print(f"First 6x6 of the sequential Hessian:\n{hessian1[:6, :6]}")
    print(f"First 6x6 of the batched Hessian:\n{hessian2[:6, :6]}")
