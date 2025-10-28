"""
ASE Calculator wrapper for Equiformer model.
"""

from typing import Optional
import numpy as np
import os

import torch

from torch_geometric.data import Data as TGData
from torch_geometric.data import Batch as TGBatch

from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.calculator import Calculator as ASECalculator

# from ocpmodels.datasets import data_list_collater
# from ocpmodels.preprocessing import AtomsToGraphs

from hip.hessian_utils import compute_hessian
from hip.inference_utils import get_model_from_checkpoint, get_dataloader
from hip.frequency_analysis import (
    analyze_frequencies_np,
    massweigh_and_eckartprojection_np,
)


def ase_atoms_to_torch_geometric(atoms):
    """
    Convert ASE Atoms object to torch_geometric Data format expected by Equiformer.

    Args:
        atoms: ASE Atoms object

    Returns:
        Data: torch_geometric Data object with required attributes
    """
    positions = atoms.get_positions().astype(np.float32)
    atomic_nums = atoms.get_atomic_numbers()

    # Convert to torch tensors
    data = TGData(
        pos=torch.tensor(positions, dtype=torch.float32),
        z=torch.tensor(atomic_nums, dtype=torch.int64),
        charges=torch.tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64),
        cell=torch.tensor(atoms.get_cell().astype(np.float32), dtype=torch.float32),
        pbc=torch.tensor(False, dtype=torch.bool),
    )
    return TGBatch.from_data_list([data])


class EquiformerASECalculator(ASECalculator):
    """
    Equiformer ASE Calculator.

    Might need to reimplement EquiformerASECalculator based on:
    ocpmodels/common/relaxation/ase_utils.py

    Args:
        checkpoint_path: Path to the Equiformer model checkpoint
        device: Optional device specification (defaults to auto-detect)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        hessian_method: str = "predict",
        model: torch.nn.Module = None,
        **kwargs,
    ):
        """
        Initialize the Equiformer calculator.

        Args:
            checkpoint_path: Path to the trained Equiformer checkpoint file
            device: Optional device specification (defaults to auto-detect)
            hessian_method: Method to compute the Hessian
            model: Optional model (otherwise loaded from checkpoint)
            **kwargs: Additional keyword arguments for parent Calculator class
        """
        ASECalculator.__init__(self, **kwargs)

        # this is where all the calculated properties are stored
        self.results = {}

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model is None:
            # Load model
            model = get_model_from_checkpoint(checkpoint_path, device)

        self.potential = model

        # Set implemented properties
        # # standard properties: ‘energy’, ‘forces’, ‘stress’, ‘dipole’, ‘charges’, ‘magmom’ and ‘magmoms’.
        self.implemented_properties = ["energy", "forces", "hessian"]

        self.hessian_method = hessian_method

    def reset(self):
        """Reset the calculator."""
        self.results = {}

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
        hessian_method=None,
    ):
        """
        Calculate properties for the given atoms.

        You can get the

        Args:
            atoms: ASE Atoms object
            properties: List of properties to compute (used by ASE internally)
            system_changes: System changes since last calculation (used by ASE internally)
        """
        # Call base class to set atoms attribute and manage caching
        ASECalculator.calculate(self, atoms, properties, system_changes)

        if hessian_method is None:
            hessian_method = self.hessian_method

        # ocpmodels/common/relaxation/ase_utils.py
        # data_object = self.a2g.convert(atoms)
        # batch = data_list_collater([data_object], otf_graph=True)

        # Convert ASE atoms to torch_geometric format
        # adds graph and Hessian indices
        batch = ase_atoms_to_torch_geometric(atoms)
        batch = batch.to(self.device)

        # Store results
        self.results = {}

        # Run prediction
        if hessian_method == "autograd":
            # Compute energy and forces with autograd
            with torch.enable_grad():
                batch.pos.requires_grad = True
                energy, forces, _ = self.potential.forward(
                    batch,
                    otf_graph=True,
                )
                # Use autograd to compute hessian
                hessian = compute_hessian(
                    coords=batch.pos,
                    energy=energy,
                    forces=forces,  # allow_unused=True
                )

        elif hessian_method == "predict":
            with torch.no_grad():
                energy, forces, out = self.potential.forward(
                    batch,
                    otf_graph=True,
                )
                hessian = out["hessian"]

        else:
            raise ValueError(f"Invalid hessian method: {hessian_method}")

        N = batch.pos.shape[0]
        self.results["hessian"] = hessian.detach().cpu().numpy().reshape(N * 3, N * 3)

        # Energy is per molecule, extract scalar value
        self.results["energy"] = float(energy.detach().cpu().item())

        # Forces shape: [n_atoms, 3]
        self.results["forces"] = forces.detach().cpu().numpy()

    def get_energy(self, atoms) -> float:
        """Return energy using ASE's property caching."""
        return self.get_property("energy", atoms)

    def get_potential_energy(self, atoms) -> float:
        return self.get_property("energy", atoms)

    def get_forces(self, atoms) -> np.ndarray:
        """Return forces using ASE's property caching."""
        return self.get_property("forces", atoms)

    def get_hessian(self, atoms, hessian_method=None) -> np.ndarray:
        """Return Hessian; optionally set method before retrieving."""
        if hessian_method is not None:
            self.hessian_method = hessian_method
        return self.get_property("hessian", atoms)

    def get_results(self, atoms) -> dict:
        """Return all results."""
        return self.results


##############################################################################################################


if __name__ == "__main__":
    # Create a simple water molecule for testing
    atoms = Atoms(
        symbols="H2O",
        positions=[
            [0.0, 0.0, 0.0],  # O
            [0.0, 0.757, 0.587],  # H
            [0.0, -0.757, 0.587],  # H
        ],
    )

    # Initialize calculator with default checkpoint path
    project_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_path = os.path.join(project_root, "ckpt/hesspred_v1.ckpt")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please provide a valid checkpoint path")
        exit()

    calculator = EquiformerASECalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
    )

    # Attach calculator to atoms
    atoms.calc = calculator

    # Calculate energy and forces
    calculator.calculate(atoms)
    results = calculator.results
    energy = results["energy"]
    forces = results["forces"]

    print(f"Energy: {energy} eV")
    print(f"Forces shape: {forces.shape}")
    print(f"Forces:\n{forces}")

    # To get hessian, we need to explicitly calculate it through the calculator
    calculator.calculate(atoms, properties=["energy", "forces", "hessian"])
    hessian = calculator.results["hessian"]
    print(f"Hessian shape: {hessian.shape}")

    # Or all at once
    print(f"Results: {results.keys()}")

    # Analyze frequencies
    frequency_analysis = analyze_frequencies_np(hessian, atoms.positions, atoms.symbols)
    print(f"eigvals: {frequency_analysis['eigvals'].shape}")
    print(f"eigvecs: {frequency_analysis['eigvecs'].shape}")
    print(f"neg_num: {frequency_analysis['neg_num']}")
    print(f"natoms: {frequency_analysis['natoms']}")
