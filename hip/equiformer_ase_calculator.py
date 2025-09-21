"""
ASE Calculator wrapper for Equiformer model.
"""

from typing import Optional
import numpy as np
import yaml
import os

import torch
from torch_geometric.data import Data as TGData
from torch_geometric.data import Batch

import ase
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.calculator import Calculator as ASECalculator
from ase.data import atomic_numbers
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.constraints import FixAtoms

from ocpmodels.datasets import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.common.relaxation.ase_utils import (
    # batch_to_atoms,
    # ase_atoms_to_torch_geometric,
    ase_atoms_to_torch_geometric_hessian,
    coord_atoms_to_torch_geometric_hessian,
)

from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from nets.prediction_utils import compute_extra_props

from hip.hessian_utils import compute_hessian
from hip.inference_utils import get_model_from_checkpoint, get_dataloader
from hip.frequency_analysis import analyze_frequencies, eckart_projection_notmw


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

    def forward(self, atoms, hessian=False):
        """
        Forward pass for the Equiformer calculator.
        If hessian is True, it will compute the Hessian via autograd and eigenvalues/eigenvectors.
        Otherwise, it will only compute the energy and forces.
        """
        properties = ["energy", "forces"]
        if hessian:
            properties += ["hessian", "eigen"]
        self.calculate(atoms, properties=properties)
        return self.results

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
        do_autograd = (hessian_method == "autodiff") and ("hessian" in properties)

        # ocpmodels/common/relaxation/ase_utils.py
        # data_object = self.a2g.convert(atoms)
        # batch = data_list_collater([data_object], otf_graph=True)

        # Convert ASE atoms to torch_geometric format
        # adds graph and Hessian indices
        batch = ase_atoms_to_torch_geometric_hessian(
            atoms,
            cutoff=self.potential.cutoff,
            max_neighbors=self.potential.max_neighbors,
            use_pbc=self.potential.use_pbc,
            with_grad=do_autograd,
        )
        batch = batch.to(self.device)

        if properties is None:
            properties = []

        # Store results
        self.results = {}

        # Prepare batch with extra properties
        batch = compute_extra_props(batch, pos_require_grad=do_autograd)

        # Run prediction
        if "hessian" in properties:
            if hessian_method == "autograd":
                # Compute energy and forces with autograd
                with torch.enable_grad():
                    # batch.pos.requires_grad = True # already set in ase_atoms_to_torch_geometric_hessian
                    energy, forces, _ = self.potential.forward(
                        batch,
                        otf_graph=False,  # TODO: does that work? we should have gradients from ase_atoms_to_torch_geometric_hessian
                        hessian=False,
                        add_props=False,
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
                        hessian=True,
                        otf_graph=False,
                        add_props=True,  # not necessary for single molecule (only needed for batching)
                    )
                    hessian = out["hessian"]

            else:
                raise ValueError(f"Invalid hessian method: {hessian_method}")

            N = batch.pos.shape[0]
            self.results["hessian"] = (
                hessian.detach().cpu().numpy().reshape(N * 3, N * 3)
            )

        else:
            # just predict energy and forces
            energy, forces, _ = self.potential.forward(
                batch,
                otf_graph=False,
                hessian=False,
                add_props=False,
            )

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
    results = calculator.get_energy(atoms)
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
    results = calculator.get_hessian(atoms)
    print(f"Results: {results.keys()}")

    # Analyze frequencies
    frequency_analysis = analyze_frequencies(hessian, atoms.positions, atoms.symbols)
    print(f"eigvals: {frequency_analysis['eigvals'].shape}")
    print(f"eigvecs: {frequency_analysis['eigvecs'].shape}")
    print(f"neg_num: {frequency_analysis['neg_num']}")
    print(f"natoms: {frequency_analysis['natoms']}")
