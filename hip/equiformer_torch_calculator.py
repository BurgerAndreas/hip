from typing import Optional
import os
import torch

from torch_geometric.data import Batch as TGBatch
from torch_geometric.data import Data as TGData
# from torch_geometric.loader import DataLoader as TGDataLoader


from hip.hessian_utils import compute_hessian
from hip.inference_utils import get_model_from_checkpoint, get_dataloader
from hip.frequency_analysis import (
    analyze_frequencies_torch,
    massweigh_and_eckartprojection_torch,
    get_trans_rot_projector_torch,
    massweigh_hessian_torch,
)
from hip.masses import MASS_DICT


def coord_atoms_to_torch_geometric(
    coords,  # (N, 3)
    atomic_nums,  # (N,)
):
    """
    Convert ASE Atoms object to torch_geometric Data format expected by Equiformer.
    with_grad=True ensures there are gradients of the energy and forces w.r.t. the positions,
    through the graph generation.

    Args:
        atoms: ASE Atoms object

    Returns:
        Data: torch_geometric Data object with required attributes
    """

    # Convert to torch tensors
    data = TGData(
        pos=torch.as_tensor(coords, dtype=torch.float32).reshape(-1, 3),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )
    return TGBatch.from_data_list(
        [data],
        # follow_batch=["diag_ij", "edge_index", "message_idx_ij"]
    )


class EquiformerTorchCalculator:
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
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model is None:
            model = get_model_from_checkpoint(checkpoint_path, device)

        self.potential = model

        self.hessian_method = hessian_method

    def to(self, device):
        self.device = device
        self.potential = self.potential.to(device)
        return self

    def predict(
        self,
        batch=None,
        coords=None,
        atomic_nums=None,
        hessian_method=None,
        do_hessian=True,
    ):
        """Predict one or multiple samples"""
        if hessian_method is None:
            hessian_method = self.hessian_method

        if batch is None:
            assert coords is not None and atomic_nums is not None, (
                "coords and atomic_nums must be provided if batch is not provided"
            )
            batch = coord_atoms_to_torch_geometric(
                coords,
                atomic_nums,
            )

        # Store results
        self.results = {}

        # Prepare batch with extra properties
        batch = batch.to(self.potential.device)

        # Run prediction
        if do_hessian:
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
            self.results["hessian"] = (
                hessian  # .reshape(N * 3, N * 3) # only reshape for a single molecule
            )

        else:
            # just predict energy and forces
            energy, forces, _ = self.potential.forward(
                batch,
                otf_graph=False,
                hessian=False,
            )

        # Energy is per molecule, extract scalar value
        self.results["energy"] = energy.detach()

        # Forces shape: [n_atoms, 3]
        self.results["forces"] = forces.detach()

        return self.results

    def get_gad(self, batch, hessian_method=None):
        """
        Gentlest Ascent Dynamics (GAD)
        dx/dt = -∇V(x) + 2(∇V, v(x))v(x)
        = F + 2(-F, v(x))v(x)
        since F=-∇V(x)
        where v(x) is the eigenvector of the Hessian with the smallest eigenvalue.

        eckart: bool, whether to use Eckart projection to remove redundant translations and rotations
        """
        assert batch.batch.max() + 1 == 1, (
            "Only one batch is supported for GAD prediction"
        )
        results = self.predict(batch, hessian_method=hessian_method)
        N = batch.pos.shape[0]
        hessian = results["hessian"].reshape(N * 3, N * 3)
        forces = results["forces"].reshape(-1)  # N*3
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        # eigval1 = eigenvalues[0]
        # eigval_prod = eigenvalues[0] * eigenvalues[1]
        v = eigenvectors[:, 0]
        # -∇V(x) + 2(∇V, v(x))v(x)
        grad = -forces
        # gad = -grad + 2 * torch.einsum("i,i->", grad, v) * v
        gad = -grad + 2 * torch.dot(grad, v) * v
        results.update(
            {
                "gad": gad,
            }
        )
        return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # you might need to change this
    project_root = os.path.dirname(os.path.dirname(__file__))
    checkpoint_path = os.path.join(project_root, "ckpt/hesspred_v1.ckpt")
    calculator = EquiformerTorchCalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
    )

    # Example 1: load a dataset file and predict the first batch
    dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
    dataloader = get_dataloader(
        dataset_path, calculator.potential, batch_size=1, shuffle=False
    )
    batch = next(iter(dataloader))
    results = calculator.predict(batch)
    print("\nExample 1:")
    print(f"  Energy: {results['energy'].shape}")
    print(f"  Forces: {results['forces'].shape}")
    print(f"  Hessian: {results['hessian'].shape}")

    print("\nGAD:")
    gad = calculator.get_gad(batch)
    print(f"  GAD: {gad['gad'].shape}")

    # Example 2: create a random data object with random positions and predict
    n_atoms = 10
    elements = torch.tensor([1, 6, 7, 8])  # H, C, N, O
    pos = torch.randn(n_atoms, 3)  # (N, 3)
    atomic_nums = elements[torch.randint(0, 4, (n_atoms,))]  # (N,)
    results = calculator.predict(coords=pos, atomic_nums=atomic_nums)
    print("\nExample 2:")
    print(f"  Energy: {results['energy'].shape}")
    print(f"  Forces: {results['forces'].shape}")
    print(f"  Hessian: {results['hessian'].shape}")

    print("\nFrequency analysis:")
    hessian = results["hessian"]
    frequency_analysis = analyze_frequencies_torch(hessian, pos, atomic_nums)
    print(f"eigvals: {frequency_analysis['eigvals'].shape}")
    print(f"eigvecs: {frequency_analysis['eigvecs'].shape}")
    print(f"neg_num: {frequency_analysis['neg_num']}")
    print(f"natoms: {frequency_analysis['natoms']}")
