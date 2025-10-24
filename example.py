import os
import torch
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.inference_utils import get_dataloader
from hip.frequency_analysis import analyze_frequencies_torch, analyze_frequencies_np
from hip.equiformer_ase_calculator import EquiformerASECalculator
from ase import Atoms


def torch_example(checkpoint_path, dataset_path, device):
    print("\n", "=" * 20 + " Torch Calculator " + "=" * 20, "\n")
    torch_calculator = EquiformerTorchCalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
        device=device,
    )

    dataloader = get_dataloader(
        dataset_path, torch_calculator.potential, batch_size=1, shuffle=False
    )
    batch = next(iter(dataloader))
    results = torch_calculator.predict(batch)
    print("Example 1:")
    print(f"  Energy: {results['energy'].shape}")
    print(f"  Forces: {results['forces'].shape}")
    print(f"  Hessian: {results['hessian'].shape}")

    # Example 2: create a random data object with random positions and predict
    n_atoms = 10
    elements = torch.tensor([1, 6, 7, 8])  # H, C, N, O
    pos = torch.randn(n_atoms, 3)  # (N, 3)
    atomic_nums = elements[torch.randint(0, 4, (n_atoms,))]  # (N,)
    results = torch_calculator.predict(coords=pos, atomic_nums=atomic_nums)
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


def ase_example(checkpoint_path, device):
    # Let's try the ASE calculator
    print("\n", "=" * 20 + " ASE Calculator " + "=" * 20, "\n")
    ase_calculator = EquiformerASECalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
        device=device,
    )
    # Create a simple water molecule for testing
    atoms = Atoms(
        symbols="H2O",
        positions=[
            [0.0, 0.0, 0.0],  # O
            [0.0, 0.757, 0.587],  # H
            [0.0, -0.757, 0.587],  # H
        ],
    )

    # Attach calculator to atoms
    atoms.calc = ase_calculator

    # Calculate energy and forces
    ase_calculator.calculate(atoms)
    results = ase_calculator.results
    energy = results["energy"]
    forces = results["forces"]

    print(f"Energy: {energy} eV")
    print(f"Forces shape: {forces.shape}")
    print(f"Forces:\n{forces}")

    # To get hessian, we need to explicitly calculate it through the calculator
    ase_calculator.calculate(atoms, properties=["energy", "forces", "hessian"])
    hessian = ase_calculator.results["hessian"]
    print(f"Hessian shape: {hessian.shape}")

    # Or all at once
    print(f"Results: {results.keys()}")

    # Analyze frequencies
    frequency_analysis = analyze_frequencies_np(hessian, atoms.positions, atoms.symbols)
    print(f"eigvals: {frequency_analysis['eigvals'].shape}")
    print(f"eigvecs: {frequency_analysis['eigvecs'].shape}")
    print(f"neg_num: {frequency_analysis['neg_num']}")
    print(f"natoms: {frequency_analysis['natoms']}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # you might need to change the directory
    project_root = os.path.dirname(__file__)
    checkpoint_path = os.path.join(project_root, "ckpt/hip_v2.ckpt")

    # Example 1: load a dataset file and predict the first batch
    dataset_path = os.path.join(project_root, "data/sample_100.lmdb")

    torch_example(checkpoint_path, dataset_path, device)
    ase_example(checkpoint_path, device)
