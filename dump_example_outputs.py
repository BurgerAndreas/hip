import argparse
import os

import torch
from ase import Atoms

from hip.equiformer_ase_calculator import EquiformerASECalculator
from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.inference_utils import get_dataloader


def _to_cpu_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="example_outputs.pt")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(__file__)
    checkpoint_path = args.checkpoint or os.path.join(project_root, "ckpt/hip_v2.ckpt")
    dataset_path = args.dataset or os.path.join(project_root, "data/sample_100.lmdb")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch_calculator = EquiformerTorchCalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
        device=device,
    )
    dataloader = get_dataloader(
        dataset_path, torch_calculator.potential, batch_size=1, shuffle=False
    )
    batch = next(iter(dataloader))
    torch_results = torch_calculator.predict(batch=batch)

    ase_calculator = EquiformerASECalculator(
        checkpoint_path=checkpoint_path,
        hessian_method="predict",
        device=device,
    )
    atoms = Atoms(
        symbols="H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.587],
            [0.0, -0.757, 0.587],
        ],
    )
    atoms.calc = ase_calculator
    ase_calculator.calculate(atoms, properties=["energy", "forces", "hessian"])
    ase_results = ase_calculator.results

    payload = {
        "device": device,
        "checkpoint_path": checkpoint_path,
        "dataset_path": dataset_path,
        "torch_dataset": {
            "energy": _to_cpu_tensor(torch_results["energy"]),
            "forces": _to_cpu_tensor(torch_results["forces"]),
            "hessian": _to_cpu_tensor(torch_results["hessian"]),
            "batch_pos": batch.pos.detach().cpu(),
            "batch_z": batch.z.detach().cpu(),
        },
        "ase_water": {
            "energy": _to_cpu_tensor(ase_results["energy"]),
            "forces": _to_cpu_tensor(ase_results["forces"]),
            "hessian": _to_cpu_tensor(ase_results["hessian"]),
        },
    }
    torch.save(payload, args.output)
    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    main()
