import torch
import numpy as np
import os
from torch_geometric.loader import DataLoader as TGDataLoader

from hip.training_module import PotentialModule
from hip.ff_lmdb import LmdbDataset
from hip.path_config import DATA_PATH_HORM_SAMPLE
from scripts.eval import compute_hessian


def predict_and_save_samples(
    checkpoint_path="ckpt/hip_v2.ckpt",
    dataset_path=None,
    num_samples=10,
    output_dir="pred_data",
):
    """
    Predict energy, force, and hessian for first N samples and save to .npz file.

    Args:
        checkpoint_path: Path to checkpoint file
        dataset_path: Path to dataset (defaults to DATA_PATH_HORM_SAMPLE)
        num_samples: Number of samples to process
        output_dir: Directory to save output files
    """
    if dataset_path is None:
        dataset_path = DATA_PATH_HORM_SAMPLE

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)

    module = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
        map_location=device,
    )
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    print(f"Model name: {model_name}")

    model = module.potential.to(device)
    model.eval()

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = LmdbDataset(dataset_path)
    dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing first {num_samples} samples...")
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        batch = batch.to(device)
        n_atoms = batch.pos.shape[0]

        # Extract true values
        if hasattr(batch, "energy") and batch.energy is not None:
            energy_true = batch.energy.squeeze().item()
        else:
            energy_true = batch.ae.squeeze().item()

        forces_true = batch.forces.detach().cpu().numpy()
        hessian_true = batch.hessian.detach().cpu().numpy()

        # Extract positions and atom types
        positions = batch.pos.detach().cpu().numpy()
        atom_types = batch.z.detach().cpu().numpy()

        # Forward pass to get predictions
        if model_name == "LEFTNet":
            batch.pos.requires_grad_()
            energy_model, force_model = model.forward_autograd(batch)
            hessian_model = compute_hessian(batch.pos, energy_model, force_model)
        elif "equiformer" in model_name.lower():
            # Use predict method (not autograd) for equiformer
            with torch.no_grad():
                energy_model, force_model, out = model.forward(
                    batch,
                    otf_graph=False,
                )
            hessian_model = out["hessian"]
        else:
            # AlphaNet or other models
            batch.pos.requires_grad_()
            energy_model, force_model = model.forward(batch)
            hessian_model = compute_hessian(batch.pos, energy_model, force_model)

        # Extract predicted values
        energy_pred = energy_model.squeeze().detach().cpu().item()
        forces_pred = force_model.detach().cpu().numpy()

        # Flatten hessian to 1D array
        hessian_model = hessian_model.reshape(-1)
        hessian_pred = hessian_model.detach().cpu().numpy()

        # Ensure true hessian is 1D
        hessian_true = hessian_true.reshape(-1)

        # Save each sample as its own .npz file
        output_file = os.path.join(output_dir, f"sample_{i:04d}.npz")
        np.savez(
            output_file,
            positions=positions,
            atom_types=atom_types,
            energy_true=energy_true,
            energy_pred=energy_pred,
            forces_true=forces_true,
            forces_pred=forces_pred,
            hessian_true=hessian_true,
            hessian_pred=hessian_pred,
        )

        print(
            f"Sample {i + 1}/{num_samples}: n_atoms={n_atoms}, energy_true={energy_true:.6f}, energy_pred={energy_pred:.6f}, saved to {output_file}"
        )

    print(f"Saved {num_samples} samples to {output_dir}/")


if __name__ == "__main__":
    predict_and_save_samples(
        checkpoint_path="ckpt/hip_v2.ckpt",
        num_samples=10,
        output_dir="pred_data",
    )
