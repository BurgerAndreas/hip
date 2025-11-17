import torch
import argparse
import numpy as np
from tqdm import tqdm
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch_geometric.loader import DataLoader as TGDataLoader

# from alphanet.models.alphanet import AlphaNet
# from leftnet.model.leftnet import LEFTNet

from hip.training_module import PotentialModule
from hip.ff_lmdb import LmdbDataset, Z_TO_ATOM_SYMBOL
from hip.path_config import fix_dataset_path
from hip.qm9_hessian_dataset import QM9HessianDataset

from hip.frequency_analysis import analyze_frequencies_np, eigval_to_wavenumber
from pathlib import Path


def find_checkpoint(checkpoint_path):
    """
    Find checkpoint path. If the provided path doesn't exist, search for it
    in checkpoint/hip/*<arg>*/last.ckpt pattern.

    Args:
        checkpoint_path: Original checkpoint path or search pattern (e.g., "1430809")

    Returns:
        Resolved checkpoint path
    """
    # If path exists, return as-is
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # Search in checkpoint/hip/ for directories containing the pattern
    checkpoint_dir = Path("checkpoint/hip")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint path {checkpoint_path} does not exist and "
            f"checkpoint directory {checkpoint_dir} not found"
        )

    # Find directories matching the pattern
    matching_dirs = [
        d for d in checkpoint_dir.iterdir() if d.is_dir() and checkpoint_path in d.name
    ]

    if not matching_dirs:
        raise FileNotFoundError(
            f"Checkpoint path {checkpoint_path} does not exist and "
            f"no matching directory found in {checkpoint_dir} containing '{checkpoint_path}'"
        )

    if len(matching_dirs) > 1:
        print(
            f"Warning: Multiple matching directories found: {[d.name for d in matching_dirs]}"
        )
        print(f"Using: {matching_dirs[0].name}")

    # Look for last.ckpt in the matching directory
    found_ckpt = matching_dirs[0] / "last.ckpt"
    if not found_ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint path {checkpoint_path} does not exist and "
            f"last.ckpt not found in {matching_dirs[0]}"
        )

    print(f"Found checkpoint: {found_ckpt}")
    return str(found_ckpt)


def _get_derivatives(x, y, retain_graph=None, create_graph=False):
    """Helper function to compute derivatives"""
    grad = torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph
    )[0]
    return grad


def compute_hessian(coords, energy, forces=None):
    """Compute Hessian matrix using autograd."""
    # Compute forces if not given
    if forces is None:
        forces = -_get_derivatives(coords, energy, create_graph=True)

    # Get number of components (n_atoms * 3)
    n_comp = forces.reshape(-1).shape[0]

    # Initialize hessian
    hess = []
    for f in forces.reshape(-1):
        # Compute second-order derivative for each element
        hess_row = _get_derivatives(coords, -f, retain_graph=True)
        hess.append(hess_row)

    # Stack hessian
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


def evaluate(
    data_path,
    checkpoint_path,
    config_path,  # not used
    hessian_method,
    max_samples=None,
    wandb_run_id=None,
    wandb_kwargs={},
    redo=False,
):
    # Auto-find checkpoint if path doesn't exist
    checkpoint_path = find_checkpoint(checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model_config = ckpt["hyper_parameters"]["model_config"]
    print(f"Model name: {model_name}")

    # Get dataset from checkpoint if not provided
    if data_path is None:
        if "training_config" in ckpt.get("hyper_parameters", {}):
            training_config = ckpt["hyper_parameters"]["training_config"]
            data_path = training_config.get("val_path")
            if data_path is None:
                raise ValueError(
                    "Dataset not provided and val_path not found in checkpoint training_config"
                )
            print(f"Using dataset from checkpoint: {data_path}")
        else:
            raise ValueError(
                "Dataset not provided and training_config not found in checkpoint"
            )

    _name = ""
    # _name += checkpoint_path.split("/")[-2]
    _name += checkpoint_path.split("/")[-1].split(".")[0]
    _name += "_" + data_path.split("/")[-1].split(".")[0]
    if hessian_method != "autograd":
        _name += "_" + hessian_method
    _name += "_" + str(max_samples)

    if wandb_run_id is None:
        wandb.init(
            project="horm",
            name=_name,
            config={
                "checkpoint": checkpoint_path,
                "dataset": data_path,
                "max_samples": max_samples,
                "model_name": model_name,
                "config_path": config_path,
                "hessian_method": hessian_method,
                "model_config": model_config,
            },
            tags=["hormmetrics"],
            **wandb_kwargs,
        )

    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()

    do_autograd = hessian_method == "autograd"
    print(f"do_autograd: {do_autograd}")

    # Create results file path
    dataset_name = data_path.split("/")[-1].split(".")[0]
    results_dir = "results_evalhorm"
    os.makedirs(results_dir, exist_ok=True)
    ckpt_name = checkpoint_path.split("/")[-1].split(".")[0]
    results_file = (
        f"{results_dir}/{ckpt_name}_{dataset_name}_{hessian_method}_metrics.csv"
    )

    time_taken_all = None
    n_total_samples = None

    # Check if results already exist and redo is False
    if os.path.exists(results_file) and not redo:
        print(f"Loading existing results from {results_file}")
        df_results = pd.read_csv(results_file)

    else:
        torch.manual_seed(42)
        np.random.seed(42)

        # Get keep_fluorine from checkpoint if available
        keep_fluorine = False
        if "training_config" in ckpt.get("hyper_parameters", {}):
            keep_fluorine = ckpt["hyper_parameters"]["training_config"].get(
                "keep_fluorine", False
            )

        # Check if QM9 dataset
        if "qm9" in data_path.lower():
            # Path format: /path/to/dataset:split_name or just /path/to/dataset
            # The split can be specified in the path with a colon separator
            dataset = QM9HessianDataset(
                dataset_path=data_path,
                split="test",  # Use test split for evaluation
                keep_fluorine=keep_fluorine,
            )
        else:
            dataset = LmdbDataset(fix_dataset_path(data_path))
        dataloader = TGDataLoader(dataset, batch_size=1, shuffle=True)

        # Initialize metrics collection for per-sample DataFrame
        sample_metrics = []
        n_samples = 0

        if max_samples is not None:
            n_total_samples = min(max_samples, len(dataloader))
        else:
            n_total_samples = len(dataloader)

        # Warmup
        for _i, batch in tqdm(enumerate(dataloader), desc="Warmup", total=10):
            if _i >= 10:
                break
            batch = batch.to(device)

            n_atoms = batch.pos.shape[0]

            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            # Forward pass
            if model_name == "LEFTNet":
                batch.pos.requires_grad_()
                energy_model, force_model = model.forward_autograd(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)
            elif "equiformer" in model_name.lower():
                if do_autograd:
                    batch.pos.requires_grad_()
                    energy_model, force_model, out = model.forward(
                        batch, otf_graph=False, hessian=False
                    )
                    hessian_model = compute_hessian(
                        batch.pos, energy_model, force_model
                    )
                else:
                    with torch.no_grad():
                        energy_model, force_model, out = model.forward(
                            batch,
                            otf_graph=False,
                        )
                    hessian_model = out["hessian"].reshape(n_atoms * 3, n_atoms * 3)
            else:
                # AlphaNet
                batch.pos.requires_grad_()
                energy_model, force_model = model.forward(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)

        for batch in tqdm(dataloader, desc="Evaluating", total=n_total_samples):
            batch = batch.to(device)

            n_atoms = batch.pos.shape[0]

            # Collect per-sample metrics
            sample_data = {
                "sample_idx": n_samples,
                "natoms": n_atoms,
            }

            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            # Forward pass
            if model_name == "LEFTNet":
                batch.pos.requires_grad_()
                energy_model, force_model = model.forward_autograd(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)
            elif "equiformer" in model_name.lower():
                if do_autograd:
                    batch.pos.requires_grad_()
                    energy_model, force_model, out = model.forward(
                        batch, otf_graph=False, hessian=False
                    )
                    hessian_model = compute_hessian(
                        batch.pos, energy_model, force_model
                    )
                else:
                    with torch.no_grad():
                        energy_model, force_model, out = model.forward(
                            batch,
                            otf_graph=False,
                        )
                    hessian_model = out["hessian"]
            else:
                # AlphaNet
                batch.pos.requires_grad_()
                energy_model, force_model = model.forward(batch)
                hessian_model = compute_hessian(batch.pos, energy_model, force_model)

            end_event.record()
            torch.cuda.synchronize()

            time_taken = start_event.elapsed_time(end_event)  # ms
            memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
            sample_data["time"] = time_taken  # ms
            sample_data["memory"] = memory_usage

            hessian_model = hessian_model.reshape(n_atoms * 3, n_atoms * 3)

            # Compute hessian eigenspectra
            eigvals_model, eigvecs_model = torch.linalg.eigh(hessian_model)

            # Compute errors
            if "energy" in batch:  # RGD1 dataset
                energy_true = batch.energy
            else:  # T1x, QM9 dataset
                energy_true = batch.ae
            e_mae = torch.mean(
                torch.abs(energy_model.squeeze() - energy_true.squeeze())
            )
            e_mae_per_atom = e_mae / n_atoms
            sample_data["energy_mae"] = e_mae.item()
            sample_data["energy_mae_per_atom"] = e_mae_per_atom.item()
            f_mae = torch.mean(torch.abs(force_model - batch.forces))
            sample_data["forces_mae"] = f_mae.item()

            # Reshape true hessian
            n_atoms = batch.pos.shape[0]
            hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)
            h_mae = torch.mean(torch.abs(hessian_model - hessian_true))
            sample_data["hessian_mae"] = h_mae.item()

            # Eigenvalue error
            eigvals_true, eigvecs_true = torch.linalg.eigh(hessian_true)

            # Asymmetry error
            asymmetry_mae = torch.mean(torch.abs(hessian_model - hessian_model.T))
            true_asymmetry_mae = torch.mean(torch.abs(hessian_true - hessian_true.T))
            sample_data["asymmetry_mae"] = asymmetry_mae.item()
            sample_data["true_asymmetry_mae"] = true_asymmetry_mae.item()

            # Additional metrics
            eigval_mae = torch.mean(
                torch.abs(eigvals_model - eigvals_true)
            )  # eV/Angstrom^2
            sample_data["eigval_mae"] = eigval_mae.item()

            ########################
            # Mass weighted + Eckart projection
            ########################

            true_freqs = analyze_frequencies_np(
                hessian=hessian_true.detach().cpu().numpy(),
                cart_coords=batch.pos.detach().cpu().numpy(),
                atomsymbols=[Z_TO_ATOM_SYMBOL[z.item()] for z in batch.z],
            )
            true_neg_num = true_freqs["neg_num"]
            true_eigvecs_eckart = torch.tensor(true_freqs["eigvecs"])
            true_eigvals_eckart = torch.tensor(true_freqs["eigvals"])

            freqs_model = analyze_frequencies_np(
                hessian=hessian_model.detach().cpu().numpy(),
                cart_coords=batch.pos.detach().cpu().numpy(),
                atomsymbols=[Z_TO_ATOM_SYMBOL[z.item()] for z in batch.z],
            )
            freqs_model_neg_num = freqs_model["neg_num"]
            eigvecs_model_eckart = torch.tensor(freqs_model["eigvecs"])
            eigvals_model_eckart = torch.tensor(freqs_model["eigvals"])

            sample_data["true_neg_num"] = true_neg_num
            sample_data["true_is_minima"] = 1 if true_neg_num == 0 else 0
            sample_data["true_is_ts"] = 1 if true_neg_num == 1 else 0
            sample_data["true_is_ts_order2"] = 1 if true_neg_num == 2 else 0
            sample_data["true_is_higher_order"] = 1 if true_neg_num > 2 else 0
            sample_data["model_neg_num"] = freqs_model_neg_num
            sample_data["model_is_ts"] = 1 if freqs_model_neg_num == 1 else 0
            sample_data["model_is_minima"] = 1 if freqs_model_neg_num == 0 else 0
            sample_data["model_is_ts_order2"] = 1 if freqs_model_neg_num == 2 else 0
            sample_data["model_is_higher_order"] = 1 if freqs_model_neg_num > 2 else 0
            sample_data["neg_num_agree"] = (
                1 if (true_neg_num == freqs_model_neg_num) else 0
            )

            sample_data["eigval_mae_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart - true_eigvals_eckart)
            ).item()
            sample_data["eigval1_mae_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart[0] - true_eigvals_eckart[0])
            ).item()
            sample_data["eigval2_mae_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart[1] - true_eigvals_eckart[1])
            ).item()
            sample_data["eigvec1_cos_eckart"] = torch.abs(
                torch.dot(eigvecs_model_eckart[:, 0], true_eigvecs_eckart[:, 0])
            ).item()
            sample_data["eigvec2_cos_eckart"] = torch.abs(
                torch.dot(eigvecs_model_eckart[:, 1], true_eigvecs_eckart[:, 1])
            ).item()

            ########################
            # Vibrational frequencies for QM9 Hessian dataset
            ########################

            # Convert eigenvalues to wavenumbers and compute MAE for 400-4000 cm⁻¹ range
            true_eigvals_np = true_eigvals_eckart.detach().cpu().numpy()
            model_eigvals_np = eigvals_model_eckart.detach().cpu().numpy()

            # Convert to wavenumbers (cm⁻¹)
            true_wavenumbers = eigval_to_wavenumber(true_eigvals_np)
            model_wavenumbers = eigval_to_wavenumber(model_eigvals_np)

            # Filter for positive eigenvalues (real vibrational modes) in 400-4000 cm⁻¹ range
            # Since eigenvalues are sorted and correspond to the same modes, we can compare directly
            true_mask = (
                (true_wavenumbers >= 400)
                & (true_wavenumbers <= 4000)
                & (true_eigvals_np > 0)
            )
            model_mask = (
                (model_wavenumbers >= 400)
                & (model_wavenumbers <= 4000)
                & (model_eigvals_np > 0)
            )

            # Use intersection of masks to ensure we compare the same vibrational modes
            combined_mask = true_mask & model_mask

            if combined_mask.sum() > 0:
                true_filtered = true_wavenumbers[combined_mask]
                model_filtered = model_wavenumbers[combined_mask]
                freq_mae = np.mean(np.abs(model_filtered - true_filtered))
                sample_data["freq_mae_400_4000"] = freq_mae
            else:
                # No frequencies in range, set to NaN
                sample_data["freq_mae_400_4000"] = np.nan

            # Compare model frequencies to dataset frequencies (if available)
            if hasattr(batch, "frequencies") and batch.frequencies is not None:
                dataset_frequencies = batch.frequencies.detach().cpu().numpy()

                # Filter dataset frequencies to 400-4000 cm⁻¹ range (already in cm⁻¹)
                # Only consider positive frequencies (real vibrational modes)
                dataset_mask = (
                    (dataset_frequencies >= 400)
                    & (dataset_frequencies <= 4000)
                    & (dataset_frequencies > 0)
                )
                dataset_filtered = dataset_frequencies[dataset_mask]

                # Filter model wavenumbers to 400-4000 cm⁻¹ range
                model_mask_dataset = (
                    (model_wavenumbers >= 400)
                    & (model_wavenumbers <= 4000)
                    & (model_eigvals_np > 0)
                )
                model_filtered_dataset = model_wavenumbers[model_mask_dataset]

                # Compare frequencies - both arrays are sorted, so compare directly
                # Use minimum length to ensure we compare the same number of modes
                if len(dataset_filtered) > 0 and len(model_filtered_dataset) > 0:
                    min_len = min(len(dataset_filtered), len(model_filtered_dataset))
                    freq_mae_dataset = np.mean(
                        np.abs(
                            model_filtered_dataset[:min_len]
                            - dataset_filtered[:min_len]
                        )
                    )
                    sample_data["freq_mae_400_4000_dataset"] = freq_mae_dataset
                else:
                    sample_data["freq_mae_400_4000_dataset"] = np.nan
            else:
                sample_data["freq_mae_400_4000_dataset"] = np.nan

            sample_metrics.append(sample_data)
            n_samples += 1

            # Memory management
            torch.cuda.empty_cache()

            if max_samples is not None and n_samples >= max_samples:
                break

        # Create DataFrame from collected metrics
        df_results = pd.DataFrame(sample_metrics)

        # Save DataFrame
        df_results.to_csv(results_file, index=False)
        print(f"Saved results to {results_file}")

    # Compute aggregated results by looping over all numeric columns
    aggregated_results = {}
    for col in df_results.columns:
        if pd.api.types.is_numeric_dtype(df_results[col]):
            aggregated_results[col] = df_results[col].mean()
        else:
            print(
                f"Skipping column {col} because it is not numeric: {df_results[col].dtype}"
            )
            continue

    # Special case: is_ts_agree computed from comparing two columns
    if "model_is_ts" in df_results.columns and "true_is_ts" in df_results.columns:
        aggregated_results["is_ts_agree"] = (
            df_results["model_is_ts"] == df_results["true_is_ts"]
        ).mean()

    wandb.log(aggregated_results)

    if wandb_run_id is None:
        wandb.finish()

    return df_results, aggregated_results


def plot_accuracy_vs_natoms(df_results, name):
    """Plot accuracy metrics over number of atoms"""

    # Create figure with subplots
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 10))
    fig.suptitle("Model Accuracy vs Number of Atoms", fontsize=16)

    # Define metrics to plot and their labels
    metrics = [
        ("energy_mae", "Energy MAE", "Energy Error"),
        ("forces_mae", "Forces MAE", "Forces Error"),
        ("hessian_mae", "Hessian MAE", "Hessian Error"),
        ("eigvec1_cos", "Eigenvector 1 Cosine", "Eigenvector 1 Cosine"),
        ("eigval1_mae", "Eigenvalue 1 MAE", "Eigenvalue 1 MAE"),
        ("is_ts_agree", "Is TS Agree", "Is TS Agree"),
        ("neg_num_agree", "Neg Num Agree", "Neg Num Agree"),
        ("true_is_ts", "True Is TS", "True Is TS"),
        ("model_is_ts", "Model Is TS", "Model Is TS"),
    ]

    # Plot each metric
    for i, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[i // 2, i % 2]

        # Skip metrics not available in results
        if metric not in df_results.columns:
            ax.set_visible(False)
            continue

        # Group by natoms and calculate mean and std
        grouped = (
            df_results.groupby("natoms")[metric].agg(["mean", "std"]).reset_index()
        )

        # Plot mean with error bars
        ax.errorbar(
            grouped["natoms"],
            grouped["mean"],
            yerr=grouped["std"],
            marker="o",
            capsize=5,
            capthick=2,
            linewidth=2,
        )

        ax.set_xlabel("Number of Atoms")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Set log scale for y-axis if needed (based on data range)
        if grouped["mean"].max() / (grouped["mean"].min() + 1e-8) > 100:
            ax.set_yscale("log")

    plt.tight_layout()

    # Save plot
    plot_dir = "plots/eval_horm"
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = f"{plot_dir}/accuracy_vs_natoms_{name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {plot_filename}")

    # Show plot
    plt.show()


"""
uv run python scripts/eval.py -c ckpt/eqv2.ckpt -d RGD1.lmdb -m 1000 -r True -hm autograd
uv run python scripts/eval.py -c ckpt/eqv2.ckpt -d ts1x-val.lmdb -m 1000 -r True -hm autograd
uv run python scripts/eval.py -c ckpt/hesspred_v1.ckpt -d ts1x-val.lmdb -m 1000 -r True -hm predict
uv run python scripts/eval.py -c ckpt/hip_v2.ckpt -d ts1x-val.lmdb -m 1000 -r True -hm predict
uv run python scripts/eval.py -c ckpt/hip_v3.ckpt -d ts1x-val.lmdb -m 1000 -r True -hm predict
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HORM model on dataset")
    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        default="ckpt/eqv2.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config file. Ignored at the moment (config from ckpt is used instead).",
    )
    parser.add_argument(
        "--hessian_method",
        "-hm",
        choices=["autograd", "predict"],
        type=str,
        default="predict",
        help="Hessian computation method: autograd, predict",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Dataset file name (e.g., ts1x-val.lmdb, ts1x_hess_train_big.lmdb, RGD1.lmdb). If not provided, will use val_path from checkpoint.",
    )
    parser.add_argument(
        "--max_samples",
        "-m",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all samples)",
    )
    parser.add_argument(
        "--redo",
        "-r",
        type=bool,
        default=False,
        help="Run eval from scratch even if results already exist",
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    checkpoint_path = args.ckpt_path
    data_path = args.dataset
    max_samples = args.max_samples
    config_path = args.config_path
    hessian_method = args.hessian_method
    redo = args.redo

    # Name will be set after dataset is resolved in evaluate function
    name = f"{checkpoint_path.split('/')[-1].split('.')[0]}_{hessian_method}"
    if data_path is not None:
        name += f"_{data_path.split('/')[-1].split('.')[0]}"

    df_results, aggregated_results = evaluate(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        hessian_method=hessian_method,
        max_samples=max_samples,
        redo=redo,
    )

    # Plot accuracy over Natoms
    # plot_accuracy_vs_natoms(df_results, name)
