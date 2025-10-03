import torch
import argparse
import numpy as np
import torch
from tqdm import tqdm
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch_geometric.loader import DataLoader as TGDataLoader

from alphanet.models.alphanet import AlphaNet
from leftnet.model.leftnet import LEFTNet

from hip.training_module import PotentialModule
from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path
from nets.prediction_utils import compute_extra_props, Z_TO_ATOM_SYMBOL

from hip.frequency_analysis import analyze_frequencies


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
    lmdb_path,
    checkpoint_path,
    config_path,  # not used
    hessian_method,
    max_samples=None,
    wandb_run_id=None,
    wandb_kwargs={},
    redo=False,
):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model_config = ckpt["hyper_parameters"]["model_config"]
    print(f"Model name: {model_name}")

    _name = ""
    # _name += checkpoint_path.split("/")[-2]
    _name += checkpoint_path.split("/")[-1].split(".")[0]
    # _name += "_" + lmdb_path.split("/")[-1].split(".")[0]
    if hessian_method != "autograd":
        _name += "_" + hessian_method
    _name += "_" + str(max_samples)

    if wandb_run_id is None:
        wandb.init(
            project="horm",
            name=_name,
            config={
                "checkpoint": checkpoint_path,
                "dataset": lmdb_path,
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
    ).potential.to("cuda")
    model.eval()

    do_autograd = hessian_method == "autograd"
    print(f"do_autograd: {do_autograd}")

    # Create results file path
    dataset_name = lmdb_path.split("/")[-1].split(".")[0]
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

        dataset = LmdbDataset(fix_dataset_path(lmdb_path))
        # dataset = LmdbDataset(fix_dataset_path(lmdb_path))
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
            batch = batch.to("cuda")
            batch = compute_extra_props(batch)

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

        start_event_all = torch.cuda.Event(enable_timing=True)
        end_event_all = torch.cuda.Event(enable_timing=True)
        start_event_all.record()

        for batch in tqdm(dataloader, desc="Evaluating", total=n_total_samples):
            batch = batch.to("cuda")
            batch = compute_extra_props(batch)

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

            hessian_model = hessian_model.reshape(n_atoms * 3, n_atoms * 3)

            # Compute hessian eigenspectra
            eigvals_model, eigvecs_model = torch.linalg.eigh(hessian_model)

            # Compute errors
            e_error = torch.mean(torch.abs(energy_model.squeeze() - batch.ae))
            f_error = torch.mean(torch.abs(force_model - batch.forces))

            # Reshape true hessian
            n_atoms = batch.pos.shape[0]
            hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)
            h_error = torch.mean(torch.abs(hessian_model - hessian_true))

            # Eigenvalue error
            eigvals_true, eigvecs_true = torch.linalg.eigh(hessian_true)

            # Asymmetry error
            asymmetry_error = torch.mean(torch.abs(hessian_model - hessian_model.T))
            true_asymmetry_error = torch.mean(torch.abs(hessian_true - hessian_true.T))

            # Additional metrics
            eigval_mae = torch.mean(
                torch.abs(eigvals_model - eigvals_true)
            )  # eV/Angstrom^2
            eigval1_mae = torch.mean(torch.abs(eigvals_model[0] - eigvals_true[0]))
            eigval2_mae = torch.mean(torch.abs(eigvals_model[1] - eigvals_true[1]))
            eigvec1_mae = torch.mean(
                torch.abs(eigvecs_model[:, 0] - eigvecs_true[:, 0])
            )
            eigvec2_mae = torch.mean(
                torch.abs(eigvecs_model[:, 1] - eigvecs_true[:, 1])
            )
            eigvec1_cos = torch.abs(torch.dot(eigvecs_model[:, 0], eigvecs_true[:, 0]))
            eigvec2_cos = torch.abs(torch.dot(eigvecs_model[:, 1], eigvecs_true[:, 1]))

            # Collect per-sample metrics
            sample_data = {
                "sample_idx": n_samples,
                "natoms": n_atoms,
                "energy_error": e_error.item(),
                "forces_error": f_error.item(),
                "hessian_error": h_error.item(),
                "asymmetry_error": asymmetry_error.item(),
                "true_asymmetry_error": true_asymmetry_error.item(),
                "eigval_mae": eigval_mae.item(),
                "eigval1_mae": eigval1_mae.item(),
                "eigval2_mae": eigval2_mae.item(),
                "eigvec1_mae": eigvec1_mae.item(),
                "eigvec2_mae": eigvec2_mae.item(),
                "eigvec1_cos": eigvec1_cos.item(),
                "eigvec2_cos": eigvec2_cos.item(),
                "time": time_taken,  # ms
                "memory": memory_usage,
            }

            ########################
            # Mass weighted + Eckart projection
            ########################

            true_freqs = analyze_frequencies(
                hessian=hessian_true.detach().cpu().numpy(),
                cart_coords=batch.pos.detach().cpu().numpy(),
                atomsymbols=[Z_TO_ATOM_SYMBOL[z.item()] for z in batch.z],
            )
            true_neg_num = true_freqs["neg_num"]
            true_eigvecs_eckart = torch.tensor(true_freqs["eigvecs"])
            true_eigvals_eckart = torch.tensor(true_freqs["eigvals"])

            freqs_model = analyze_frequencies(
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
            )
            sample_data["eigval1_mae_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart[0] - true_eigvals_eckart[0])
            )
            sample_data["eigval2_mae_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart[1] - true_eigvals_eckart[1])
            )
            sample_data["eigvec1_mae_eckart"] = torch.mean(
                torch.abs(eigvecs_model_eckart[:, 0] - true_eigvecs_eckart[:, 0])
            )
            sample_data["eigvec2_mae_eckart"] = torch.mean(
                torch.abs(eigvecs_model_eckart[:, 1] - true_eigvecs_eckart[:, 1])
            )
            sample_data["eigvec1_cos_eckart"] = torch.abs(
                torch.dot(eigvecs_model_eckart[:, 0], true_eigvecs_eckart[:, 0])
            )
            sample_data["eigvec2_cos_eckart"] = torch.abs(
                torch.dot(eigvecs_model_eckart[:, 1], true_eigvecs_eckart[:, 1])
            )

            sample_metrics.append(sample_data)
            n_samples += 1

            # Memory management
            torch.cuda.empty_cache()

            if max_samples is not None and n_samples >= max_samples:
                break

        end_event_all.record()
        torch.cuda.synchronize()

        time_taken_all = start_event_all.elapsed_time(end_event_all)  # ms

        # Create DataFrame from collected metrics
        df_results = pd.DataFrame(sample_metrics)

        # Save DataFrame
        df_results.to_csv(results_file, index=False)
        print(f"Saved results to {results_file}")

    aggregated_results = {
        "energy_mae": df_results["energy_error"].mean(),
        "forces_mae": df_results["forces_error"].mean(),
        "hessian_mae": df_results["hessian_error"].mean(),
        "asymmetry_mae": df_results["asymmetry_error"].mean(),
        "true_asymmetry_mae": df_results["true_asymmetry_error"].mean(),
        "eigval_mae": df_results["eigval_mae"].mean(),
        "eigval1_mae": df_results["eigval1_mae"].mean(),
        "eigval2_mae": df_results["eigval2_mae"].mean(),
        "eigvec1_mae": df_results["eigvec1_mae"].mean(),
        "eigvec2_mae": df_results["eigvec2_mae"].mean(),
        "eigvec1_cos": df_results["eigvec1_cos"].mean(),
        "eigvec2_cos": df_results["eigvec2_cos"].mean(),
        # Eckart projection
        "eigval_mae_eckart": df_results["eigval_mae_eckart"].mean(),
        "eigval1_mae_eckart": df_results["eigval1_mae_eckart"].mean(),
        "eigval2_mae_eckart": df_results["eigval2_mae_eckart"].mean(),
        "eigvec1_mae_eckart": df_results["eigvec1_mae_eckart"].mean(),
        "eigvec2_mae_eckart": df_results["eigvec2_mae_eckart"].mean(),
        "eigvec1_cos_eckart": df_results["eigvec1_cos_eckart"].mean(),
        "eigvec2_cos_eckart": df_results["eigvec2_cos_eckart"].mean(),
        # Frequencies
        "neg_num_agree": df_results["neg_num_agree"].mean(),
        "true_neg_num": df_results["true_neg_num"].mean(),
        "model_neg_num": df_results["model_neg_num"].mean(),
        "true_is_ts": df_results["true_is_ts"].mean(),
        "true_is_minima": df_results["true_is_minima"].mean(),
        "true_is_ts_order2": df_results["true_is_ts_order2"].mean(),
        "true_is_higher_order": df_results["true_is_higher_order"].mean(),
        "model_is_ts": df_results["model_is_ts"].mean(),
        "model_is_minima": df_results["model_is_minima"].mean(),
        "model_is_ts_order2": df_results["model_is_ts_order2"].mean(),
        "model_is_higher_order": df_results["model_is_higher_order"].mean(),
        "is_ts_agree": (df_results["model_is_ts"] == df_results["true_is_ts"]).mean(),
        # Speed
        "time": df_results["time"].mean(),  # ms
        "memory": df_results["memory"].mean(),
    }
    if time_taken_all is not None:
        # ms per forward pass
        aggregated_results["time_incltransform"] = time_taken_all / n_total_samples

    # print(f"\nResults for {dataset_name}:")
    # print(f"Energy MAE: {aggregated_results['energy_mae']:.6f}")
    # print(f"Forces MAE: {aggregated_results['forces_mae']:.6f}")
    # print(f"Hessian MAE: {aggregated_results['hessian_mae']:.6f}")
    # print(f"Asymmetry MAE: {aggregated_results['asymmetry_mae']:.6f}")
    # print(f"True Asymmetry MAE: {aggregated_results['true_asymmetry_mae']:.6f}")
    # print(f"Eigenvalue MAE: {aggregated_results['eigval_mae']:.6f} eV/Angstrom^2")
    # print(f"Eigenvalue 1 MAE: {aggregated_results['eigval1_mae']:.6f}")
    # print(f"Eigenvalue 2 MAE: {aggregated_results['eigval2_mae']:.6f}")
    # print(f"Eigenvector 1 MAE: {aggregated_results['eigvec1_mae']:.6f}")
    # print(f"Eigenvector 2 MAE: {aggregated_results['eigvec2_mae']:.6f}")
    # print(f"Eigenvector 1 Cosine: {aggregated_results['eigvec1_cos']:.6f}")
    # print(f"Eigenvector 2 Cosine: {aggregated_results['eigvec2_cos']:.6f}")

    # # Frequencies
    # print(f"True Neg Num: {aggregated_results['true_neg_num']:.6f}")
    # print(f"Model Neg Num: {aggregated_results['model_neg_num']:.6f}")
    # print(f"Neg Num Agree: {aggregated_results['neg_num_agree']:.6f}")
    # print(f"True Is TS: {aggregated_results['true_is_ts']:.6f}")
    # print(f"Model Is TS: {aggregated_results['model_is_ts']:.6f}")
    # print(f"Is TS Agree: {aggregated_results['is_ts_agree']:.6f}")

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
        ("energy_error", "Energy MAE", "Energy Error"),
        ("forces_error", "Forces MAE", "Forces Error"),
        ("hessian_error", "Hessian MAE", "Hessian Error"),
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
        type=str,
        default="autograd",
        help="Hessian computation method",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ts1x-val.lmdb",
        help="Dataset file name (e.g., ts1x-val.lmdb, ts1x_hess_train_big.lmdb, RGD1.lmdb)",
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
    lmdb_path = args.dataset
    max_samples = args.max_samples
    config_path = args.config_path
    hessian_method = args.hessian_method
    redo = args.redo

    name = f"{checkpoint_path.split('/')[-1].split('.')[0]}_{lmdb_path.split('/')[-1].split('.')[0]}_{hessian_method}"

    df_results, aggregated_results = evaluate(
        lmdb_path=lmdb_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        hessian_method=hessian_method,
        max_samples=max_samples,
        redo=redo,
    )

    # Plot accuracy over Natoms
    plot_accuracy_vs_natoms(df_results, name)
