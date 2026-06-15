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
from torch.utils.data import Subset

from hip.training_module import PotentialModule
from hip.ff_lmdb import LmdbDataset, Z_TO_ATOM_SYMBOL
from hip.path_config import fix_dataset_path
from nets.equiformer_v2.equiformer_v2_oc20 import center_batch_positions

from hip.frequency_analysis import analyze_frequencies_np


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    if value.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


def save_results_csv(df, path):
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model_config = ckpt["hyper_parameters"]["model_config"]
    print(f"Model name: {model_name}")
    if "equiformer" not in model_name.lower():
        raise ValueError("HIP evaluation only supports Equiformer checkpoints")

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
    ).potential.to(device)
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
    n_timed_samples = None
    n_total_samples = None
    required_result_columns = [
        "energy_model",
        "energy_true",
        "energy_difference",
        "force_model_norm",
        "force_true_norm",
        "force_l1_error",
        "force_l2_error",
        "force_cos_error",
        "hessian_model_fro_norm",
        "hessian_true_fro_norm",
    ]

    df_results = None
    df_existing = None

    if not redo and os.path.exists(results_file):
        df_existing = pd.read_csv(results_file)

    if df_existing is not None and "dataset_idx" not in df_existing.columns:
        missing_columns = [
            col for col in required_result_columns if col not in df_existing.columns
        ]
        if (
            max_samples is not None
            and len(df_existing) >= max_samples
            and not missing_columns
        ):
            print(f"Loading existing results from {results_file}")
            df_results = df_existing.iloc[:max_samples]
        else:
            print(
                f"Existing results at {results_file} do not include dataset_idx; "
                "rerunning because partial resume is not possible for that file."
            )

    if df_results is None:
        torch.manual_seed(42)
        np.random.seed(42)

        dataset = LmdbDataset(fix_dataset_path(lmdb_path))

        generator = torch.Generator()
        generator.manual_seed(42)
        eval_indices = torch.randperm(len(dataset), generator=generator).tolist()
        if max_samples is not None:
            eval_indices = eval_indices[:max_samples]

        n_total_samples = len(eval_indices)

        completed_indices = set()
        sample_metrics = []
        if df_existing is not None and "dataset_idx" in df_existing.columns:
            missing_columns = [
                col for col in required_result_columns if col not in df_existing.columns
            ]
            if missing_columns:
                print(
                    f"Existing results at {results_file} are missing columns "
                    f"{missing_columns}; recomputing those samples."
                )
                df_existing = df_existing.iloc[0:0]
            df_existing = df_existing[df_existing["dataset_idx"].isin(eval_indices)]
            df_existing = df_existing.drop_duplicates("dataset_idx", keep="last")
            completed_indices = set(df_existing["dataset_idx"].astype(int).tolist())
            sample_metrics = df_existing.to_dict("records")
            print(
                f"Resuming from {results_file}: "
                f"{len(completed_indices)}/{n_total_samples} samples already complete"
            )

        pending_indices = [idx for idx in eval_indices if idx not in completed_indices]

        if not pending_indices:
            df_results = pd.DataFrame(sample_metrics)
            print(f"Loading complete results from {results_file}")
        else:
            # dataset = LmdbDataset(fix_dataset_path(lmdb_path))
            dataloader = TGDataLoader(
                Subset(dataset, pending_indices), batch_size=1, shuffle=False
            )
            n_timed_samples = len(pending_indices)

            sample_idx_by_dataset_idx = {
                dataset_idx: sample_idx
                for sample_idx, dataset_idx in enumerate(eval_indices)
            }

            # Warmup
            warmup_indices = pending_indices[:10]
            warmup_loader = TGDataLoader(
                Subset(dataset, warmup_indices), batch_size=1, shuffle=False
            )
            for batch in tqdm(warmup_loader, desc="Warmup", total=len(warmup_indices)):
                batch = center_batch_positions(batch.to(device))

                n_atoms = batch.pos.shape[0]

                torch.cuda.reset_peak_memory_stats()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                if do_autograd:
                    batch.pos.requires_grad_()
                    energy_model, force_model, out = model.forward(
                        batch, otf_graph=False, hessian=False
                    )
                    hessian_model = compute_hessian(batch.pos, energy_model, force_model)
                else:
                    with torch.no_grad():
                        energy_model, force_model, out = model.forward(
                            batch,
                            otf_graph=False,
                        )
                    hessian_model = out["hessian"].reshape(n_atoms * 3, n_atoms * 3)

            start_event_all = torch.cuda.Event(enable_timing=True)
            end_event_all = torch.cuda.Event(enable_timing=True)
            start_event_all.record()

            for dataset_idx, batch in tqdm(
                zip(pending_indices, dataloader),
                desc="Evaluating",
                total=len(pending_indices),
            ):
                batch = center_batch_positions(batch.to(device))

                n_atoms = batch.pos.shape[0]

                torch.cuda.reset_peak_memory_stats()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                if do_autograd:
                    batch.pos.requires_grad_()
                    energy_model, force_model, out = model.forward(
                        batch, otf_graph=False, hessian=False
                    )
                    hessian_model = compute_hessian(batch.pos, energy_model, force_model)
                else:
                    with torch.no_grad():
                        energy_model, force_model, out = model.forward(
                            batch,
                            otf_graph=False,
                        )
                    hessian_model = out["hessian"]

                end_event.record()
                torch.cuda.synchronize()

                time_taken = start_event.elapsed_time(end_event)  # ms
                memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB

                hessian_model = hessian_model.reshape(n_atoms * 3, n_atoms * 3)

                # Compute hessian eigenspectra
                eigvals_model, eigvecs_model = torch.linalg.eigh(hessian_model)

                # Compute errors
                if "energy" in batch.keys():
                    true_energy = batch.energy
                else:
                    true_energy = batch.ae
                energy_model_scalar = energy_model.squeeze()
                energy_true_scalar = true_energy.squeeze()
                energy_difference = energy_model_scalar - energy_true_scalar
                e_error = torch.mean(torch.abs(energy_difference))

                force_diff = force_model - batch.forces
                f_error = torch.mean(torch.abs(force_diff))
                force_diff_flat = force_diff.reshape(-1)
                force_model_flat = force_model.reshape(-1)
                force_true_flat = batch.forces.reshape(-1)
                force_model_norm = torch.linalg.vector_norm(force_model_flat, ord=2)
                force_true_norm = torch.linalg.vector_norm(force_true_flat, ord=2)
                force_l1_error = torch.linalg.vector_norm(force_diff_flat, ord=1)
                force_l2_error = torch.linalg.vector_norm(force_diff_flat, ord=2)
                force_cos_similarity = torch.dot(force_model_flat, force_true_flat) / (
                    force_model_norm * force_true_norm + 1e-8
                )
                force_cos_error = 1.0 - force_cos_similarity

                # Reshape true hessian
                n_atoms = batch.pos.shape[0]
                hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)
                hessian_model_fro_norm = torch.linalg.matrix_norm(hessian_model, ord="fro")
                hessian_true_fro_norm = torch.linalg.matrix_norm(hessian_true, ord="fro")
                h_error = torch.mean(torch.abs(hessian_model - hessian_true))
                h_mre = torch.mean(
                    torch.abs(hessian_model - hessian_true)
                    / (torch.abs(hessian_true) + 1e-8)
                )

                # Eigenvalue error
                eigvals_true, eigvecs_true = torch.linalg.eigh(hessian_true)

                # Asymmetry error
                asymmetry_error = torch.mean(torch.abs(hessian_model - hessian_model.T))
                true_asymmetry_error = torch.mean(
                    torch.abs(hessian_true - hessian_true.T)
                )

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
                eigvec1_cos = torch.abs(
                    torch.dot(eigvecs_model[:, 0], eigvecs_true[:, 0])
                )
                eigvec2_cos = torch.abs(
                    torch.dot(eigvecs_model[:, 1], eigvecs_true[:, 1])
                )

                # Collect per-sample metrics
                sample_data = {
                    "sample_idx": sample_idx_by_dataset_idx[dataset_idx],
                    "dataset_idx": dataset_idx,
                    "natoms": n_atoms,
                    "energy_model": energy_model_scalar.item(),
                    "energy_true": energy_true_scalar.item(),
                    "energy_difference": energy_difference.item(),
                    "energy_error": e_error.item(),
                    "force_model_norm": force_model_norm.item(),
                    "force_true_norm": force_true_norm.item(),
                    "force_l1_error": force_l1_error.item(),
                    "force_l2_error": force_l2_error.item(),
                    "force_cos_error": force_cos_error.item(),
                    "forces_error": f_error.item(),
                    "hessian_model_fro_norm": hessian_model_fro_norm.item(),
                    "hessian_true_fro_norm": hessian_true_fro_norm.item(),
                    "hessian_error": h_error.item(),
                    "hessian_mre": h_mre.item(),
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
                sample_data["model_is_ts_order2"] = (
                    1 if freqs_model_neg_num == 2 else 0
                )
                sample_data["model_is_higher_order"] = (
                    1 if freqs_model_neg_num > 2 else 0
                )
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
                sample_data["eigvec1_mae_eckart"] = torch.mean(
                    torch.abs(eigvecs_model_eckart[:, 0] - true_eigvecs_eckart[:, 0])
                ).item()
                sample_data["eigvec2_mae_eckart"] = torch.mean(
                    torch.abs(eigvecs_model_eckart[:, 1] - true_eigvecs_eckart[:, 1])
                ).item()
                sample_data["eigvec1_cos_eckart"] = torch.abs(
                    torch.dot(eigvecs_model_eckart[:, 0], true_eigvecs_eckart[:, 0])
                ).item()
                sample_data["eigvec2_cos_eckart"] = torch.abs(
                    torch.dot(eigvecs_model_eckart[:, 1], true_eigvecs_eckart[:, 1])
                ).item()

                # Global eigenvector overlap: ||abs(Q_model @ Q_true^T) - I||_F
                M = eigvecs_model_eckart.T @ true_eigvecs_eckart
                sample_data["eigvec_overlap_error"] = torch.norm(
                    M.abs() - torch.eye(M.shape[0]), p="fro"
                ).item()

                sample_metrics.append(sample_data)
                df_results = pd.DataFrame(sample_metrics)
                df_results = df_results.sort_values("sample_idx")
                save_results_csv(df_results, results_file)

                # Memory management
                torch.cuda.empty_cache()

            end_event_all.record()
            torch.cuda.synchronize()

            time_taken_all = start_event_all.elapsed_time(end_event_all)  # ms

            # Create DataFrame from collected metrics
            df_results = pd.DataFrame(sample_metrics)
            df_results = df_results.sort_values("sample_idx")

            # Save DataFrame
            save_results_csv(df_results, results_file)
            print(f"Saved results to {results_file}")

    aggregated_results = {
        "energy_difference": df_results["energy_difference"].mean(),
        "energy_mae": df_results["energy_error"].mean(),
        "force_l1_error": df_results["force_l1_error"].mean(),
        "force_l2_error": df_results["force_l2_error"].mean(),
        "force_cos_error": df_results["force_cos_error"].mean(),
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
        aggregated_results["time_incltransform"] = time_taken_all / n_timed_samples

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


"""
uv run python scripts/eval_horm.py -c ckpt/eqv2.ckpt -d ts1x-val.lmdb -m 1000 -r True
uv run python scripts/eval_horm.py -c ckpt/hesspred_v1.ckpt -d ts1x-val.lmdb -m 1000 -r True -hm predict
uv run python scripts/eval_horm.py -c ckpt/hip_v3.ckpt -d ts1x-val.lmdb -m 1000 -r True -hm predict
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
        default="autograd",
        help="Hessian computation method: autograd, predict",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ts1x-val.lmdb",
        help="Dataset file name (e.g., ts1x-val.lmdb, ts1x_hess_train.lmdb, RGD1.lmdb)",
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
        type=str2bool,
        nargs="?",
        const=True,
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
    # plot_accuracy_vs_natoms(df_results, name)
