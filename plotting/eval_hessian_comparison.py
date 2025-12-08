import torch
import argparse
from tqdm import tqdm
import wandb
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader as TGDataLoader
from pathlib import Path

from hip.training_module import PotentialModule
from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path
from hip.hessian_utils import compute_hessian

# Set seaborn style for poster
sns.set_style("whitegrid")
sns.set_context("poster")


def evaluate_hessian_comparison(
    checkpoint_path,
    dataset_path,
    max_samples=None,
    redo=False,
):
    """
    Evaluate hip_v2 on dataset and compare three Hessian computation methods:
    1. HIP Hessians (prediction)
    2. AD Hessians with direct force
    3. AD Hessians with conservative force

    Logs MAEs vs ground truth, pairwise MAEs between methods, and histograms to wandb.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    print(f"Model name: {model_name}")

    module = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
        map_location=device,
    )

    model = module.potential.to(device)
    model.eval()

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    fixed_dataset_path = fix_dataset_path(dataset_path)
    dataset = LmdbDataset(fixed_dataset_path)
    dataloader = TGDataLoader(dataset, batch_size=1, shuffle=False)

    # Create results directory
    results_dir = "results_eval_hessian_comparison"
    os.makedirs(results_dir, exist_ok=True)

    dataset_name = Path(fixed_dataset_path).stem
    ckpt_name = Path(checkpoint_path).stem
    results_file = f"{results_dir}/{ckpt_name}_{dataset_name}_hessian_comparison.csv"

    # Check if results already exist
    if os.path.exists(results_file) and not redo:
        print(f"Loading existing results from {results_file}")
        df_results = pd.read_csv(results_file)
    else:
        # Initialize metrics collection
        sample_metrics = []
        n_samples = 0

        if max_samples is not None:
            n_total_samples = min(max_samples, len(dataloader))
        else:
            n_total_samples = len(dataloader)

        print(f"Evaluating {n_total_samples} samples...")

        for batch in tqdm(dataloader, desc="Evaluating", total=n_total_samples):
            batch = batch.to(device)
            n_atoms = batch.pos.shape[0]

            # Extract ground truth Hessian
            hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)

            # Collect per-sample data
            sample_data = {
                "sample_idx": n_samples,
                "natoms": n_atoms,
            }

            # Method 1: HIP Hessians (prediction)
            with torch.no_grad():
                energy_hip, force_hip, out_hip = model.forward(
                    batch,
                    otf_graph=True,
                    hessian=True,
                )
            hessian_hip = out_hip["hessian"].reshape(n_atoms * 3, n_atoms * 3)

            # Method 2: AD with direct force
            batch_ad_direct = batch.clone()
            batch_ad_direct.pos.requires_grad_()
            energy_ad_direct, force_ad_direct, out_ad_direct = model.forward(
                batch_ad_direct,
                otf_graph=True,
                hessian=False,
            )
            hessian_ad_direct = compute_hessian(
                batch_ad_direct.pos, energy_ad_direct, force_ad_direct
            )

            # Method 3: AD with conservative force
            batch_ad_conservative = batch.clone()
            batch_ad_conservative.pos.requires_grad_()
            energy_ad_conservative, force_ad_conservative, out_ad_conservative = (
                model.forward(
                    batch_ad_conservative,
                    otf_graph=True,
                    hessian=False,
                    conservative_forces=True,
                    retain_forces_graph=True,
                )
            )
            hessian_ad_conservative = compute_hessian(
                batch_ad_conservative.pos,
                energy_ad_conservative,
                force_ad_conservative,
            )

            # Compute MAEs vs ground truth
            h_mae_hip = torch.mean(torch.abs(hessian_hip - hessian_true)).item()
            h_mae_ad_direct = torch.mean(
                torch.abs(hessian_ad_direct - hessian_true)
            ).item()
            h_mae_ad_conservative = torch.mean(
                torch.abs(hessian_ad_conservative - hessian_true)
            ).item()

            sample_data["hessian_mae_hip"] = h_mae_hip
            sample_data["hessian_mae_ad_direct"] = h_mae_ad_direct
            sample_data["hessian_mae_ad_conservative"] = h_mae_ad_conservative

            # Compute pairwise MAEs between methods
            h_mae_hip_vs_ad_direct = torch.mean(
                torch.abs(hessian_hip - hessian_ad_direct)
            ).item()
            h_mae_hip_vs_ad_conservative = torch.mean(
                torch.abs(hessian_hip - hessian_ad_conservative)
            ).item()
            h_mae_ad_direct_vs_conservative = torch.mean(
                torch.abs(hessian_ad_direct - hessian_ad_conservative)
            ).item()

            sample_data["hessian_mae_hip_vs_ad_direct"] = h_mae_hip_vs_ad_direct
            sample_data["hessian_mae_hip_vs_ad_conservative"] = (
                h_mae_hip_vs_ad_conservative
            )
            sample_data["hessian_mae_ad_direct_vs_conservative"] = (
                h_mae_ad_direct_vs_conservative
            )

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

    # Compute aggregated results
    aggregated_results = {}
    for col in df_results.columns:
        if pd.api.types.is_numeric_dtype(df_results[col]):
            aggregated_results[col] = df_results[col].mean()

    # Initialize wandb
    wandb_config = {
        "checkpoint": checkpoint_path,
        "dataset": dataset_path,
        "max_samples": max_samples,
        "model_name": model_name,
        "n_samples": len(df_results),
    }

    wandb.init(
        project="horm",
        name=f"{ckpt_name}_{dataset_name}_hessian_comparison",
        config=wandb_config,
        tags=["hessian_comparison"],
    )

    # Log aggregated metrics
    wandb.log(aggregated_results, step=0)

    # Log histograms (wandb native)
    wandb.log(
        {
            "histogram/hessian_mae_hip": wandb.Histogram(
                df_results["hessian_mae_hip"].values
            ),
            "histogram/hessian_mae_ad_direct": wandb.Histogram(
                df_results["hessian_mae_ad_direct"].values
            ),
            "histogram/hessian_mae_ad_conservative": wandb.Histogram(
                df_results["hessian_mae_ad_conservative"].values
            ),
            "histogram/hessian_mae_hip_vs_ad_direct": wandb.Histogram(
                df_results["hessian_mae_hip_vs_ad_direct"].values
            ),
            "histogram/hessian_mae_hip_vs_ad_conservative": wandb.Histogram(
                df_results["hessian_mae_hip_vs_ad_conservative"].values
            ),
            "histogram/hessian_mae_ad_direct_vs_conservative": wandb.Histogram(
                df_results["hessian_mae_ad_direct_vs_conservative"].values
            ),
        },
        step=0,
    )

    # Create and log seaborn histograms
    plot_dir = f"{results_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Define histogram configurations
    hist_configs = [
        {
            "data": df_results["hessian_mae_hip"],
            "title": "HIP Hessian MAE vs Ground Truth",
            "xlabel": "MAE (eV/Å²)",
            "filename": f"{plot_dir}/histogram_hessian_mae_hip.png",
            "wandb_key": "plot/hessian_mae_hip",
        },
        {
            "data": df_results["hessian_mae_ad_direct"],
            "title": "AD (Direct Force) Hessian MAE vs Ground Truth",
            "xlabel": "MAE (eV/Å²)",
            "filename": f"{plot_dir}/histogram_hessian_mae_ad_direct.png",
            "wandb_key": "plot/hessian_mae_ad_direct",
        },
        {
            "data": df_results["hessian_mae_ad_conservative"],
            "title": "AD (Conservative Force) Hessian MAE vs Ground Truth",
            "xlabel": "MAE (eV/Å²)",
            "filename": f"{plot_dir}/histogram_hessian_mae_ad_conservative.png",
            "wandb_key": "plot/hessian_mae_ad_conservative",
        },
        {
            "data": df_results["hessian_mae_hip_vs_ad_direct"],
            "title": "HIP vs AD (Direct Force) Hessian MAE",
            "xlabel": "MAE (eV/Å²)",
            "filename": f"{plot_dir}/histogram_hessian_mae_hip_vs_ad_direct.png",
            "wandb_key": "plot/hessian_mae_hip_vs_ad_direct",
        },
        {
            "data": df_results["hessian_mae_hip_vs_ad_conservative"],
            "title": "HIP vs AD (Conservative Force) Hessian MAE",
            "xlabel": "MAE (eV/Å²)",
            "filename": f"{plot_dir}/histogram_hessian_mae_hip_vs_ad_conservative.png",
            "wandb_key": "plot/hessian_mae_hip_vs_ad_conservative",
        },
        {
            "data": df_results["hessian_mae_ad_direct_vs_conservative"],
            "title": "AD (Direct) vs AD (Conservative) Hessian MAE",
            "xlabel": "MAE (eV/Å²)",
            "filename": f"{plot_dir}/histogram_hessian_mae_ad_direct_vs_conservative.png",
            "wandb_key": "plot/hessian_mae_ad_direct_vs_conservative",
        },
    ]

    # Create individual histograms
    for config in hist_configs:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=config["data"], bins=30, kde=True, ax=ax)
        ax.set_title(config["title"], fontsize=16, fontweight="bold")
        ax.set_xlabel(config["xlabel"], fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(config["filename"], dpi=300, bbox_inches="tight")
        plt.close()

        # Log to wandb
        wandb.log({config["wandb_key"]: wandb.Image(config["filename"])}, step=0)

    # Create combined histogram plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, config in enumerate(hist_configs):
        sns.histplot(data=config["data"], bins=30, kde=True, ax=axes[idx])
        axes[idx].set_title(config["title"], fontsize=14, fontweight="bold")
        axes[idx].set_xlabel(config["xlabel"], fontsize=12)
        axes[idx].set_ylabel("Frequency", fontsize=12)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    combined_plot_path = f"{plot_dir}/histogram_combined.png"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Log combined plot to wandb
    wandb.log({"plot/hessian_mae_combined": wandb.Image(combined_plot_path)}, step=0)

    wandb.finish()

    # Print summary
    print("\n" + "=" * 60)
    print("Hessian Comparison Summary")
    print("=" * 60)
    print(f"MAE vs Ground Truth:")
    print(f"  HIP:              {aggregated_results['hessian_mae_hip']:.6f}")
    print(f"  AD (direct):      {aggregated_results['hessian_mae_ad_direct']:.6f}")
    print(
        f"  AD (conservative): {aggregated_results['hessian_mae_ad_conservative']:.6f}"
    )
    print(f"\nPairwise MAEs:")
    print(
        f"  HIP vs AD (direct):      {aggregated_results['hessian_mae_hip_vs_ad_direct']:.6f}"
    )
    print(
        f"  HIP vs AD (conservative): {aggregated_results['hessian_mae_hip_vs_ad_conservative']:.6f}"
    )
    print(
        f"  AD (direct) vs AD (conservative): {aggregated_results['hessian_mae_ad_direct_vs_conservative']:.6f}"
    )
    print("=" * 60)

    return df_results, aggregated_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare three Hessian computation methods on hip_v2"
    )
    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        default="ckpt/hip_v2.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="data/sample_100.lmdb",
        help="Dataset file path",
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
        action="store_true",
        help="Recompute even if results already exist",
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    df_results, aggregated_results = evaluate_hessian_comparison(
        checkpoint_path=args.ckpt_path,
        dataset_path=args.dataset,
        max_samples=args.max_samples,
        redo=args.redo,
    )

    print("\nDone!")
