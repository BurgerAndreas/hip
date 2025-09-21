import torch
import argparse
from tqdm import tqdm
import wandb
from torch_geometric.loader import DataLoader as TGDataLoader
import pandas as pd
import plotly.graph_objects as go
import time
import sys
import json
from pathlib import Path
import os

import math
from hip.training_module import PotentialModule, compute_extra_props
from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path, DATASET_FILES_HORM
from ocpmodels.hessian_graph_transform import HessianGraphTransform, FOLLOW_BATCH
from hip.training_module import (
    SchemaUniformDataset,
)

# https://plotly.com/python/templates/
# ['ggplot2', 'seaborn', 'simple_white', 'plotly',
# 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
# 'ygridoff', 'gridon', 'none']
PLOTLY_TEMPLATE = "plotly_white"

from torch.utils.data import Subset
from collections import defaultdict


def save_idx_by_natoms(args):
    if isinstance(args, str):
        args = argparse.Namespace(dataset_path=args)
    elif isinstance(args, dict):
        args = argparse.Namespace(**args)

    print(f"Loading dataset: {args.dataset_path}")
    fixed_path = fix_dataset_path(args.dataset_path)
    print(f"Fixed path: {fixed_path}")

    dataset = LmdbDataset(Path(fixed_path))
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) == 0:
        print(f"Dataset {args.dataset_path} is empty!")
        return

    indices_by_natoms = defaultdict(list)

    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        sample = dataset[i]
        natoms = 0
        if hasattr(sample, "z"):
            natoms = len(sample.z)
        elif hasattr(sample, "pos"):
            natoms = sample.pos.shape[0]
        else:
            print(f"Warning: Could not determine number of atoms for sample {i}")
            continue

        if natoms > 0:
            indices_by_natoms[natoms].append(i)

    # Sort keys
    sorted_indices_by_natoms = {
        k: sorted(v) for k, v in sorted(indices_by_natoms.items())
    }

    # Generate output file path
    dataset_name = Path(fixed_path).stem
    output_filename = f"{dataset_name}_indices_by_natoms.json"
    output_path = Path(fixed_path).parent / output_filename

    print(f"Saving indices to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(sorted_indices_by_natoms, f, indent=2)

    # --- Save a smaller version for quick access ---
    truncated_indices_by_natoms = {
        k: v[:1000] for k, v in sorted_indices_by_natoms.items()
    }
    small_output_filename = f"{dataset_name}_indices_by_natoms_small.json"
    small_output_path = Path(fixed_path).parent / small_output_filename

    print(f"Saving smaller indices file to: {small_output_path}")
    with open(small_output_path, "w") as f:
        json.dump(truncated_indices_by_natoms, f, indent=2)

    print("Done.")
    return {
        "all_idx": sorted_indices_by_natoms,
        "small_idx": truncated_indices_by_natoms,
        "all_path": output_path,
        "small_path": small_output_path,
    }


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


def hess2eigenvalues(hess):
    """Convert Hessian to eigenvalues with unit conversion (hartree to eV, bohr to angstrom)"""
    hartree_to_ev = 27.2114
    bohr_to_angstrom = 0.529177
    ev_angstrom_2_to_hartree_bohr_2 = (bohr_to_angstrom**2) / hartree_to_ev

    hess = hess * ev_angstrom_2_to_hartree_bohr_2
    eigen_values, _ = torch.linalg.eigh(hess)
    return eigen_values


def time_hessian_computation(model, batch, hessian_method):
    """Times a single hessian computation and measures memory usage."""
    do_autograd = hessian_method == "autograd"
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    if "equiformer" in model.name.lower():
        if do_autograd:
            batch.pos.requires_grad_()
            ener, force, out = model.forward(batch, otf_graph=True, hessian=False)
            hess = compute_hessian(batch.pos, ener, force)
        else:
            with torch.no_grad():
                # for a fair comparison
                # compute graph and Hessian indices on the fly
                ener, force, out = model.forward(
                    batch, otf_graph=True, hessian=True, add_props=True
                )
                hess = out["hessian"]
    else:
        batch.pos.requires_grad_()
        ener, force, out = model.forward(batch)
        hess = compute_hessian(batch.pos, ener, force)

    end_event.record()
    torch.cuda.synchronize()

    time_taken = start_event.elapsed_time(end_event)
    memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    return time_taken, memory_usage


def speed_comparison(
    checkpoint_path,
    dataset_name,
    max_samples_per_n,
    device="cuda",
    output_dir="./results_speed",
    output_path=None,
):
    """Compares the speed of autograd vs prediction for Hessian computation."""
    print("\nSpeed comparison")
    # Load model
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()
    model.name = model_name

    # Get indices by number of atoms
    fixed_dataset_path = Path(fix_dataset_path(dataset_name))
    indices_file = (
        fixed_dataset_path.parent
        / f"{fixed_dataset_path.stem}_indices_by_natoms_small.json"
    )

    if indices_file.exists():
        print(f"Loading indices from {indices_file}")
        with open(indices_file, "r") as f:
            indices_by_natoms = json.load(f)
    else:
        print(f"Indices file not found. Generating new indices for {dataset_name}")
        results = save_idx_by_natoms({"dataset_path": dataset_name})
        indices_by_natoms = results["small_idx"]

    # Prepare dataset and dataloader
    transform = HessianGraphTransform(
        cutoff=model.cutoff,
        cutoff_hessian=model.cutoff_hessian,
        max_neighbors=model.max_neighbors,
        use_pbc=model.use_pbc,
    )
    dataset = LmdbDataset(fix_dataset_path(dataset_name), transform=transform)
    dataset = SchemaUniformDataset(dataset)

    # do a couple of forward passes to warm up the model
    # populate caches, jit, load cuda kernels, and what not
    loader = TGDataLoader(dataset, batch_size=1, shuffle=False)
    for i, sample in enumerate(loader):
        batch = sample.to(device)
        batch = compute_extra_props(batch)
        time_hessian_computation(model, batch, "prediction")
        torch.cuda.empty_cache()
        time_hessian_computation(model, batch, "autograd")
        torch.cuda.empty_cache()
        if i > 10:
            break
    print("Model warmed up")

    results = []

    for n_atoms, indices in tqdm(indices_by_natoms.items(), desc="Processing N_atoms"):
        if len(indices) == 0:
            continue
        n_atoms = int(n_atoms)

        # Limit number of samples
        indices_to_test = indices[:max_samples_per_n]

        subset = Subset(dataset, indices_to_test)
        loader = TGDataLoader(
            subset, batch_size=1, shuffle=False, follow_batch=FOLLOW_BATCH
        )

        for hessian_method in ["prediction", "autograd"]:
            do_autograd = hessian_method == "autograd"
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            for batch in tqdm(loader, desc=f"N={n_atoms}", leave=False):
                batch = batch.to(device)
                batch = compute_extra_props(batch)

                # Time prediction
                if "equiformer" in model.name.lower():
                    if do_autograd:
                        batch.pos.requires_grad_()
                        ener, force, out = model.forward(
                            batch, otf_graph=True, hessian=False
                        )
                        hess = compute_hessian(batch.pos, ener, force)
                    else:
                        with torch.no_grad():
                            # for a fair comparison
                            # compute graph and Hessian indices on the fly
                            ener, force, out = model.forward(
                                batch, otf_graph=True, hessian=True, add_props=True
                            )
                            hess = out["hessian"]
                else:
                    batch.pos.requires_grad_()
                    ener, force, out = model.forward(batch)
                    hess = compute_hessian(batch.pos, ener, force)

                # clear memory
                torch.cuda.empty_cache()

            end_event.record()
            torch.cuda.synchronize()

            time_taken = start_event.elapsed_time(end_event)
            memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB

            results.append(
                {
                    "n_atoms": n_atoms,
                    "nsamples": len(indices_to_test),
                    "time": time_taken,
                    "time_per_sample": time_taken / len(indices_to_test),
                    "memory": memory_usage,
                    "method": hessian_method,
                }
            )

    # Save results
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # output_path = (
    #     output_dir / f"{dataset_name}_speed_comparison_incltransform_results.csv"
    # )
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return results_df


def plot_speed_comparison(results_df, output_dir="./results_speed"):
    output_dir = Path(output_dir)
    # Plot results for speed
    avg_times = (
        results_df.groupby(["n_atoms", "method"])["time_per_sample"].mean().unstack()
    )
    std_times = (
        results_df.groupby(["n_atoms", "method"])["time_per_sample"].std().unstack()
    )

    fig = go.Figure()
    for method in avg_times.columns:
        display_name = (
            "Prediction (ours)" if str(method).lower() == "prediction" else str(method)
        )
        fig.add_trace(
            go.Scatter(
                x=avg_times.index,
                y=avg_times[method],
                mode="lines+markers",
                name=display_name,
                # error_y=dict(type="data", array=std_times[method]),
            )
        )

    fig.update_layout(
        title="Hessian Computation Speed: Autograd vs. Prediction",
        xaxis_title="Number of Atoms (N)",
        yaxis_title="Average Time (ms)",
        legend_title="Method",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
    )
    # output_path = output_dir / "speed_comparison_plot_incltransform.html"
    # fig.write_html(output_path)
    # print(f"Plot saved to {output_path}")
    output_path = output_dir / "speed_comparison_plot_incltransform.png"
    fig.write_image(output_path)
    print(f"Plot saved to {output_path}")

    # Plot results for memory
    plot_memory_usage(results_df, output_dir)


def plot_memory_usage(results_df, output_dir="./results_speed"):
    output_dir = Path(output_dir)
    avg_memory = results_df.groupby(["n_atoms", "method"])["memory"].mean().unstack()

    fig = go.Figure()
    for method in avg_memory.columns:
        display_name = (
            "Prediction (ours)" if str(method).lower() == "prediction" else str(method)
        )
        fig.add_trace(
            go.Scatter(
                x=avg_memory.index,
                y=avg_memory[method],
                mode="lines+markers",
                name=display_name,
            )
        )

    fig.update_layout(
        title="Hessian Computation Memory Usage: Autograd vs. Prediction",
        xaxis_title="Number of Atoms (N)",
        yaxis_title="Peak Memory (MB)",
        legend_title="Method",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
    )
    # output_path = output_dir / "memory_usage_plot_incltransform.html"
    # fig.write_html(output_path)
    # print(f"Plot saved to {output_path}")
    output_path = output_dir / "memory_usage_plot_incltransform.png"
    fig.write_image(output_path)
    print(f"Plot saved to {output_path}")


def batchsize_prediction_speed_test(
    checkpoint_path,
    dataset_name,
    num_samples=100,
    batch_sizes=None,
    device="cuda",
    output_dir="./results_speed",
    output_path=None,
):
    """Benchmark predicted Hessian speed over varying batch sizes on random samples.

    - Randomly selects num_samples items with mixed N atoms from the dataset
    - Times only the prediction path (no autograd)
    - Tests each batch size and records total wall time and peak memory
    """
    print("\nBatch-size prediction speed test")
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    # Load model
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()
    model.name = model_name

    # Prepare dataset (with transform to compute Hessian indices on-the-fly)
    transform = HessianGraphTransform(
        cutoff=model.cutoff,
        cutoff_hessian=model.cutoff_hessian,
        max_neighbors=model.max_neighbors,
        use_pbc=model.use_pbc,
    )
    dataset = LmdbDataset(fix_dataset_path(dataset_name), transform=transform)
    dataset = SchemaUniformDataset(dataset)

    if len(dataset) == 0:
        raise RuntimeError(f"Dataset {dataset_name} is empty")

    # Random subset indices to mix N atoms
    # respect global torch seed set in main
    random_indices = torch.randperm(len(dataset))[:num_samples].tolist()

    results = []

    # Warmup a couple of steps
    warm_loader = TGDataLoader(
        dataset, batch_size=1, shuffle=False, follow_batch=FOLLOW_BATCH
    )
    for i, sample in enumerate(warm_loader):
        batch = sample.to(device)
        batch = compute_extra_props(batch)
        if "equiformer" in model.name.lower():
            with torch.no_grad():
                _ener, _force, out = model.forward(
                    batch, otf_graph=True, hessian=True, add_props=True
                )
        else:
            # Fallback warmup for non-equiformer models (no prediction path)
            batch.pos.requires_grad_()
            _ener, _force, _out = model.forward(batch)
            _ = compute_hessian(batch.pos, _ener, _force)
        torch.cuda.empty_cache()
        if i > 5:
            break

    print("Batch-size test: model warmed up")

    subset = Subset(dataset, random_indices)

    for bs in batch_sizes:
        print(f"Batch size: {bs}")
        transform = HessianGraphTransform(
            cutoff=model.cutoff,
            cutoff_hessian=model.cutoff_hessian,
            max_neighbors=model.max_neighbors,
            use_pbc=model.use_pbc,
        )
        loader = TGDataLoader(
            subset, batch_size=bs, shuffle=True, follow_batch=FOLLOW_BATCH
        )

        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        for batch in tqdm(loader, desc=f"batch_size={bs}", leave=False):
            batch = batch.to(device)
            batch = compute_extra_props(batch, pos_require_grad=False)

            if "equiformer" in model.name.lower():
                with torch.no_grad():
                    _ener, _force, out = model.forward(
                        batch, otf_graph=False, hessian=True, add_props=True
                    )
            else:
                # For non-equiformer models, prediction path isn't available
                # Fall back to autograd timing to keep the loop functional
                batch.pos.requires_grad_()
                _ener, _force, _out = model.forward(batch)
                _ = compute_hessian(batch.pos, _ener, _force)

            torch.cuda.empty_cache()

        end_event.record()
        torch.cuda.synchronize()

        time_ms = start_event.elapsed_time(end_event)
        mem_mb = torch.cuda.max_memory_allocated() / 1e6

        # Guard against division by zero if loader had no batches
        num_items = len(subset)
        time_per_sample_ms = time_ms / max(1, num_items)

        results.append(
            {
                "batch_size": bs,
                "time_total_ms": time_ms,
                "time_per_sample_ms": time_per_sample_ms,
                "memory_mb": mem_mb,
                "num_samples": num_items,
                "method": "prediction"
                if "equiformer" in model.name.lower()
                else "autograd",
            }
        )

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Batch-size results saved to {output_path}")
    return df


def plot_batchsize_prediction(results_df, output_dir="./results_speed"):
    output_dir = Path(output_dir)

    # Time per sample vs batch size
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results_df["batch_size"],
            y=results_df["time_per_sample_ms"],
            mode="lines+markers",
            name="time/sample (ms)",
        )
    )
    fig.update_layout(
        title="Predicted Hessian: Time per sample vs Batch size",
        xaxis_title="Batch size",
        yaxis_title="Time per sample (ms)",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
    )
    output_path = output_dir / "batchsize_prediction_time_per_sample.html"
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")
    output_path = output_dir / "batchsize_prediction_time_per_sample.png"
    fig.write_image(output_path)
    print(f"Plot saved to {output_path}")

    # Time per batch vs batch size
    num_batches = [
        max(1, math.ceil(ns / bs))
        for bs, ns in zip(results_df["batch_size"], results_df["num_samples"])
    ]
    time_per_batch_ms = results_df["time_total_ms"] / num_batches

    fig_batch = go.Figure()
    fig_batch.add_trace(
        go.Scatter(
            x=results_df["batch_size"],
            y=time_per_batch_ms,
            mode="lines+markers",
            name="time/batch (ms)",
        )
    )
    fig_batch.update_layout(
        title="Predicted Hessian: Time per batch vs Batch size",
        xaxis_title="Batch size",
        yaxis_title="Time per batch (ms)",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
    )
    output_path = output_dir / "batchsize_prediction_time_per_batch.html"
    fig_batch.write_html(output_path)
    print(f"Plot saved to {output_path}")
    output_path = output_dir / "batchsize_prediction_time_per_batch.png"
    fig_batch.write_image(output_path)
    print(f"Plot saved to {output_path}")

    # Peak memory vs batch size
    fig_mem = go.Figure()
    fig_mem.add_trace(
        go.Scatter(
            x=results_df["batch_size"],
            y=results_df["memory_mb"],
            mode="lines+markers",
            name="peak memory (MB)",
        )
    )
    fig_mem.update_layout(
        title="Predicted Hessian: Peak memory vs Batch size",
        xaxis_title="Batch size",
        yaxis_title="Peak memory (MB)",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
    )
    output_path = output_dir / "batchsize_prediction_memory.html"
    fig_mem.write_html(output_path)
    print(f"Plot saved to {output_path}")
    output_path = output_dir / "batchsize_prediction_memory.png"
    fig_mem.write_image(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    """
    python scripts/speed_comparison_incltransform.py --dataset RGD1.lmdb --max_samples_per_n 10 --ckpt_path ../ReactBench/ckpt/hesspred/eqv2hp1.ckpt
    """
    parser = argparse.ArgumentParser(description="Speed comparison")

    # Subparser for speed comparison
    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        default="ckpt/eqv2.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="RGD1.lmdb",  # "ts1x-val.lmdb",
        help="Dataset file name",
    )
    parser.add_argument(
        "--max_samples_per_n",
        type=int,
        default=10,
        help="Maximum number of samples per N atoms to test.",
    )
    parser.add_argument(
        "--redo",
        type=bool,
        default=False,
        help="Redo the speed comparison. If false attempt to load existing results.",
    )
    parser.add_argument(
        "--run_batchsize",
        action="store_true",
        help="Also run batch-size prediction speed test",
        default=True,
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,2,4,8,16,32,64,128",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--batchsize_num_samples",
        type=int,
        default=100,
        help="Number of random samples to use in batch-size test",
    )

    args = parser.parse_args()
    torch.manual_seed(42)

    redo = args.redo

    output_dir = "./results_speed"
    output_dir = Path(output_dir)
    output_path = (
        output_dir / f"{args.dataset}_speed_comparison_incltransform_results.csv"
    )
    if not redo:
        if output_path.exists():
            results_df = pd.read_csv(output_path)
            print(f"Loaded existing results from {output_path}")
        else:
            redo = True

    if redo:
        results_df = speed_comparison(
            checkpoint_path=args.ckpt_path,
            dataset_name=args.dataset,
            max_samples_per_n=args.max_samples_per_n,
            output_dir=output_dir,
            output_path=output_path,
        )

    # Plot results
    plot_speed_comparison(results_df)

    if args.run_batchsize:
        try:
            batch_sizes = [
                int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()
            ]
        except Exception as e:
            raise ValueError(f"Failed to parse --batch_sizes '{args.batch_sizes}': {e}")

        output_path = (
            output_dir
            / f"{args.dataset}_batchsize_prediction_incltransform_results.csv"
        )

        redo = args.redo

        if not redo:
            if output_path.exists():
                results_df = pd.read_csv(output_path)
                print(f"Loaded existing results from {output_path}")
            else:
                redo = True

        if redo:
            bs_df = batchsize_prediction_speed_test(
                checkpoint_path=args.ckpt_path,
                dataset_name=args.dataset,
                num_samples=args.batchsize_num_samples,
                batch_sizes=batch_sizes,
                output_dir=output_dir,
            )

        plot_batchsize_prediction(bs_df, output_dir=output_dir)

    print("Done.")
