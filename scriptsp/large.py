import torch
from torch_geometric.loader import DataLoader as TGDataLoader
from torch.utils.data import Subset
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import wandb

import json
from pathlib import Path
import os
from collections import defaultdict

import ase.io
from torch_geometric.data import Batch as TGBatch
from torch_geometric.data import Data as TGData

from hip.training_module import PotentialModule
from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path
from hip.colours import (
    HESSIAN_METHOD_TO_COLOUR,
    ANNOTATION_FONT_SIZE,
    ANNOTATION_BOLD_FONT_SIZE,
    AXES_FONT_SIZE,
    AXES_TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
    TITLE_FONT_SIZE,
)

from plotly.subplots import make_subplots

# https://plotly.com/python/templates/
# ['ggplot2', 'seaborn', 'simple_white', 'plotly',
# 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
# 'ygridoff', 'gridon', 'none']
PLOTLY_TEMPLATE = "plotly_white"


def coord_atoms_to_torch_geometric(coords, atomic_nums):
    """
    Convert coordinates and atomic numbers to torch_geometric Data format expected by Equiformer.

    Args:
        coords: (N, 3) array of atomic positions
        atomic_nums: (N,) array of atomic numbers

    Returns:
        Batch: torch_geometric Batch object with required attributes
    """
    data = TGData(
        pos=torch.as_tensor(coords, dtype=torch.float32).reshape(-1, 3),
        z=torch.as_tensor(atomic_nums, dtype=torch.int64),
        charges=torch.as_tensor(atomic_nums, dtype=torch.int64),
        natoms=torch.tensor([len(atomic_nums)], dtype=torch.int64),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )
    return TGBatch.from_data_list([data])


def load_large_molecules(data_large_dir="data/large", maxnatoms=None):
    """
    Load molecules from data/large/ directory.

    Args:
        data_large_dir: Path to the data/large directory
        maxnatoms: Maximum number of atoms to load (None = no limit)

    Returns:
        dict: Dictionary mapping n_atoms to list of batch objects
    """
    data_large_path = Path(data_large_dir)
    molecules_file = data_large_path / "molecules.txt"

    if not molecules_file.exists():
        print(f"Warning: molecules.txt not found at {molecules_file}")
        return {}

    # Read molecules.txt
    molecules_by_natoms = defaultdict(list)

    with open(molecules_file, "r") as f:
        lines = f.readlines()
        # Skip header line
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name = parts[0]
            n_atoms = int(parts[1])

            # Skip if exceeds maxnatoms limit
            if maxnatoms is not None and n_atoms > maxnatoms:
                print(
                    f"Skipping {name} ({n_atoms} atoms) - exceeds maxnatoms limit ({maxnatoms})"
                )
                continue

            file_format = parts[2] if len(parts) > 2 else "sdf"

            # Construct file path
            if file_format.lower() == "sdf":
                file_path = data_large_path / f"{name}.sdf"
            elif file_format.lower() == "pdb":
                file_path = data_large_path / f"{name}.pdb"
            else:
                print(f"Warning: Unknown format {file_format} for {name}, skipping")
                continue

            if not file_path.exists():
                print(f"Warning: File not found: {file_path}, skipping")
                continue

            # Load molecule using ASE
            atoms = ase.io.read(str(file_path))
            coords = atoms.get_positions()
            atomic_nums = atoms.get_atomic_numbers()

            # Convert to torch geometric format
            batch = coord_atoms_to_torch_geometric(coords, atomic_nums)
            molecules_by_natoms[n_atoms].append(batch)
            print(f"Loaded {n_atoms} atoms molecule {name}")

    # Sort keys
    sorted_molecules_by_natoms = {k: v for k, v in sorted(molecules_by_natoms.items())}

    print(
        f"Loaded {sum(len(v) for v in sorted_molecules_by_natoms.values())} large molecules"
    )
    return sorted_molecules_by_natoms


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


def time_hessian_computation(model, batch, hessian_method, cutoff, cutoff_hessian):
    """Times a single hessian computation and measures memory usage."""
    do_autograd = hessian_method == "autograd"

    model.cutoff = cutoff
    model.cutoff_hessian = cutoff_hessian

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    if do_autograd:
        batch.pos.requires_grad_()
        ener, force, out = model.forward(batch, otf_graph=True, hessian=False)
        compute_hessian(batch.pos, ener, force)
    else:
        with torch.no_grad():
            # for a fair comparison
            # compute graph and Hessian indices on the fly
            ener, force, out = model.forward(
                batch, otf_graph=True, hessian=True, add_props=True
            )

    end_event.record()
    torch.cuda.synchronize()

    time_taken = start_event.elapsed_time(end_event)
    memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB

    return time_taken, memory_usage


def speed_comparison(
    checkpoint_path,
    dataset_name,
    max_samples_per_n,
    device,
    output_dir,
    output_path,
    cutoff,
    cutoff_hessian,
    maxnatoms,
    maxnatoms_fc,  # fully connected
    maxnatoms_ad,  # autograd
    largerepeat=1,  # number of times to repeat large molecule samples
):
    """Compares the speed of autograd vs prediction for Hessian computation."""
    # Load existing results if available
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    existing_results = []
    processed_natoms = {
        "prediction": set(),
        "prediction_fc": set(),
        "autograd": set(),
    }

    if output_path.exists():
        print(f"Loading existing results from {output_path}")
        existing_df = pd.read_csv(output_path)
        existing_results = existing_df.to_dict("records")
        # Track which natoms have been processed for each method separately
        for n_atoms in existing_df["n_atoms"].unique():
            n_atoms = int(n_atoms)
            natoms_data = existing_df[existing_df["n_atoms"] == n_atoms]

            # Check each method separately
            for method in ["prediction", "prediction_fc", "autograd"]:
                method_data = natoms_data[natoms_data["method"] == method]
                if len(method_data) == 0:
                    continue

                # For regular dataset (n_atoms <= 100), need max_samples_per_n samples
                # For large molecules (n_atoms > 100), need at least 1 sample
                required_samples = max_samples_per_n if n_atoms <= 100 else 1

                # Check if method is applicable for this natoms
                is_applicable = True
                if method == "prediction" and n_atoms >= maxnatoms:
                    is_applicable = False
                elif method == "prediction_fc" and n_atoms >= maxnatoms_fc:
                    is_applicable = False
                elif method == "autograd" and n_atoms >= maxnatoms_ad:
                    is_applicable = False

                # If method is applicable and we have enough samples, mark as processed
                if is_applicable and len(method_data) >= required_samples:
                    processed_natoms[method].add(n_atoms)

        print(f"Found {len(existing_results)} existing results")
        print(
            f"Processed natoms per method: "
            f"prediction={len(processed_natoms['prediction'])}, "
            f"prediction_fc={len(processed_natoms['prediction_fc'])}, "
            f"autograd={len(processed_natoms['autograd'])}"
        )

        # Filter out partial results - keep only results for natoms that are processed for that method
        existing_results = [
            r
            for r in existing_results
            if int(r["n_atoms"]) in processed_natoms.get(r["method"], set())
        ]
        print(
            f"Keeping {len(existing_results)} results from processed natoms per method"
        )

    # Load model
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()
    model.name = model_name
    model.max_neighbors = 100000

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
    dataset = LmdbDataset(fix_dataset_path(dataset_name))

    # do a couple of forward passes to warm up the model
    # populate caches, jit, load cuda kernels, and what not
    loader = TGDataLoader(dataset, batch_size=1, shuffle=False)
    for i, sample in enumerate(loader):
        batch = sample.to(device)
        time_hessian_computation(
            model, batch, "prediction", cutoff=cutoff, cutoff_hessian=cutoff_hessian
        )

        time_hessian_computation(
            model, batch, "autograd", cutoff=cutoff, cutoff_hessian=cutoff_hessian
        )

        if i > 10:
            break
    print("Model warmed up")

    results = existing_results.copy()

    for n_atoms, indices in tqdm(indices_by_natoms.items(), desc="Processing N_atoms"):
        if len(indices) == 0:
            continue
        n_atoms = int(n_atoms)

        # Check which methods still need processing
        needs_prediction = n_atoms not in processed_natoms["prediction"]
        needs_prediction_fc = n_atoms not in processed_natoms["prediction_fc"]
        needs_autograd = n_atoms not in processed_natoms["autograd"]

        # Skip if all methods are already processed
        if not (needs_prediction or needs_prediction_fc or needs_autograd):
            print(f"Skipping N={n_atoms} (all methods already processed)")
            continue

        # Limit number of samples
        indices_to_test = indices[:max_samples_per_n]

        subset = Subset(dataset, indices_to_test)
        loader = TGDataLoader(subset, batch_size=1, shuffle=False)

        natoms_results = []
        for _batch in tqdm(loader, desc=f"N={n_atoms}", leave=False):
            ################ Prediction ################
            if needs_prediction:
                # fresh batch
                batch = _batch.clone().to(device)

                # Time prediction
                time_prediction, mem_prediction = time_hessian_computation(
                    model,
                    batch,
                    "prediction",
                    cutoff=cutoff,
                    cutoff_hessian=cutoff_hessian,
                )
                natoms_results.append(
                    {
                        "n_atoms": n_atoms,
                        "n_edges": batch.edge_index.shape[1],  # [E, 2]
                        "method": "prediction",
                        "time": time_prediction,
                        "memory": mem_prediction,
                    }
                )

            ################ Fully Connected ################
            if needs_prediction_fc:
                # fresh batch
                batch = _batch.clone().to(device)

                # Time prediction
                time_prediction, mem_prediction = time_hessian_computation(
                    model, batch, "prediction", cutoff=1e8, cutoff_hessian=1e8
                )
                natoms_results.append(
                    {
                        "n_atoms": n_atoms,
                        "n_edges": batch.edge_index.shape[1],  # [E, 2]
                        "method": "prediction_fc",
                        "time": time_prediction,
                        "memory": mem_prediction,
                    }
                )

            ################ AD ################
            if needs_autograd:
                # fresh batch
                batch = _batch.clone().to(device)

                # Time autograd
                time_autograd, mem_autograd = time_hessian_computation(
                    model,
                    batch,
                    "autograd",
                    cutoff=cutoff,
                    cutoff_hessian=cutoff_hessian,
                )
                natoms_results.append(
                    {
                        "n_atoms": n_atoms,
                        "n_edges": batch.edge_index.shape[1],  # [E, 2]
                        "method": "autograd",
                        "time": time_autograd,
                        "memory": mem_autograd,
                    }
                )

        # Add results for this natoms and save immediately
        results.extend(natoms_results)
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"Saved results for N={n_atoms} to {output_path}")

    # Process large molecules (only prediction method)
    print("\nProcessing large molecules...")
    large_molecules = load_large_molecules(maxnatoms=maxnatoms)
    if large_molecules:
        # Track which large molecules (n_atoms > 100) have been processed per method
        # Since there's only one sample per n_atoms for large molecules, we just check if it exists
        processed_large_natoms = defaultdict(set)
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            for _, row in existing_df.iterrows():
                n_atoms = int(row["n_atoms"])
                if n_atoms > 100:  # Large molecule
                    method = row["method"]
                    processed_large_natoms[method].add(n_atoms)

        pbar = tqdm(large_molecules.items(), desc="Processing large molecules")
        for n_atoms, batches in pbar:
            n_atoms = int(n_atoms)
            # For large molecules, there's only one batch per n_atoms
            _batch = batches[0]

            # Check which methods still need processing
            needs_prediction = (
                n_atoms < maxnatoms
                and n_atoms not in processed_large_natoms["prediction"]
            )
            needs_prediction_fc = (
                n_atoms < maxnatoms_fc
                and n_atoms not in processed_large_natoms["prediction_fc"]
            )
            needs_autograd = (
                n_atoms < maxnatoms_ad
                and n_atoms not in processed_large_natoms["autograd"]
            )

            # Skip if all applicable methods are already processed
            if not (needs_prediction or needs_prediction_fc or needs_autograd):
                print(
                    f"Skipping large molecule N={n_atoms} (all applicable methods already processed)"
                )
                continue

            pbar.set_description(f"Processing large molecules (N={n_atoms})")
            natoms_results = []
            print(f"N={n_atoms}", flush=True)
            
            ################ Prediction ################
            if needs_prediction:
                batch = _batch.clone().to(device)
                times = []
                memories = []
                
                # Repeat computation largerepeat times
                for _ in range(largerepeat):
                    time_prediction, mem_prediction = time_hessian_computation(
                        model,
                        batch,
                        "prediction",
                        cutoff=cutoff,
                        cutoff_hessian=cutoff_hessian,
                    )
                    times.append(time_prediction)
                    memories.append(mem_prediction)
                
                # Average results
                avg_time = sum(times) / len(times)
                avg_memory = sum(memories) / len(memories)
                
                natoms_results.append(
                    {
                        "n_atoms": n_atoms,
                        "n_edges": batch.edge_index.shape[1],  # [E, 2]
                        "method": "prediction",
                        "time": avg_time,
                        "memory": avg_memory,
                    }
                )

            ################ Fully Connected ################
            if needs_prediction_fc:
                batch = _batch.clone().to(device)
                times = []
                memories = []
                
                # Repeat computation largerepeat times
                for _ in range(largerepeat):
                    time_prediction, mem_prediction = time_hessian_computation(
                        model, batch, "prediction", cutoff=1e8, cutoff_hessian=1e8
                    )
                    times.append(time_prediction)
                    memories.append(mem_prediction)
                
                # Average results
                avg_time = sum(times) / len(times)
                avg_memory = sum(memories) / len(memories)
                
                natoms_results.append(
                    {
                        "n_atoms": n_atoms,
                        "n_edges": batch.edge_index.shape[1],  # [E, 2]
                        "method": "prediction_fc",
                        "time": avg_time,
                        "memory": avg_memory,
                    }
                )

            ################ AD ################
            if needs_autograd:
                batch = _batch.clone().to(device)
                times = []
                memories = []
                
                # Repeat computation largerepeat times
                for _ in range(largerepeat):
                    time_autograd, mem_autograd = time_hessian_computation(
                        model,
                        batch,
                        "autograd",
                        cutoff=cutoff,
                        cutoff_hessian=cutoff_hessian,
                    )
                    times.append(time_autograd)
                    memories.append(mem_autograd)
                
                # Average results
                avg_time = sum(times) / len(times)
                avg_memory = sum(memories) / len(memories)
                
                natoms_results.append(
                    {
                        "n_atoms": n_atoms,
                        "n_edges": batch.edge_index.shape[1],  # [E, 2]
                        "method": "autograd",
                        "time": avg_time,
                        "memory": avg_memory,
                    }
                )

            # Add results for this natoms and save immediately
            results.extend(natoms_results)
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            print(f"Saved results for large molecule N={n_atoms} to {output_path}")

    # Final save (results already saved incrementally, but ensure everything is saved)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Final results saved to {output_path} (total: {len(results)} entries)")
    return results_df


def plot_speed_comparison(results_df, output_dir="./results_speed"):
    output_dir = Path(output_dir)
    # Plot results for speed
    avg_times = results_df.groupby(["n_atoms", "method"])["time"].mean().unstack()

    fig = go.Figure()
    for method in avg_times.columns:
        fig.add_trace(
            go.Scatter(
                x=avg_times.index,
                y=avg_times[method],
                mode="lines+markers",
                name=method,
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
    # Increase line width slightly for readability
    fig.update_traces(line=dict(width=3))
    output_path = output_dir / "speed_comparison_plot.html"
    fig.write_html(output_path)
    print(f"Plot saved to \n{output_path}")
    output_path = output_dir / "speed_comparison_plot.png"
    fig.write_image(output_path, scale=2)
    print(f"Plot saved to \n{output_path}")

    # Plot results for memory
    plot_memory_usage(results_df, output_dir)


def plot_memory_usage(results_df, output_dir="./results_speed"):
    output_dir = Path(output_dir)
    avg_memory = results_df.groupby(["n_atoms", "method"])["memory"].mean().unstack()

    fig = go.Figure()
    for method in avg_memory.columns:
        fig.add_trace(
            go.Scatter(
                x=avg_memory.index,
                y=avg_memory[method],
                mode="lines+markers",
                name=method,
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
    # Increase line width slightly for readability
    fig.update_traces(line=dict(width=3))
    output_path = output_dir / "memory_usage_plot.html"
    fig.write_html(output_path)
    print(f"Plot saved to \n {output_path}")
    output_path = output_dir / "memory_usage_plot.png"
    fig.write_image(output_path, scale=2)
    print(f"Plot saved to \n {output_path}")


def plot_combined_speed_memory(
    results_df,
    output_dir,
    show_std,
    cutoff,
    cutoff_hessian,
):
    cutoff = int(cutoff)
    cutoff_hessian = int(cutoff_hessian)

    output_dir = Path(output_dir)
    height = 400
    width = height * 2

    # Map method names to colours (handle both "predict" and "prediction")
    def _color_for_method(method):
        key = method
        return HESSIAN_METHOD_TO_COLOUR.get(key)

    # Aggregations for speed and memory vs N
    avg_times = results_df.groupby(["n_atoms", "method"])["time"].mean().unstack()
    std_times = results_df.groupby(["n_atoms", "method"])["time"].std().unstack()
    avg_memory = results_df.groupby(["n_atoms", "method"])["memory"].mean().unstack()
    std_memory = results_df.groupby(["n_atoms", "method"])["memory"].std().unstack()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Time (Single Sample)", "Memory"),
        horizontal_spacing=0.05,
        vertical_spacing=0.0,
    )

    #########################################################
    # Col 1: Speed vs N
    for method in avg_times.columns:
        color = _color_for_method(method)
        if str(method).lower() == "prediction":
            display_name = f"HIP Hessians (r={cutoff})"
        elif str(method).lower() == "prediction_fc":
            display_name = f"HIP Hessians (dense)"
        elif str(method).lower() == "autograd":
            display_name = f"AD Hessians (r={cutoff})"
        else:
            display_name = str(method).capitalize()

        _err_kwargs = {}
        if show_std and (method in std_times.columns):
            std_vals = std_times[method].reindex(avg_times.index)
            if std_vals is not None:
                _err_kwargs = {"error_y": dict(type="data", array=std_vals.values)}
        fig.add_trace(
            go.Scatter(
                x=avg_times.index,
                y=avg_times[method],
                mode="lines+markers",
                name=display_name,
                legend="legend",
                showlegend=True,
                line=dict(color=color) if color else None,
                marker=dict(color=color) if color else None,
                **_err_kwargs,
            ),
            row=1,
            col=1,
        )

    # Col 2: Memory vs N
    for method in avg_memory.columns:
        color = _color_for_method(method)
        display_name = (
            "Prediction (ours)"
            if str(method).lower() == "prediction"
            else str(method).capitalize()
        )
        _err_kwargs = {}
        if show_std and (method in std_memory.columns):
            std_vals = std_memory[method].reindex(avg_memory.index)
            if std_vals is not None:
                _err_kwargs = {"error_y": dict(type="data", array=std_vals.values)}
        fig.add_trace(
            go.Scatter(
                x=avg_memory.index,
                y=avg_memory[method],
                mode="lines+markers",
                name=display_name,
                # legend="legend2",
                showlegend=False,
                line=dict(color=color) if color else None,
                marker=dict(color=color) if color else None,
                **_err_kwargs,
            ),
            row=1,
            col=2,
        )

    # derive subplot domains to place legends at the top-middle of each subplot
    dom1 = fig.layout.xaxis.domain if hasattr(fig.layout, "xaxis") else [0.0, 0.45]
    dom2 = fig.layout.xaxis2.domain if hasattr(fig.layout, "xaxis2") else [0.55, 1.0]
    x2 = 0.5 * (dom2[0] + dom2[1])

    # Add axis titles for each subplot
    fig.update_xaxes(title_text="Number of Atoms (N)", title_standoff=5, row=1, col=1)
    fig.update_yaxes(title_text="Average Time (ms)", title_standoff=10, row=1, col=1)
    # fig.update_yaxes(tickformat=".0e", exponentformat="e",  row=1, col=1)
    fig.update_xaxes(title_text="Number of Atoms (N)", title_standoff=5, row=1, col=2)
    fig.update_yaxes(title_text="Peak Memory (MB)", title_standoff=0, row=1, col=2)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=0, b=0, t=20),
        # margin=dict(l=0, r=0, b=0, t=0),
        width=width,
        height=height,
        yaxis=dict(
            type="log",
            dtick=1,  # ticks are set every 10^(n"dtick)
            exponentformat="power",  # "none" | "e" | "E" | "power" | "SI" | "B"
            showexponent="all",
            showgrid=True,
            minor=dict(
                showgrid=True,
                gridcolor="#eee",  # "rgba(50,50,50,0.1)",
                # gridwidth=1,
            ),
        ),
        legend=dict(
            x=x2 - 0.07,
            y=0.98,
            xanchor="center",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0.6)",
            font=dict(size=LEGEND_FONT_SIZE),
            # title_text="Hessian",
        ),
    )
    # Increase line width slightly for readability across all subplots
    fig.update_traces(line=dict(width=3))

    # Increase global font sizes for axes and annotations
    fig.update_xaxes(
        tickfont=dict(size=AXES_FONT_SIZE), title_font=dict(size=AXES_TITLE_FONT_SIZE)
    )
    fig.update_yaxes(
        tickfont=dict(size=AXES_FONT_SIZE), title_font=dict(size=AXES_TITLE_FONT_SIZE)
    )
    fig.update_annotations(font=dict(size=ANNOTATION_FONT_SIZE))

    # Set subplot title fonts specifically to TITLE_FONT_SIZE
    for ann in fig.layout.annotations:
        if ann.text in ("Time (Single Sample)", "Memory"):
            ann.update(font=dict(size=TITLE_FONT_SIZE))

    # Add subplot panel labels (a, b) at top-left outside each subplot
    fig.add_annotation(
        x=dom1[0],  # -0.005
        y=0.999,
        xref="paper",
        yref="paper",
        text="<b>a</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )
    fig.add_annotation(
        x=dom2[0],
        y=0.999,
        xref="paper",
        yref="paper",
        text="<b>b</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )

    _name = "speedmemorylarge"
    # if show_ad:
    #     _name += "_ad"
    # if show_fd:
    #     _name += "_fd"
    # if show_fwd:
    #     _name += "_fwd"
    output_path = output_dir / f"{_name}.png"
    # The height of the exported image in layout pixels. If the scale property is 1.0, this will also be the height of the exported image in physical pixels.
    # Scale > 1 increases the image resolution
    fig.write_image(output_path, width=width, height=height, scale=2)
    print(f"Plot saved to \n {output_path}")


if __name__ == "__main__":
    """
    python scripts/speed_comparison.py --dataset RGD1.lmdb --max_samples_per_n 10 --ckpt_path ckpt/hip_v2.ckpt
    python scripts/speed_comparison.py --dataset ts1x-val.lmdb --max_samples_per_n 100
    python scripts/speed_comparison.py --dataset ts1x_hess_train_big.lmdb --max_samples_per_n 1000
    """
    parser = argparse.ArgumentParser(description="Speed comparison")

    # Subparser for speed comparison
    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        # default="ckpt/eqv2.ckpt",
        default="ckpt/hip_v2.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ts1x-val.lmdb",
        help="Dataset file name",
    )
    parser.add_argument(
        "--max_samples_per_n",
        type=int,
        default=10,
        help="Maximum number of samples per N atoms to test.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=16.0,
        help="Cutoff radius for message passing in angstroms.",
    )
    parser.add_argument(
        "--cutoff_hessian",
        type=float,
        default=16.0,
        help="Cutoff radius for Hessian computation in angstroms.",
    )
    parser.add_argument(
        "--redo",
        type=bool,
        default=False,
        help="Redo the speed comparison. If false attempt to load existing results.",
    )
    parser.add_argument(
        "--show_std",
        type=bool,
        default=False,
        help="Show standard deviation as error bars in the combined plot.",
    )
    parser.add_argument(
        "--maxnatoms",
        type=int,
        default=200,
        help="Maximum number of atoms for large molecules (None = no limit). Prevents OOM errors.",
    )
    parser.add_argument(
        "--maxnatoms_fc",
        type=int,
        default=170,
        help="Maximum number of atoms for fully connected Hessian computation.",
    )
    parser.add_argument(
        "--maxnatoms_ad",
        type=int,
        default=100,
        help="Maximum number of atoms for autograd Hessian computation.",
    )
    parser.add_argument(
        "--usewandb",
        type=bool,
        default=True,
        help="Log results and images to Weights & Biases.",
    )
    parser.add_argument(
        "--largerepeat",
        type=int,
        default=1,
        help="Number of times to repeat the same large molecule sample to get an average timing.",
    )
    """
    uv run scriptsp/large.py --redo True --maxnatoms 250 --maxnatoms_fc 170 --maxnatoms_ad 100
    """

    args = parser.parse_args()
    torch.manual_seed(42)


    redo = args.redo

    output_dir = "./results_speed"
    output_dir = Path(output_dir)
    output_path = (
        output_dir
        / f"{args.dataset}_speedmemory_large_results_{args.max_samples_per_n}_r{args.cutoff}_rh{args.cutoff_hessian}.csv"
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
            device="cuda",
            output_dir=output_dir,
            output_path=output_path,
            cutoff=args.cutoff,
            cutoff_hessian=args.cutoff_hessian,
            # Maximum number of atoms for large molecules (None = no limit). Prevents OOM errors.
            maxnatoms=args.maxnatoms,
            maxnatoms_fc=args.maxnatoms_fc,
            maxnatoms_ad=args.maxnatoms_ad,
            largerepeat=args.largerepeat,
        )

    # Plot results
    # plot_speed_comparison(results_df)

    # Combined side-by-side plot
    plot_combined_speed_memory(
        results_df,
        output_dir=output_dir,
        show_std=args.show_std,
        cutoff=args.cutoff,
        cutoff_hessian=args.cutoff_hessian,
    )

    if args.usewandb:
        wandb.init(
            project="hip-speed-comparison",
            config={
                "dataset": args.dataset,
                "max_samples_per_n": args.max_samples_per_n,
                "cutoff": args.cutoff,
                "cutoff_hessian": args.cutoff_hessian,
                "maxnatoms": args.maxnatoms,
                "maxnatoms_fc": args.maxnatoms_fc,
                "maxnatoms_ad": args.maxnatoms_ad,
                "ckpt_path": args.ckpt_path,
                "largerepeat": args.largerepeat,
            },
        )
        # Log results as a wandb table
        results_table = wandb.Table(dataframe=results_df)
        wandb.log({"results_table": results_table})

        # Log the generated image
        image_path = output_dir / "speedmemorylarge.png"
        if image_path.exists():
            wandb.log({"speed_memory_plot": wandb.Image(str(image_path))})

    print("\nDone!")
