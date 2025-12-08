from math import log
import torch
from torch_geometric.loader import DataLoader as TGDataLoader
from torch.utils.data import Subset
import argparse
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go

import json
from pathlib import Path
import os
from collections import defaultdict

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
from hip.vibrations_torch import compute_hessian_finite_difference_batched


# https://plotly.com/python/templates/
# ['ggplot2', 'seaborn', 'simple_white', 'plotly',
# 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
# 'ygridoff', 'gridon', 'none']
PLOTLY_TEMPLATE = "plotly_white"


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


def time_forward_pass(model, batch):
    """Times a single forward pass and measures memory usage."""
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    with torch.no_grad():
        if "equiformer" in model.name.lower():
            ener, force, out = model.forward(batch, otf_graph=True, hessian=False)
        else:
            ener, force, out = model.forward(batch)

    end_event.record()
    torch.cuda.synchronize()

    time_taken = start_event.elapsed_time(end_event)
    memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    return time_taken, memory_usage


def time_hessian_computation(model, batch, hessian_method):
    """Times a single hessian computation and measures memory usage."""
    do_autograd = (
        hessian_method == "autograd" or hessian_method == "autograd_conservative"
    )
    do_conservative = hessian_method == "autograd_conservative"
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    if "equiformer" in model.name.lower():
        if do_autograd:
            batch.pos.requires_grad_()
            if do_conservative:
                ener, force, out = model.forward(
                    batch,
                    otf_graph=True,
                    hessian=False,
                    conservative_forces=True,
                    retain_forces_graph=True,
                )
            else:
                ener, force, out = model.forward(batch, otf_graph=True, hessian=False)
            compute_hessian(batch.pos, ener, force)
        else:
            with torch.no_grad():
                ener, force, out = model.forward(
                    batch,
                    otf_graph=True,
                    hessian=True,
                )
    else:
        batch.pos.requires_grad_()
        ener, force, out = model.forward(batch)
        compute_hessian(batch.pos, ener, force)

    end_event.record()
    torch.cuda.synchronize()

    time_taken = start_event.elapsed_time(end_event)
    memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    return time_taken, memory_usage


def time_finite_difference_hessian(model, batch, batch_size):
    """Times finite difference Hessian computation and measures memory usage."""
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(
        enable_timing=True, blocking=False, interprocess=False
    )
    end_event = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)

    start_event.record()

    # Extract positions and atomic numbers from batch
    positions = batch.pos
    atomic_numbers = batch.z
    device = positions.device

    # Compute Hessian using finite differences
    compute_hessian_finite_difference_batched(
        positions=positions,
        atomic_numbers=atomic_numbers,
        model=model,
        device=device,
        indices=None,  # All atoms
        delta=0.01,
        batch_size=batch_size,
    )

    end_event.record()
    torch.cuda.synchronize()

    time_taken = start_event.elapsed_time(end_event)
    memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    return time_taken, memory_usage


def get_model_ckpt(checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()
    model.name = model_name
    return model


def speed_comparison(
    checkpoint_path,
    dataset_name,
    max_samples_per_n,
    device="cuda",
    output_dir="./results_speed",
    output_path=None,
    cutoff=10.0,
    cutoff_hessian=10.0,
    show_ad=True,
    show_fd=True,
    show_fwd=True,
):
    """Compares the speed of various methods for Hessian computation."""
    # Load model
    model = get_model_ckpt(checkpoint_path, device)
    model.cutoff = cutoff
    model.cutoff_hessian = cutoff_hessian

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

    warmup_reps = 3

    results = []

    for n_atoms, indices in tqdm(indices_by_natoms.items(), desc="Processing N_atoms"):
        if len(indices) == 0:
            continue
        n_atoms = int(n_atoms)

        # Limit number of samples
        indices_to_test = indices[: max_samples_per_n + warmup_reps]

        subset = Subset(dataset, indices_to_test)
        loader = TGDataLoader(subset, batch_size=1, shuffle=False)
        i = 0
        for batch in tqdm(loader, desc=f"N={n_atoms}", leave=False):
            # Time finite difference with batch_size=1
            batch = batch.to(device)
            time_fd1, mem_fd1 = time_finite_difference_hessian(
                model, batch, batch_size=1
            )
            if i >= warmup_reps:
                results.append(
                    {
                        "n_atoms": n_atoms,
                        "method": "finite_difference_bz1",
                        "time": time_fd1,
                        "memory": mem_fd1,
                    }
                )
            torch.cuda.empty_cache()
            i += 1

        subset = Subset(dataset, indices_to_test)
        loader = TGDataLoader(subset, batch_size=1, shuffle=False)
        i = 0
        for batch in tqdm(loader, desc=f"N={n_atoms}", leave=False):
            # Time finite difference with batch_size=32
            batch = batch.to(device)
            time_fd32, mem_fd32 = time_finite_difference_hessian(
                model, batch, batch_size=32
            )
            if i >= warmup_reps:
                results.append(
                    {
                        "n_atoms": n_atoms,
                        "method": "finite_difference_bz32",
                        "time": time_fd32,
                        "memory": mem_fd32,
                    }
                )
            torch.cuda.empty_cache()
            i += 1

        subset = Subset(dataset, indices_to_test)
        loader = TGDataLoader(subset, batch_size=1, shuffle=False)
        i = 0
        for batch in tqdm(loader, desc=f"N={n_atoms}", leave=False):
            batch = batch.to(device)
            # Time forward pass
            time_fwd, mem_fwd = time_forward_pass(model, batch)
            if i >= warmup_reps:
                results.append(
                    {
                        "n_atoms": n_atoms,
                        "method": "forward_pass",
                        "time": time_fwd,
                        "memory": mem_fwd,
                    }
                )
            torch.cuda.empty_cache()
            i += 1

        subset = Subset(dataset, indices_to_test)
        loader = TGDataLoader(subset, batch_size=1, shuffle=False)
        i = 0
        for batch in tqdm(loader, desc=f"N={n_atoms}", leave=False):
            # Time prediction
            batch = batch.clone().to(device)
            time_prediction, mem_prediction = time_hessian_computation(
                model, batch, "prediction"
            )
            if i >= warmup_reps:
                results.append(
                    {
                        "n_atoms": n_atoms,
                        "method": "prediction",
                        "time": time_prediction,
                        "memory": mem_prediction,
                    }
                )
            torch.cuda.empty_cache()
            i += 1

        subset = Subset(dataset, indices_to_test)
        loader = TGDataLoader(subset, batch_size=1, shuffle=False)
        i = 0
        for batch in tqdm(loader, desc=f"N={n_atoms}", leave=False):
            # Time autograd
            batch = batch.clone().to(device)
            time_autograd, mem_autograd = time_hessian_computation(
                model, batch, "autograd"
            )
            if i >= warmup_reps:
                results.append(
                    {
                        "n_atoms": n_atoms,
                        "method": "autograd",
                        "time": time_autograd,
                        "memory": mem_autograd,
                    }
                )
            torch.cuda.empty_cache()
            i += 1

        subset = Subset(dataset, indices_to_test)
        loader = TGDataLoader(subset, batch_size=1, shuffle=False)
        i = 0
        for batch in tqdm(loader, desc=f"N={n_atoms}", leave=False):
            # Time autograd with conservative forces
            batch = batch.clone().to(device)
            time_autograd_conservative, mem_autograd_conservative = (
                time_hessian_computation(model, batch, "autograd_conservative")
            )
            if i >= warmup_reps:
                results.append(
                    {
                        "n_atoms": n_atoms,
                        "method": "autograd_conservative",
                        "time": time_autograd_conservative,
                        "memory": mem_autograd_conservative,
                    }
                )
            torch.cuda.empty_cache()
            i += 1

    # Save results
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return results_df


def plot_combined_speed_memory(
    results_df,
    output_dir="./results_speed",
    show_std=False,
    show_ad=True,
    show_ad_cf=True,
    show_fd=True,
    show_fd32=True,
    show_fwd=True,
    show_hip=False,
    logy_time=True,
    logy_memory=True,
    ymin_time=None,
    ymin_memory=None,
    ymax_time=None,
    ymax_memory=None,
):
    from plotly.subplots import make_subplots

    output_dir = Path(output_dir)
    height = 400
    width = height * 2

    # Map method names to colours - ensure consistent colors across both panels
    def _color_for_method(method):
        method_lower = str(method).lower()
        if method_lower == "prediction":
            # color = HESSIAN_METHOD_TO_COLOUR.get("predict")
            color = "#d96001"
        elif method_lower == "autograd":
            color = HESSIAN_METHOD_TO_COLOUR.get("autograd")
        elif method_lower == "autograd_conservative":
            # Use a distinct color for conservative autograd (slightly different shade)
            color = "#9b59b6"  # Purple color to distinguish from regular autograd
        elif method_lower == "forward_pass":
            # Use a distinct color for forward pass
            color = "#68c4af"  # Green-ish color
        elif "finite_difference_bz1" in method_lower:
            # Use a distinct color for FD bz=1
            color = "#ffaaa5"
            # color = "#1b85b8"
            color = "#5a5255"
            # color = "#ffb482"
        elif "finite_difference_bz32" in method_lower:
            # Use a distinct color for FD bz=32
            color = "#ff8b94"
        else:
            # Fallback to default color mapping or a default color
            color = HESSIAN_METHOD_TO_COLOUR.get(method_lower)
            if color is None:
                # Ultimate fallback - use a gray color
                color = "#cfcfcf"
        return color

    # Map method names to line dash patterns for memory subplot
    def _dash_for_method(method):
        """Returns dash pattern for memory subplot to distinguish overlapping lines."""
        method_lower = str(method).lower()
        if method_lower == "prediction":
            # return "dot"  # HIP
            return ""  # HIP
        elif method_lower == "forward_pass":
            if show_hip or show_fd:
                return "dot"
            else:
                return "solid"
        elif "finite_difference" in method_lower:
            return "solid"
        else:
            return "solid"  # All others (autograd, autograd_conservative) - solid

    # Aggregations for speed and memory vs N
    avg_times = results_df.groupby(["n_atoms", "method"])["time"].mean().unstack()
    std_times = results_df.groupby(["n_atoms", "method"])["time"].std().unstack()
    avg_memory = results_df.groupby(["n_atoms", "method"])["memory"].mean().unstack()
    std_memory = results_df.groupby(["n_atoms", "method"])["memory"].std().unstack()

    # Filter methods based on flags
    def should_show_method(method_name):
        method_lower = str(method_name).lower()
        if show_hip and method_lower == "prediction":
            return True  # Always show prediction (HIP)
        if show_fwd and method_lower == "forward_pass":
            return True
        if show_ad and method_lower == "autograd":
            return True
        if show_ad_cf and method_lower == "autograd_conservative":
            return True
        if show_fd and "finite_difference_bz1" in method_lower:
            return True
        if show_fd32 and "finite_difference_bz32" in method_lower:
            return True
        return False

    # Filter columns to only include methods we want to show
    methods_to_keep = [m for m in avg_times.columns if should_show_method(m)]
    avg_times = avg_times[methods_to_keep]
    std_times = std_times[methods_to_keep]
    avg_memory = avg_memory[methods_to_keep]
    std_memory = std_memory[methods_to_keep]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Time per sample (ms)", "Peak Memory (MB)"),
        horizontal_spacing=0.05,
        vertical_spacing=0.0,
    )

    #########################################################
    def _linear_scaling_line(df):
        reference_order = ["prediction", "forward_pass", "autograd"]
        for name in reference_order:
            if name in df.columns:
                series = df[name].dropna()
                if len(series) > 0:
                    base_x = float(series.index[0])
                    base_y = float(series.iloc[0])
                    if base_x > 0 and base_y > 0:
                        x_vals = df.index.to_numpy(dtype=float)
                        return x_vals, base_y * (x_vals / base_x)
        return None, None

    time_linear_x, time_linear_y = (
        _linear_scaling_line(avg_times)
        if logy_time and not avg_times.empty
        else (None, None)
    )
    memory_linear_x, memory_linear_y = (
        _linear_scaling_line(avg_memory)
        if logy_memory and not avg_memory.empty
        else (None, None)
    )

    #########################################################
    # Col 1: Speed vs N
    for method in avg_times.columns:
        color = _color_for_method(method)
        if str(method).lower() == "prediction":
            display_name = "HIP Hessians (ours)"
        elif str(method).lower() == "autograd":
            display_name = "AD Hessians (direct force)"
        elif str(method).lower() == "autograd_conservative":
            display_name = "AD Hessians (conservative)"
        elif str(method).lower() == "forward_pass":
            display_name = "Forward Pass"
        elif "finite_difference_bz1" in str(method).lower():
            display_name = "FD Hessians"
        elif "finite_difference_bz32" in str(method).lower():
            display_name = "FD Hessians (bz=32)"
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
                line=dict(color=color),
                marker=dict(color=color),
                **_err_kwargs,
            ),
            row=1,
            col=1,
        )
    # fig.add_trace(
    #     go.Scatter(
    #         x=time_linear_x,
    #         y=time_linear_y,
    #         mode="lines",
    #         name="Linear scaling",
    #         legend="legend",
    #         showlegend=True,
    #         line=dict(color="#000000", dash="dash"),
    #     ),
    #     row=1,
    #     col=1,
    # )

    # Col 2: Memory vs N
    for method in avg_memory.columns:
        color = _color_for_method(method)
        dash_pattern = _dash_for_method(method)
        if str(method).lower() == "prediction":
            display_name = "Prediction (ours)"
        elif str(method).lower() == "autograd":
            display_name = "AD Hessians"
        elif str(method).lower() == "autograd_conservative":
            display_name = "AD Hessians (conservative)"
        elif str(method).lower() == "forward_pass":
            display_name = "Forward Pass"
        elif "finite_difference_bz1" in str(method).lower():
            display_name = "FD Hessians"
        elif "finite_difference_bz32" in str(method).lower():
            display_name = "FD Hessians (bz=32)"
        else:
            display_name = str(method).capitalize()
        _err_kwargs = {}
        if show_std and (method in std_memory.columns):
            std_vals = std_memory[method].reindex(avg_memory.index)
            if std_vals is not None:
                _err_kwargs = {"error_y": dict(type="data", array=std_vals.values)}

        # Set mode and line based on dash_pattern
        if dash_pattern == "":
            mode = "markers"
            line_dict = None
        else:
            mode = "lines+markers"
            line_dict = dict(color=color, dash=dash_pattern)

        fig.add_trace(
            go.Scatter(
                x=avg_memory.index,
                y=avg_memory[method],
                mode=mode,
                name=display_name,
                showlegend=False,
                line=line_dict,
                marker=dict(color=color),
                **_err_kwargs,
            ),
            row=1,
            col=2,
        )
    # fig.add_trace(
    #     go.Scatter(
    #         x=memory_linear_x,
    #         y=memory_linear_y,
    #         mode="lines",
    #         name="Linear",
    #         showlegend=False,
    #         line=dict(color="#000000", dash="dash"),
    #     ),
    #     row=1,
    #     col=2,
    # )

    # derive subplot domains to place legends at the top-middle of each subplot
    dom1 = fig.layout.xaxis.domain if hasattr(fig.layout, "xaxis") else [0.0, 0.45]
    dom2 = fig.layout.xaxis2.domain if hasattr(fig.layout, "xaxis2") else [0.55, 1.0]
    x2 = 0.5 * (dom2[0] + dom2[1])

    # Add axis titles for each subplot
    fig.update_xaxes(title_text="Number of Atoms (N)", title_standoff=5, row=1, col=1)
    # fig.update_yaxes(title_text="Time per Sample (ms)", title_standoff=10, row=1, col=1)
    fig.update_yaxes(title_text="", title_standoff=10, row=1, col=1)
    fig.update_xaxes(title_text="Number of Atoms (N)", title_standoff=5, row=1, col=2)
    fig.update_yaxes(title_text="", title_standoff=0, row=1, col=2)

    # Build yaxis configs conditionally based on log flags
    yaxis_config = {"showgrid": True}
    if logy_time:
        yaxis_config.update(
            {
                "type": "log",
                "dtick": 1,
                "exponentformat": "power",
                "showexponent": "all",
                "minor": dict(
                    showgrid=True,
                    gridcolor="#eee",
                ),
            }
        )

    yaxis2_config = {"showgrid": True}
    if logy_memory:
        yaxis2_config.update(
            {
                "type": "log",
                "dtick": 1,
                "exponentformat": "power",
                "showexponent": "all",
                "minor": dict(
                    showgrid=True,
                    gridcolor="#eee",
                ),
            }
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=0, b=0, t=20),
        width=width,
        height=height,
        yaxis=yaxis_config,
        yaxis2=yaxis2_config,
        legend=dict(
            x=x2 - 0.08,
            y=0.999,
            xanchor="center",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0.6)",
            font=dict(size=LEGEND_FONT_SIZE - 2),
        ),
    )

    # Set y-axis ranges after layout update (for subplots, use update_yaxes with row/col)
    if ymin_time is not None or ymax_time is not None:
        if len(avg_times) > 0 and not avg_times.empty:
            data_min_time = float(avg_times.min().min())
            data_max_time = float(avg_times.max().max())
        else:
            data_min_time = ymin_time if ymin_time is not None else 1e-3
            data_max_time = ymax_time if ymax_time is not None else data_min_time * 10
        ymin_time_eff = (
            ymin_time if ymin_time is not None else max(data_min_time * 0.9, 1e-9)
        )
        ymax_time_eff = ymax_time if ymax_time is not None else data_max_time * 1.1
        fig.update_yaxes(
            range=[ymin_time_eff, ymax_time_eff], autorange=False, row=1, col=1
        )

    if ymin_memory is not None or ymax_memory is not None:
        if len(avg_memory) > 0 and not avg_memory.empty:
            data_min_memory = float(avg_memory.min().min())
            data_max_memory = float(avg_memory.max().max())
        else:
            data_min_memory = ymin_memory if ymin_memory is not None else 1e-3
            data_max_memory = (
                ymax_memory if ymax_memory is not None else data_min_memory * 10
            )
        ymin_memory_eff = (
            ymin_memory if ymin_memory is not None else max(data_min_memory * 0.9, 1e-9)
        )
        ymax_memory_eff = (
            ymax_memory if ymax_memory is not None else data_max_memory * 1.1
        )
        fig.update_yaxes(
            range=[ymin_memory_eff, ymax_memory_eff], autorange=False, row=1, col=2
        )

    # set xaxis range
    fig.update_xaxes(range=[4.5, 21.5], autorange=False, row=1, col=2)
    fig.update_xaxes(range=[4.5, 21.5], autorange=False, row=1, col=1)

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
        if ann.text in ("Time per sample (ms)", "Peak Memory (MB)"):
            ann.update(font=dict(size=TITLE_FONT_SIZE))

    # Add subplot panel labels (a, b) at top-left outside each subplot
    fig.add_annotation(
        x=dom1[0],
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
    # instead of the yaxis title, because this we can move around more freely
    # fig.add_annotation(
    #     x=max(dom2[0] - 0.019, 0.0),
    #     y=0.5,
    #     xref="paper",
    #     yref="paper",
    #     text="Peak Memory (MB)",
    #     textangle=-90,
    #     showarrow=False,
    #     xanchor="center",
    #     yanchor="middle",
    #     font=dict(size=AXES_TITLE_FONT_SIZE),
    # )

    _name = "speedmemory"
    if show_ad:
        _name += "_ad"
    if show_ad_cf:
        _name += "_ad_cf"
    if show_fd:
        _name += "_fd"
    if show_fd32:
        _name += "_fd32"
    if show_fwd:
        _name += "_fwd"
    if show_hip:
        _name += "_hip"
    if ymin_time is not None:
        _name += f"_ymin{ymin_time}"
    if ymax_time is not None:
        _name += f"_ymax{ymax_time}"
    if ymin_memory is not None:
        _name += f"_ymin{ymin_memory}"
    if ymax_memory is not None:
        _name += f"_ymax{ymax_memory}"
    output_path = output_dir / f"{_name}.png"
    fig.write_image(output_path, width=width, height=height, scale=2)
    print(f"Plot saved to \n {output_path}")


if __name__ == "__main__":
    """
    uv run scriptsp/speed_comparison_extended.py --dataset RGD1.lmdb --max_samples_per_n 10 --ckpt_path ckpt/hip_v2.ckpt
    uv run scriptsp/speed_comparison_extended.py --dataset ts1x-val.lmdb 
    uv run scriptsp/speed_comparison_extended.py --dataset ts1x_hess_train_big.lmdb
    """
    parser = argparse.ArgumentParser(description="Extended speed comparison")

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
        default=100.0,
        help="Cutoff radius for message passing in angstroms.",
    )
    parser.add_argument(
        "--cutoff_hessian",
        type=float,
        default=100.0,
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

    args = parser.parse_args()
    torch.manual_seed(42)

    redo = args.redo

    output_dir = "./results_speed2"
    output_dir = Path(output_dir)
    output_path = (
        output_dir
        / f"{args.dataset}_speed_comparison_extended_{args.max_samples_per_n}_r{args.cutoff}_rh{args.cutoff_hessian}.csv"
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
            cutoff=args.cutoff,
            cutoff_hessian=args.cutoff_hessian,
        )

    #########################################################
    # Plot results - fixed axis ranges
    #########################################################

    # Combined side-by-side plot
    # Plot everything (except fd32)
    plot_combined_speed_memory(
        results_df,
        output_dir=output_dir,
        show_std=args.show_std,
        show_ad=True,
        show_fd=True,
        show_fd32=False,
        show_hip=True,
        show_fwd=True,
        ymax_time=3700,
        ymax_memory=2100,
        logy_time=False,
        logy_memory=False,
        ymin_time=0.0,
        ymin_memory=0.0,
    )
    # Plot non-cf AD vs Fwd
    plot_combined_speed_memory(
        results_df,
        output_dir=output_dir,
        show_std=args.show_std,
        show_hip=False,
        show_ad=True,
        show_ad_cf=False,
        show_fd=False,
        show_fd32=False,
        show_fwd=True,
        ymax_time=3700,
        ymax_memory=2100,
        logy_time=False,
        logy_memory=False,
        ymin_time=0.0,
        ymin_memory=0.0,
    )
    # Plot AD vs Fwd
    plot_combined_speed_memory(
        results_df,
        output_dir=output_dir,
        show_std=args.show_std,
        show_hip=False,
        show_ad=True,
        show_ad_cf=True,
        show_fd=False,
        show_fd32=False,
        show_fwd=True,
        ymax_time=3700,
        ymax_memory=2100,
        logy_time=False,
        logy_memory=False,
        ymin_time=0.0,
        ymin_memory=0.0,
    )
    # Plot baselines (not HIP)
    plot_combined_speed_memory(
        results_df,
        output_dir=output_dir,
        show_std=args.show_std,
        show_hip=False,
        show_ad=True,
        show_fd=True,
        show_fd32=False,
        show_fwd=True,
        ymax_time=3700,
        ymax_memory=2100,
        logy_time=False,
        logy_memory=False,
        ymin_time=0.0,
        ymin_memory=0.0,
    )
    # Plot forward
    plot_combined_speed_memory(
        results_df,
        output_dir=output_dir,
        show_std=args.show_std,
        show_hip=False,
        show_ad=False,
        show_fd=False,
        show_fd32=False,
        show_fwd=True,
        ymax_time=3700,
        ymax_memory=2100,
        logy_time=False,
        logy_memory=False,
        ymin_time=0.0,
        ymin_memory=0.0,
    )

    #########################################################
    # Plot results - variable axis ranges
    #########################################################

    # Plot forward
    plot_combined_speed_memory(
        results_df,
        output_dir=output_dir,
        show_std=args.show_std,
        show_ad=False,
        show_ad_cf=False,
        show_hip=False,
        show_fd=False,
        show_fd32=False,
        show_fwd=True,
        logy_time=False,
        logy_memory=False,
        ymin_time=0.0,
        ymin_memory=0.0,
    )
    # Plot forward vs HIP
    plot_combined_speed_memory(
        results_df,
        output_dir=output_dir,
        show_std=args.show_std,
        show_ad=False,
        show_ad_cf=False,
        show_fd=False,
        show_fd32=False,
        show_fwd=True,
        logy_time=False,
        logy_memory=False,
        ymin_time=0.0,
        ymin_memory=0.0,
    )
    # Plot forward vs AD
    plot_combined_speed_memory(
        results_df,
        output_dir=output_dir,
        show_std=args.show_std,
        show_ad=True,
        show_fd=False,
        show_fd32=False,
        show_fwd=True,
        show_hip=False,
        logy_time=False,
        logy_memory=False,
        ymin_time=0.0,
        ymin_memory=0.0,
    )

    # plot forward, ad conservative, ad direct force, finite difference bz1
    plot_combined_speed_memory(
        results_df,
        output_dir=output_dir,
        show_std=args.show_std,
        show_ad=True,
        show_fd=True,
        show_fd32=False,
        show_fwd=True,
        logy_time=False,
        logy_memory=False,
        ymin_time=0.0,
        ymin_memory=0.0,
    )

    model = get_model_ckpt(args.ckpt_path, device="cuda")
    print(f"Number of layers: {len(model.blocks)}")
    print(f"Number of Hessian layers: {len(model.hessian_layers)}")
    print(f"Cutoff: {model.cutoff}")
    print(f"Cutoff Hessian: {model.cutoff_hessian}")

    #########################################################
    # Calculate and print speedup of forward pass over HIP for each N
    #########################################################
    print("\n" + "=" * 60)
    print("Speedup: Forward Pass / HIP (Prediction)")
    print("=" * 60)
    # Group by n_atoms and method, get mean times
    avg_times = results_df.groupby(["n_atoms", "method"])["time"].mean().unstack()
    # Calculate speedup for each N
    speedups = avg_times["prediction"] / avg_times["forward_pass"]

    print(
        f"{'N (atoms)':<15} {'Forward Pass (ms)':<20} {'HIP (ms)':<15} {'Speedup':<10}"
    )
    print("-" * 60)
    for n_atoms in sorted(speedups.index):
        fwd_time = avg_times.loc[n_atoms, "forward_pass"]
        hip_time = avg_times.loc[n_atoms, "prediction"]
        speedup = speedups.loc[n_atoms]
        print(f"{n_atoms:<15} {fwd_time:<20.2f} {hip_time:<15.2f} {speedup:<10.2f}")

    print("-" * 60)
    print(
        f"{'Average':<15} {avg_times['forward_pass'].mean():<20.2f} {avg_times['prediction'].mean():<15.2f} {speedups.mean():<10.2f}"
    )

    print("\nDone!")
