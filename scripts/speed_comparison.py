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

from hip.training_module import PotentialModule, compute_extra_props
from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path
from ocpmodels.hessian_graph_transform import FOLLOW_BATCH
from hip.training_module import SchemaUniformDataset
from hip.colours import (
    HESSIAN_METHOD_TO_COLOUR,
    ANNOTATION_FONT_SIZE,
    ANNOTATION_BOLD_FONT_SIZE,
    AXES_FONT_SIZE,
    AXES_TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
    TITLE_FONT_SIZE,
)


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
    do_hip = hessian_method == "hip"
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    with torch.no_grad():
        # compute graph and Hessian indices on the fly
        ener, force, out = model.forward(
            batch, otf_graph=True, hessian=do_hip, add_props=True
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
    output_dir,
    output_path,
    device="cuda",
):
    """Compares the speed of autograd vs hip for Hessian computation."""
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
    dataset = LmdbDataset(fix_dataset_path(dataset_name))
    dataset = SchemaUniformDataset(dataset)

    # do a couple of forward passes to warm up the model
    # populate caches, jit, load cuda kernels, and what not
    loader = TGDataLoader(
        dataset, batch_size=1, shuffle=False, 
        # follow_batch=FOLLOW_BATCH
    )
    for i, sample in enumerate(loader):
        batch = sample.to(device)
        batch = compute_extra_props(batch, pos_require_grad=False)
        time_hessian_computation(model, batch, "hip")
        torch.cuda.empty_cache()
        time_hessian_computation(model, batch, "no_hessian")
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
            subset, batch_size=1, shuffle=False, 
            # follow_batch=FOLLOW_BATCH
        )

        for _batch in tqdm(loader, desc=f"N={n_atoms}", leave=False):
            batch = _batch.clone().to(device)
            batch = compute_extra_props(batch, pos_require_grad=False)

            # Time hip (with hessian)
            time_hip, mem_hip = time_hessian_computation(
                model, batch, "hip"
            )
            results.append(
                {
                    "n_atoms": n_atoms,
                    "method": "hip",
                    "time": time_hip,
                    "memory": mem_hip,
                }
            )

            # clear memory
            torch.cuda.empty_cache()

            # fresh batch
            batch = _batch.clone().to(device)
            batch = compute_extra_props(batch, pos_require_grad=False)

            # Time no hessian
            time_no_hessian, mem_no_hessian = time_hessian_computation(
                model, batch, "no_hessian"
            )
            results.append(
                {
                    "n_atoms": n_atoms,
                    "method": "no_hessian",
                    "time": time_no_hessian,
                    "memory": mem_no_hessian,
                }
            )

            # clear memory
            torch.cuda.empty_cache()

    # Save results
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return results_df


def plot_speed_comparison(results_df, output_dir):
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
        title="Hessian Computation Speed: hip vs. No Hessian",
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


def plot_memory_usage(results_df, output_dir):
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
        title="Hessian Computation Memory Usage: hip vs. No Hessian",
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
    print(f"Plot saved to \n{output_path}")
    output_path = output_dir / "memory_usage_plot.png"
    fig.write_image(output_path, scale=2)
    print(f"Plot saved to \n{output_path}")


# ---------------------------------
# hip vs Batch Size Benchmark
# ---------------------------------
def hip_batchsize_benchmark(
    checkpoint_path,
    dataset_name,
    # bz 128 only sometimes fits into memory of a RTX3060
    batch_sizes=(1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64),
    num_batches=10,
    device="cuda",
    output_path="./results_speed/speed_bz.csv",
):
    """Benchmark Hessian hip speed vs batch size using random batches of any N atoms.

    Returns a DataFrame with columns: batch_size, time, memory
    """
    # Load model
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()
    model.name = model_name

    # Prepare dataset (same transform settings as speed_comparison)
    dataset = LmdbDataset(fix_dataset_path(dataset_name))
    dataset = SchemaUniformDataset(dataset)

    # Light warm-up
    warm_loader = TGDataLoader(
        dataset, batch_size=1, shuffle=True, 
        # follow_batch=FOLLOW_BATCH
    )
    for i, sample in enumerate(warm_loader):
        batch = sample.to(device)
        batch = compute_extra_props(batch, pos_require_grad=False)
        time_hessian_computation(model, batch, "hip")
        torch.cuda.empty_cache()
        time_hessian_computation(model, batch, "no_hessian")
        torch.cuda.empty_cache()
        if i >= 5:
            break
    del warm_loader
    # gc.collect()
    # torch.cuda.empty_cache()

    results = []
    dataset_len = len(dataset)
    for bsz in batch_sizes:
        print(f"\n# Batch size: {bsz}")
        # Prepare a subset with random indices; allow duplicates via randint
        num_needed = num_batches * bsz
        if dataset_len == 0:
            break
        rand_idx = torch.randint(
            low=0, high=dataset_len, size=(num_needed,), dtype=torch.long
        ).tolist()
        subset = Subset(dataset, rand_idx)
        loader = TGDataLoader(
            subset, batch_size=bsz, shuffle=True, follow_batch=FOLLOW_BATCH
        )

        torch.cuda.empty_cache()

        measured = 0
        for sample in loader:
            batch = sample.clone().to(device)
            batch = compute_extra_props(batch, pos_require_grad=False)

            # Time hip
            time_hip, mem_hip = time_hessian_computation(
                model, batch, "hip"
            )
            results.append(
                {
                    # "n_atoms": n_atoms,
                    "method": "hip",
                    "time": time_hip,
                    "memory": mem_hip,
                    "batch_size": bsz,
                }
            )

            # clear memory
            torch.cuda.empty_cache()

            # fresh batch
            batch = sample.clone().to(device)
            batch = compute_extra_props(batch, pos_require_grad=False)

            # Time no hessian
            time_no_hessian, mem_no_hessian = time_hessian_computation(
                model, batch, "no_hessian"
            )
            results.append(
                {
                    # "n_atoms": n_atoms,
                    "method": "no_hessian",
                    "time": time_no_hessian,
                    "memory": mem_no_hessian,
                    "batch_size": bsz,
                }
            )

            # clear memory
            torch.cuda.empty_cache()

            msg = f"Bz={bsz}, avg natoms={batch.natoms.clone().to(torch.float32).mean():.1f}"
            msg += f", pred={time_hip:.1f} ms"
            msg += f", no_hess={time_no_hessian:.1f} ms"
            print(msg)

            measured += 1
            if measured >= num_batches:
                break

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Batch size benchmark results saved to {output_path}")
    return results_df


def plot_hip_batchsize(results_df, output_dir, logy=False):
    output_dir = Path(output_dir)
    width = 1600
    height = 450
    fig = go.Figure()
    # hip
    pred_results_df = results_df[results_df["method"] == "hip"]
    avg_times = pred_results_df.groupby(["batch_size"])["time"].mean()
    fig.add_trace(
        go.Scatter(
            x=avg_times.index,
            y=avg_times.values,
            mode="lines+markers",
            name="hip",
            # error_y=dict(type="data", array=std_times.values),
        )
    )
    # no_hessian
    no_hessian_results_df = results_df[results_df["method"] == "no_hessian"]
    avg_times_no_hessian = no_hessian_results_df.groupby(["batch_size"])["time"].mean()
    fig.add_trace(
        go.Scatter(
            x=avg_times_no_hessian.index,
            y=avg_times_no_hessian.values,
            mode="lines+markers",
            name="no_hessian",
        )
    )
    fig.update_layout(
        title="hip Speed vs Batch Size",
        xaxis_title="Batch Size",
        yaxis_title="Average Time (ms)",
        legend_title="Method",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
    )
    # Increase line width slightly for readability
    fig.update_traces(line=dict(width=3))
    # output_path = output_dir / "hip_batchsize_plot.html"
    # fig.write_html(output_path)
    # print(f"Plot saved to \n{output_path}")
    output_path = output_dir / "hip_batchsize_plot.png"
    fig.write_image(output_path, width=width, height=height, scale=2)
    print(f"Plot saved to \n{output_path}")

    #######################
    # plot time per sample
    fig = go.Figure()
    # hip
    avg_times = (
        pred_results_df.groupby(["batch_size"])["time"].mean()
        / pred_results_df.groupby(["batch_size"])["batch_size"].mean()
    )
    fig.add_trace(
        go.Scatter(
            x=avg_times.index,
            y=avg_times.values,
            mode="lines+markers",
            name="hip",
            # error_y=dict(type="data", array=std_times.values),
        )
    )
    # no_hessian
    avg_times_no_hessian = (
        no_hessian_results_df.groupby(["batch_size"])["time"].mean()
        / no_hessian_results_df.groupby(["batch_size"])["batch_size"].mean()
    )
    fig.add_trace(
        go.Scatter(
            x=avg_times_no_hessian.index,
            y=avg_times_no_hessian.values,
            mode="lines+markers",
            name="no_hessian",
        )
    )
    fig.update_layout(
        # title="Hessian hip Speed vs Batch Size",
        xaxis_title="Batch Size",
        yaxis_title="Average Time per Sample (ms)",
        legend_title="Method",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, b=40, t=40),
        width=width,
        height=height,
        legend=dict(
            x=0.5,
            y=0.98,
            xanchor="center",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0.6)",
        ),
        yaxis=dict(type="log") if logy else None,
    )
    # Increase line width slightly for readability
    fig.update_traces(line=dict(width=3))
    # output_path = output_dir / "hip_batchsize_plot.html"
    # fig.write_html(output_path)
    # print(f"Plot saved to \n{output_path}")
    output_path = (
        output_dir
        / f"hip_batchsize_time_per_sample{'_logy' if logy else ''}.png"
    )
    fig.write_image(output_path, scale=2)
    print(f"Plot saved to \n{output_path}")


def plot_combined_speed_memory_batchsize(
    results_df, bz_results_df, output_dir, show_std=False, _name=""
):
    from plotly.subplots import make_subplots

    output_dir = Path(output_dir)
    height = 400
    width = height * 3

    # Map method names to colours (handle both "hip" and "hip")
    def _color_for_method(method):
        key = method
        if method.lower().startswith("pred"):
            key = "hip"
        return HESSIAN_METHOD_TO_COLOUR.get(key)

    # Aggregations for speed and memory vs N
    avg_times = results_df.groupby(["n_atoms", "method"])["time"].mean().unstack()
    std_times = results_df.groupby(["n_atoms", "method"])["time"].std().unstack()
    avg_memory = results_df.groupby(["n_atoms", "method"])["memory"].mean().unstack()
    std_memory = results_df.groupby(["n_atoms", "method"])["memory"].std().unstack()

    # Aggregation for hip vs batch size (per-sample)
    no_hessian_results_df = bz_results_df[bz_results_df["method"] == "no_hessian"].copy()
    pred_results_df = bz_results_df[bz_results_df["method"] == "hip"].copy()
    # Compute per-sample time first, then aggregate mean/std
    pred_results_df["time_per_sample"] = pred_results_df["time"] / pred_results_df[
        "batch_size"
    ].replace(0, pd.NA)
    no_hessian_results_df["time_per_sample"] = no_hessian_results_df[
        "time"
    ] / no_hessian_results_df["batch_size"].replace(0, pd.NA)
    pred_group = pred_results_df.groupby(["batch_size"])["time_per_sample"]
    no_hessian_group = no_hessian_results_df.groupby(["batch_size"])["time_per_sample"]
    pred_avg_times = pred_group.mean()
    pred_std_times = pred_group.std()
    no_hessian_avg_times = no_hessian_group.mean()
    no_hessian_std_times = no_hessian_group.std()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Time (Single Sample)", "Memory", "Time (Batching)"),
        horizontal_spacing=0.05,
        vertical_spacing=0.0,
    )

    #########################################################
    # Col 1: Speed vs N
    for method in avg_times.columns:
        color = _color_for_method(method)
        if str(method).lower() == "hip":
            display_name = "HIP Hessians (ours)"
        elif str(method).lower() == "no_hessian":
            display_name = "No Hessian"
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
            "hip (ours)"
            if str(method).lower() == "hip"
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


    # Col 3: hip vs Batch Size (per-sample)
    # Ensure strictly positive values for log scale
    pred_avg_times_plot = pred_avg_times.copy()
    no_hessian_avg_times_plot = no_hessian_avg_times.copy()

    color = _color_for_method("no_hessian")
    _err_kwargs = {}
    if show_std and len(no_hessian_avg_times_plot.index) > 0:
        std_vals = no_hessian_std_times.reindex(no_hessian_avg_times_plot.index)
        if std_vals is not None:
            _err_kwargs = {"error_y": dict(type="data", array=std_vals.values)}
    fig.add_trace(
        go.Scatter(
            x=no_hessian_avg_times_plot.index,
            y=no_hessian_avg_times_plot.values,
            mode="lines+markers",
            name="No Hessian (Batching)",
            # legend="legend3",
            showlegend=False,
            line=dict(color=color) if color else None,
            marker=dict(color=color) if color else None,
            **_err_kwargs,
        ),
        row=1,
        col=3,
    )
    color = _color_for_method("hip")
    _err_kwargs = {}
    if show_std and len(pred_avg_times_plot.index) > 0:
        std_vals = pred_std_times.reindex(pred_avg_times_plot.index)
        if std_vals is not None:
            _err_kwargs = {"error_y": dict(type="data", array=std_vals.values)}
    fig.add_trace(
        go.Scatter(
            x=pred_avg_times_plot.index,
            y=pred_avg_times_plot.values,
            mode="lines+markers",
            name="hip (Batching)",
            # legend="legend3",
            showlegend=False,
            line=dict(color=color) if color else None,
            marker=dict(color=color) if color else None,
            **_err_kwargs,
        ),
        row=1,
        col=3,
    )

    # derive subplot domains to place legends at the top-middle of each subplot
    dom1 = fig.layout.xaxis.domain if hasattr(fig.layout, "xaxis") else [0.0, 0.3]
    dom2 = fig.layout.xaxis2.domain if hasattr(fig.layout, "xaxis2") else [0.35, 0.65]
    dom3 = fig.layout.xaxis3.domain if hasattr(fig.layout, "xaxis3") else [0.7, 1.0]
    x1 = 0.5 * (dom1[0] + dom1[1])
    x2 = 0.5 * (dom2[0] + dom2[1])
    x3 = 0.5 * (dom3[0] + dom3[1])

    # Add axis titles for each subplot
    fig.update_xaxes(title_text="Number of Atoms (N)", title_standoff=5, row=1, col=1)
    fig.update_yaxes(title_text="Average Time (ms)", title_standoff=10, row=1, col=1)
    # fig.update_yaxes(tickformat=".0e", exponentformat="e",  row=1, col=1)
    fig.update_xaxes(title_text="Number of Atoms (N)", title_standoff=5, row=1, col=2)
    fig.update_yaxes(title_text="Peak Memory (MB)", title_standoff=0, row=1, col=2)
    fig.update_xaxes(title_text="Batch Size", title_standoff=1, row=1, col=3)
    fig.update_yaxes(
        title_text="Average Time per Sample (ms)", title_standoff=0, row=1, col=3
    )

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
                gridwidth=1,
            ),
        ),
        yaxis3=dict(
            type="log",
            dtick=1,
            exponentformat="power",
            showexponent="all",
            showgrid=True,
            minor=dict(
                showgrid=True,
                gridcolor="#eee",  # "rgba(50,50,50,0.1)",
                gridwidth=1,
            ),
        ),
        legend=dict(
            # x=x2 - 0.07,
            x=x2 - 0.05,
            y=0.91,
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
        if ann.text in ("Time (Single Sample)", "Memory", "Time (Batching)"):
            ann.update(font=dict(size=TITLE_FONT_SIZE))

    # Add subplot panel labels (a, b, c) at top-left outside each subplot
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
    fig.add_annotation(
        x=dom3[0],
        y=0.999,
        xref="paper",
        yref="paper",
        text="<b>c</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )

    ############ Subplot 1 ############
    # Add arrow for subplot 1 manually in normalized domain coordinates (log scale friendly)
    # and add separate labels for tail (autograd) and head (hip), in ms
    last_n = avg_times.index.max()
    final_vals_speed = {}
    for m in ["no_hessian", "hip"]:
        if m in avg_times.columns and last_n in avg_times.index:
            val = avg_times[m].loc[last_n]
            if pd.notna(val):
                final_vals_speed[m] = float(val)
    speed_no_hess_text = (
        # f"<b>{round(final_vals_speed['no_hessian'])} ms</b>"
        f"{round(final_vals_speed['no_hessian'])} ms"
        if "no_hessian" in final_vals_speed
        else ""
    )
    speed_pred_text = (
        # f"<b>{round(final_vals_speed['hip'])} ms</b>"
        f"{round(final_vals_speed['hip'])} ms"
        if "hip" in final_vals_speed
        else ""
    )
    # Manual label positions (domain coordinates) for subplot 1
    speed_tail_label_x = 0.9
    speed_tail_label_y = 0.2
    speed_head_label_x = 0.9
    speed_head_label_y = 0.8
    # Labels at tail/head for subplot 1
    if speed_no_hess_text:
        fig.add_annotation(
            x=speed_tail_label_x,
            y=speed_tail_label_y,
            xref="x domain",
            yref="y domain",
            showarrow=False,
            text=speed_no_hess_text,
            font=dict(size=ANNOTATION_FONT_SIZE),
        )
    if speed_pred_text:
        fig.add_annotation(
            x=speed_head_label_x,
            y=speed_head_label_y,
            xref="x domain",
            yref="y domain",
            showarrow=False,
            text=speed_pred_text,
            font=dict(size=ANNOTATION_FONT_SIZE),
        )
    # Reduction label (no_hessian vs hip) in the middle, vertical
    if "no_hessian" in final_vals_speed and "hip" in final_vals_speed:
        _r = final_vals_speed["no_hessian"] / max(1e-12, final_vals_speed["hip"])
        speed_mid_text = f"<b>{round(_r, 3)}x</b>"
        speed_mid_label_x = 0.92
        speed_mid_label_y = 0.48
        fig.add_annotation(
            x=speed_mid_label_x,
            y=speed_mid_label_y,
            xref="x domain",
            yref="y domain",
            showarrow=False,
            text=speed_mid_text,
            # textangle=-90,
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
        )
    ############ Subplot 2 ############
    last_n = avg_memory.index.max()
    final_vals_memory = {}
    for m in ["no_hessian", "hip"]:
        if m in avg_memory.columns and last_n in avg_memory.index:
            val = avg_memory[m].loc[last_n]
            if pd.notna(val):
                final_vals_memory[m] = float(val)
    memory_no_hess_text = (
        # f"<b>{round(final_vals_memory['no_hessian'])} MB</b>"
        f"{round(final_vals_memory['no_hessian'])} MB"
        if "no_hessian" in final_vals_memory
        else ""
    )
    memory_pred_text = (
        # f"<b>{round(final_vals_memory['hip'])} MB</b>"
        f"{round(final_vals_memory['hip'])} MB"
        if "hip" in final_vals_memory
        else ""
    )
    # Manual label positions (domain coordinates) for subplot 2
    memory_tail_label_x = 0.83
    memory_tail_label_y = 0.89
    memory_head_label_x = 0.9
    memory_head_label_y = 0.6
    # Labels at tail/head for subplot 2
    fig.add_annotation(
        x=memory_tail_label_x,
        y=memory_tail_label_y,
        xref="x2 domain",
        yref="y2 domain",
        showarrow=False,
        text=memory_no_hess_text,
        font=dict(size=ANNOTATION_FONT_SIZE),
    )
    fig.add_annotation(
        x=memory_head_label_x,
        y=memory_head_label_y,
        xref="x2 domain",
        yref="y2 domain",
        showarrow=False,
        text=memory_pred_text,
        font=dict(size=ANNOTATION_FONT_SIZE),
    )
    memory_mid_text = f"<b>{int(round(final_vals_memory['no_hessian'] / final_vals_memory['hip']))}x</b>"
    memory_mid_label_x = 0.90
    memory_mid_label_y = 0.50
    fig.add_annotation(
        x=memory_mid_label_x,
        y=memory_mid_label_y,
        xref="x2 domain",
        yref="y2 domain",
        showarrow=False,
        text=memory_mid_text,
        # textangle=-90,
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )
    ############ Subplot 3 ############
    # add manual arrow for subplot 3 in normalized domain coords with labels
    # For hip use the final (largest bsz); for no_hessian use the first (smallest bsz)
    pred_text = ""
    no_hess_text = ""
    if len(pred_avg_times_plot.index) > 0:
        last_bz = pred_avg_times_plot.index.max()
        pred_val = float(pred_avg_times_plot.loc[last_bz])
        # pred_text = f"{last_bz}: {pred_val:.2f} ms"
        # pred_text = f"<b>{round(pred_val)} ms</b>"
        pred_text = f"{round(pred_val)} ms"
    if len(no_hessian_avg_times_plot.index) > 0:
        first_bz = no_hessian_avg_times_plot.index.max()
        no_hess_val = float(no_hessian_avg_times_plot.loc[first_bz])
        # no_hess_text = f"{first_bz}: {no_hess_val:.2f} ms"
        # no_hess_text = f"<b>{round(no_hess_val)} ms</b>"
        no_hess_text = f"{round(no_hess_val)} ms"
    # Labels at tail/head for subplot 3
    bz_tail_label_x = 0.85
    bz_tail_label_y = 0.1
    bz_head_label_x = 0.85
    bz_head_label_y = 0.2
    fig.add_annotation(
        x=bz_tail_label_x,
        y=bz_tail_label_y,
        xref="x3 domain",
        yref="y3 domain",
        showarrow=False,
        text=no_hess_text,
        font=dict(size=ANNOTATION_FONT_SIZE),
    )
    fig.add_annotation(
        x=bz_head_label_x,
        y=bz_head_label_y,
        xref="x3 domain",
        yref="y3 domain",
        showarrow=False,
        text=pred_text,
        font=dict(size=ANNOTATION_FONT_SIZE),
    )
    # Reduction label (no_hessian first vs hip last) in the middle, vertical
    if ("no_hess_val" in locals()) and ("pred_val" in locals()) and pred_val > 0:
        bz_mid_text = f"<b>{round(no_hess_val / pred_val, 3)}x</b>"
        bz_mid_label_x = 0.60
        bz_mid_label_y = 0.40
        fig.add_annotation(
            x=bz_mid_label_x,
            y=bz_mid_label_y,
            xref="x3 domain",
            yref="y3 domain",
            showarrow=False,
            text=bz_mid_text,
            textangle=0,
            font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
        )

    # Save only PNG to keep output concise
    output_path = output_dir / f"combined_speed_memory_batchsize_{_name}.png"
    # The height of the exported image in layout pixels. If the scale property is 1.0, this will also be the height of the exported image in physical pixels.
    # Scale > 1 increases the image resolution
    fig.write_image(output_path, width=width, height=height, scale=2)
    print(f"Plot saved to \n{output_path}")


"""
uv run scripts/speed_comparison.py --dataset ts1x-val.lmdb --max_samples_per_n 10 --ckpt_path ./ckpt/hesspred_v1.ckpt
uv run scripts/speed_comparison.py --dataset ts1x-val.lmdb --max_samples_per_n 10 --ckpt_path ./ckpt/hip_v1.ckpt

python scripts/speed_comparison.py --dataset RGD1.lmdb --max_samples_per_n 10 --ckpt_path ../ReactBench/ckpt/hesspred/eqv2hp1.ckpt
python scripts/speed_comparison.py --dataset ts1x-val.lmdb --max_samples_per_n 100
python scripts/speed_comparison.py --dataset ts1x_hess_train_big.lmdb --max_samples_per_n 1000
"""
if __name__ == "__main__":
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
        default="ts1x-val.lmdb",
        help="Dataset file name",
    )
    parser.add_argument(
        "--max_samples_per_n",
        type=int,
        default=100,
        help="Maximum number of samples per N atoms to test.",
    )
    parser.add_argument(
        "--redo",
        type=bool,
        default=False,
        help="Redo the speed comparison. If false attempt to load existing results.",
    )
    parser.add_argument(
        "--redobz",
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

    ckpt_name = args.ckpt_path.split("/")[-1].split(".")[0]

    output_dir = "./results_speed_hip"
    output_dir = Path(output_dir)
    output_path = (
        output_dir
        / f"{args.dataset}_speed_comparison_results_{ckpt_name}_{args.max_samples_per_n}.csv"
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
    # plot_speed_comparison(results_df)

    ##############################################################
    # Second benchmark: hip-only vs batch size (random N)
    print()

    # dataset_name = args.dataset
    dataset_name = "ts1x-val.lmdb"
    output_path_speedbz = (
        output_dir
        / f"{dataset_name}_hip_batchsize_results_{ckpt_name}_{args.max_samples_per_n}.csv"
    )
    if output_path_speedbz.exists() and not args.redobz:
        bz_results_df = pd.read_csv(output_path_speedbz)
        print(
            f"Loaded existing hip batch-size results from {output_path_speedbz}"
        )
    else:
        bz_results_df = hip_batchsize_benchmark(
            checkpoint_path=args.ckpt_path,
            dataset_name=dataset_name,
            output_path=output_path_speedbz,
        )

    # plot_hip_batchsize(bz_results_df, output_dir=output_dir)
    # plot_hip_batchsize(bz_results_df, output_dir=output_dir, logy=True)

    # Combined side-by-side plot
    plot_combined_speed_memory_batchsize(
        results_df, bz_results_df, output_dir=output_dir, show_std=args.show_std, _name=ckpt_name
    )

    print("\nDone!")
