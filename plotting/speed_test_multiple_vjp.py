import torch
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Batch as TGBatch
from torch_geometric.data import Data as TGData
from pathlib import Path
import plotly.graph_objects as go

from hip.training_module import PotentialModule
from hip.colours import (
    HESSIAN_METHOD_TO_COLOUR,
    ANNOTATION_BOLD_FONT_SIZE,
    AXES_FONT_SIZE,
    AXES_TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
)

# https://plotly.com/python/templates/
PLOTLY_TEMPLATE = "plotly_white"


def compute_vjp(coords, forces, v=None, retain_graph=False, create_graph=False):
    """Compute a single VJP: v^T @ J where J is d(forces)/d(coords).

    If v is None, a random vector is generated.
    """
    if v is None:
        v = torch.randn_like(forces)

    # VJP: v^T @ J = grad(v^T @ forces, coords)
    vjp = torch.autograd.grad(
        [(v * forces).sum()],
        [coords],
        retain_graph=retain_graph,
        create_graph=create_graph,
    )[0]
    return vjp


def create_random_batch(n_atoms, device="cuda", seed=None):
    """Create a random batch with n_atoms atoms."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Random positions in a 1x1x1 box
    positions = torch.rand(n_atoms, 3, dtype=torch.float32) * 0.5

    # Random atomic numbers (common elements: H=1, C=6, N=7, O=8)
    atomic_numbers = torch.randint(1, 9, (n_atoms,), dtype=torch.int64)

    # Create torch_geometric Data object
    data = TGData(
        pos=positions.to(device),
        z=atomic_numbers.to(device),
        charges=atomic_numbers.to(device),
        natoms=torch.tensor([n_atoms], dtype=torch.int64, device=device),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool, device=device),
    )
    return TGBatch.from_data_list([data])


def time_forward_pass(model, batch):
    """Times a single forward pass and measures memory usage."""
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    with torch.no_grad():
        ener, force, out = model.forward(batch, otf_graph=True, hessian=False)

    end_event.record()
    torch.cuda.synchronize()

    time_taken = start_event.elapsed_time(end_event)
    memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    return time_taken, memory_usage


def time_forward_plus_n_vjps(
    model, batch, n_vjps, retain_graph=True, create_graph=False
):
    """Times forward pass + n_vjps VJPs and measures memory usage."""
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    batch.pos.requires_grad_()

    start_event.record()

    ener, force, out = model.forward(batch, otf_graph=True, hessian=False)

    # Compute n_vjps VJPs
    for i in range(n_vjps):
        # retain_graph = i < n_vjps - 1  # Retain graph for all but the last VJP
        compute_vjp(
            batch.pos, force, retain_graph=retain_graph, create_graph=create_graph
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


def speed_test_multiple_vjp(
    checkpoint_path,
    n_trials=10,
    device="cuda",
    cutoff=10.0,
    cutoff_hessian=10.0,
    n_atoms=5,
    max_vjps=10,
    retain_graph=True,
    create_graph=False,
):
    """Test speed of forward pass vs forward + N VJPs with fixed N atoms."""
    # Load model
    model = get_model_ckpt(checkpoint_path, device)
    model.cutoff = cutoff
    model.cutoff_hessian = cutoff_hessian

    # Warm up - do a couple of forward passes to warm up the model
    print("Warming up model...")
    for i in range(12):
        warmup_batch = create_random_batch(n_atoms, device, seed=i)
        time_forward_pass(model, warmup_batch)
        torch.cuda.empty_cache()
        warmup_batch = create_random_batch(n_atoms, device, seed=i)
        time_forward_plus_n_vjps(
            model, warmup_batch, max_vjps, retain_graph, create_graph
        )
        torch.cuda.empty_cache()
    print("Model warmed up\n")

    results = []

    # Test from 0 to max_vjps VJPs
    for n_vjps in tqdm(range(0, max_vjps + 1), desc="Testing N VJPs"):
        for trial in range(n_trials):
            # Create random batch
            batch = create_random_batch(n_atoms, device, seed=trial)

            # Time forward + n_vjps VJPs
            batch = create_random_batch(n_atoms, device, seed=trial)
            time_vjp, mem_vjp = time_forward_plus_n_vjps(
                model, batch, n_vjps, retain_graph, create_graph
            )
            results.append(
                {
                    "n_vjps": n_vjps,
                    "method": "forward_plus_vjps",
                    "time": time_vjp,
                    "memory": mem_vjp,
                    "trial": trial,
                }
            )
            torch.cuda.empty_cache()

            # Time forward pass only
            time_fwd, mem_fwd = time_forward_pass(model, batch)
            results.append(
                {
                    "n_vjps": n_vjps,
                    "method": "forward_pass",
                    "time": time_fwd,
                    "memory": mem_fwd,
                    "trial": trial,
                }
            )
            torch.cuda.empty_cache()

    return results


def plot_speed_comparison(
    results_df,
    output_dir="./results_speed",
    show_std=False,
    logy_time=False,
    ymin_time=None,
):
    """Plot speed comparison: forward pass vs forward + N VJPs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    height = 400
    width = 400

    # Map method names to colours
    def _color_for_method(method):
        method_lower = str(method).lower()
        if method_lower == "forward_plus_vjps":
            color = HESSIAN_METHOD_TO_COLOUR.get("autograd")
        elif method_lower == "forward_pass":
            color = "#68c4af"  # Green-ish color
        else:
            color = HESSIAN_METHOD_TO_COLOUR.get(method_lower)
            if color is None:
                color = "#cfcfcf"
        return color

    # Aggregations for speed vs N VJPs
    avg_times = results_df.groupby(["n_vjps", "method"])["time"].mean().unstack()
    std_times = results_df.groupby(["n_vjps", "method"])["time"].std().unstack()

    fig = go.Figure()

    # Add traces for each method
    for method in avg_times.columns:
        color = _color_for_method(method)
        if str(method).lower() == "forward_plus_vjps":
            display_name = "Forward + N VJPs"
        elif str(method).lower() == "forward_pass":
            display_name = "Forward Pass"
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
                showlegend=True,
                line=dict(color=color, width=3),
                marker=dict(color=color),
                **_err_kwargs,
            )
        )

    # Add axis titles
    fig.update_xaxes(
        title_text="Number of VJPs (N)",
        title_standoff=5,
        tickfont=dict(size=AXES_FONT_SIZE),
        title_font=dict(size=AXES_TITLE_FONT_SIZE),
    )
    fig.update_yaxes(
        title_text="Average Time (ms)",
        title_standoff=10,
        tickfont=dict(size=AXES_FONT_SIZE),
        title_font=dict(size=AXES_TITLE_FONT_SIZE),
    )

    # Build yaxis config
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

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=0, b=0, t=20),
        width=width,
        height=height,
        yaxis=yaxis_config,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(255,255,255,0.6)",
            font=dict(size=LEGEND_FONT_SIZE),
        ),
    )

    # Set y-axis minimum if specified
    if ymin_time is not None:
        if len(avg_times) > 0 and not avg_times.empty:
            ymax_time = float(avg_times.max().max()) * 1.1
        else:
            ymax_time = ymin_time * 10
        fig.update_yaxes(range=[ymin_time, ymax_time], autorange=False)
    else:
        # Default to starting at 0
        if len(avg_times) > 0 and not avg_times.empty:
            ymax_time = float(avg_times.max().max()) * 1.1
            fig.update_yaxes(range=[0, ymax_time], autorange=False)

    # Add panel label (a) at top-left
    fig.add_annotation(
        x=0.01,
        y=0.999,
        xref="paper",
        yref="paper",
        text="<b>a</b>",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=ANNOTATION_BOLD_FONT_SIZE),
    )

    output_path = output_dir / "speed_comparison_multiple_vjp.png"
    fig.write_image(output_path, width=width, height=height, scale=2)
    print(f"Plot saved to \n {output_path}")


def str_to_bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


"""
uv run scriptsp/speed_test_multiple_vjp.py --n_trials 10 --max_vjps 10
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speed test: Forward pass vs Forward + N VJPs (fixed N atoms)"
    )

    parser.add_argument(
        "--ckpt_path",
        "-c",
        type=str,
        default="ckpt/hip_v2.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of samples per N VJPs (default: 10)",
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
        "--n_atoms",
        type=int,
        default=5,
        help="Number of atoms (fixed, default: 5)",
    )
    parser.add_argument(
        "--max_vjps",
        type=int,
        default=10,
        help="Maximum number of VJPs to test (default: 10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_speed",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--show_std",
        type=str_to_bool,
        default=False,
        help="Show standard deviation as error bars in the plot.",
    )
    parser.add_argument(
        "--retain_graph",
        type=str_to_bool,
        default=True,
        help="Retain graph for all VJPs.",
    )
    parser.add_argument(
        "--create_graph",
        type=str_to_bool,
        default=False,
        help="Create graph for all VJPs.",
    )

    # torch.jit.optimized_execution(False)
    # torch._C._get_graph_executor_optimize(False) # Disables JIT graph executor optimizations
    # torch.backends.cudnn.benchmark = False # Disables cuDNN benchmark mode (uses deterministic algorithms)
    # torch.backends.cudnn.deterministic = True # Forces deterministic cuDNN algorithms

    args = parser.parse_args()
    torch.manual_seed(42)

    results = speed_test_multiple_vjp(
        checkpoint_path=args.ckpt_path,
        n_trials=args.n_trials,
        cutoff=args.cutoff,
        cutoff_hessian=args.cutoff_hessian,
        n_atoms=args.n_atoms,
        max_vjps=args.max_vjps,
        retain_graph=args.retain_graph,
        create_graph=args.create_graph,
    )

    # Print summary
    results_df = pd.DataFrame(results)

    # Group by n_vjps and method
    avg_times = results_df.groupby(["n_vjps", "method"])["time"].mean().unstack()
    std_times = results_df.groupby(["n_vjps", "method"])["time"].std().unstack()
    avg_memory = results_df.groupby(["n_vjps", "method"])["memory"].mean().unstack()
    std_memory = results_df.groupby(["n_vjps", "method"])["memory"].std().unstack()

    # Print comparison
    print(
        f"N atoms = {args.n_atoms}, retain_graph = {args.retain_graph}, create_graph = {args.create_graph}"
    )
    print("\n" + "=" * 90)
    print("Speed Comparison: Forward Pass vs Forward + N VJPs")
    print("=" * 90)
    print(
        f"\n{'N VJPs':<12} {'Forward Pass (ms)':<20} {'Forward+N VJPs (ms)':<25} {'Ratio':<12}"
    )
    print("-" * 90)
    for n_vjps in sorted(avg_times.index):
        fwd_time = avg_times.loc[n_vjps, "forward_pass"]
        vjp_time = avg_times.loc[n_vjps, "forward_plus_vjps"]
        ratio = vjp_time / fwd_time
        print(f"{n_vjps:<12} {fwd_time:<20.2f} {vjp_time:<25.2f} {ratio:<12.2f}")

    print("\n" + "=" * 70)
    print("Memory Comparison: Forward Pass vs Forward + N VJPs")
    print("=" * 70)
    print(
        f"\n{'N VJPs':<12} {'Forward Pass (MB)':<20} {'Forward+N VJPs (MB)':<25} {'Ratio':<12}"
    )
    print("-" * 70)
    for n_vjps in sorted(avg_memory.index):
        fwd_mem = avg_memory.loc[n_vjps, "forward_pass"]
        vjp_mem = avg_memory.loc[n_vjps, "forward_plus_vjps"]
        ratio = vjp_mem / fwd_mem
        print(f"{n_vjps:<12} {fwd_mem:<20.2f} {vjp_mem:<25.2f} {ratio:<12.2f}")

    # Compute linear fits
    print("\n" + "=" * 70)
    print("Linear Fit Analysis")
    print("=" * 70)

    # Linear fits
    n_vjps_array = np.array(sorted(avg_times.index))
    fwd_times = np.array([avg_times.loc[n, "forward_pass"] for n in n_vjps_array])
    vjp_times = np.array([avg_times.loc[n, "forward_plus_vjps"] for n in n_vjps_array])
    ratios = vjp_times / fwd_times

    # Linear fit on ratio vs N VJPs: ratio = slope * n_vjps + intercept
    ratio_fit = np.polyfit(n_vjps_array, ratios, 1)
    ratio_slope, ratio_intercept = ratio_fit

    # Linear fit on Forward+N VJPs time vs N VJPs: time = slope * n_vjps + intercept
    vjp_time_fit = np.polyfit(n_vjps_array, vjp_times, 1)
    vjp_time_slope, vjp_time_intercept = vjp_time_fit

    # Compute R² for ratio fit
    ratio_pred = ratio_slope * n_vjps_array + ratio_intercept
    ratio_ss_res = np.sum((ratios - ratio_pred) ** 2)
    ratio_ss_tot = np.sum((ratios - np.mean(ratios)) ** 2)
    ratio_r2 = 1 - (ratio_ss_res / ratio_ss_tot) if ratio_ss_tot > 0 else 0

    # Compute R² for VJP time fit
    vjp_time_pred = vjp_time_slope * n_vjps_array + vjp_time_intercept
    vjp_time_ss_res = np.sum((vjp_times - vjp_time_pred) ** 2)
    vjp_time_ss_tot = np.sum((vjp_times - np.mean(vjp_times)) ** 2)
    vjp_time_r2 = 1 - (vjp_time_ss_res / vjp_time_ss_tot) if vjp_time_ss_tot > 0 else 0

    print(f"\nLinear fit on Ratio (Forward+N VJPs / Forward Pass) vs N VJPs:")
    print(f"  Ratio = {ratio_slope:.4f} * N_VJPs + {ratio_intercept:.4f}")
    print(f"  R² = {ratio_r2:.4f}")

    print(f"\nLinear fit on Forward+N VJPs Time (ms) vs N VJPs:")
    print(f"  Time = {vjp_time_slope:.4f} * N_VJPs + {vjp_time_intercept:.4f}")
    print(f"  R² = {vjp_time_r2:.4f}")

    # Plot results
    plot_speed_comparison(
        results_df,
        output_dir=args.output_dir,
        show_std=args.show_std,
        logy_time=False,
        ymin_time=0,
    )

    print("\nDone!")
