import torch
from torch.utils.benchmark import Timer
import argparse
from tqdm import tqdm
import numpy as np
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


def compute_vjp(coords, forces, v=None):
    """Compute a single VJP: v^T @ J where J is d(forces)/d(coords).

    If v is None, a random vector is generated.
    """
    if v is None:
        v = torch.randn_like(forces)

    # VJP: v^T @ J = grad(v^T @ forces, coords)
    vjp = torch.autograd.grad(
        [(v * forces).sum()],
        [coords],
        retain_graph=False,
        create_graph=False,
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


def time_vjp(model, batch):
    """Times VJP computation and measures memory usage."""
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    batch.pos.requires_grad_()
    ener, force, out = model.forward(batch, otf_graph=True, hessian=False)
    compute_vjp(batch.pos, force)

    end_event.record()
    torch.cuda.synchronize()

    time_taken = start_event.elapsed_time(end_event)
    memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    return time_taken, memory_usage


def time_forward_pass_benchmark(model, batch):
    """Times forward pass using torch.utils.benchmark.Timer."""
    timer = Timer(
        stmt="""
with torch.no_grad():
    ener, force, out = model.forward(batch, otf_graph=True, hessian=False)
""",
        globals={"model": model, "batch": batch, "torch": torch},
        num_threads=1,
    )
    measurement = timer.blocked_autorange(min_run_time=0.1)
    time_taken_ms = measurement.mean * 1000  # Convert to ms
    return time_taken_ms


def time_vjp_benchmark(model, batch):
    """Times VJP computation using torch.utils.benchmark.Timer."""
    timer = Timer(
        stmt="""
batch.pos.requires_grad_()
ener, force, out = model.forward(batch, otf_graph=True, hessian=False)
vjp_result = compute_vjp(batch.pos, force)
""",
        globals={
            "model": model,
            "batch": batch,
            "torch": torch,
            "compute_vjp": compute_vjp,
        },
        num_threads=1,
    )
    measurement = timer.blocked_autorange(min_run_time=0.1)
    time_taken_ms = measurement.mean * 1000  # Convert to ms
    return time_taken_ms


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


def speed_test_random(
    checkpoint_path,
    n_trials=10,
    device="cuda",
    cutoff=10.0,
    cutoff_hessian=10.0,
):
    """Test speed of forward pass vs VJP computation on random samples."""
    # Load model
    model = get_model_ckpt(checkpoint_path, device)
    model.cutoff = cutoff
    model.cutoff_hessian = cutoff_hessian

    # Warm up - do a couple of forward passes to warm up the model
    # populate caches, jit, load cuda kernels, and what not
    print("Warming up model...")
    for i in range(12):
        n_warmup = (i % 9) + 2  # Cycle through N=2 to N=10
        warmup_batch = create_random_batch(n_warmup, device, seed=i)
        time_forward_pass(model, warmup_batch)
        torch.cuda.empty_cache()
        time_vjp(model, warmup_batch)
        torch.cuda.empty_cache()
    print("Model warmed up\n")

    results = []

    # Test from N=2 to N=10
    for n_atoms in tqdm(range(2, 11), desc="Testing N atoms"):
        for trial in range(n_trials):
            # Create random batch
            batch = create_random_batch(n_atoms, device, seed=trial)

            # Time forward pass (CUDA events)
            time_fwd, mem_fwd = time_forward_pass(model, batch)
            results.append(
                {
                    "n_atoms": n_atoms,
                    "method": "forward_pass",
                    "timing_method": "cuda_events",
                    "time": time_fwd,
                    "memory": mem_fwd,
                    "trial": trial,
                }
            )
            torch.cuda.empty_cache()

            # Time forward pass (benchmark Timer)
            batch_bench = create_random_batch(n_atoms, device, seed=trial)
            time_fwd_bench = time_forward_pass_benchmark(model, batch_bench)
            results.append(
                {
                    "n_atoms": n_atoms,
                    "method": "forward_pass",
                    "timing_method": "benchmark",
                    "time": time_fwd_bench,
                    "memory": mem_fwd,  # Use same memory as cuda_events
                    "trial": trial,
                }
            )
            torch.cuda.empty_cache()

            # Time VJP (CUDA events)
            batch = create_random_batch(n_atoms, device, seed=trial)
            time_vjp_result, mem_vjp = time_vjp(model, batch)
            results.append(
                {
                    "n_atoms": n_atoms,
                    "method": "vjp",
                    "timing_method": "cuda_events",
                    "time": time_vjp_result,
                    "memory": mem_vjp,
                    "trial": trial,
                }
            )
            torch.cuda.empty_cache()

            # Time VJP (benchmark Timer)
            batch_bench = create_random_batch(n_atoms, device, seed=trial)
            time_vjp_bench = time_vjp_benchmark(model, batch_bench)
            results.append(
                {
                    "n_atoms": n_atoms,
                    "method": "vjp",
                    "timing_method": "benchmark",
                    "time": time_vjp_bench,
                    "memory": mem_vjp,  # Use same memory as cuda_events
                    "trial": trial,
                }
            )
            torch.cuda.empty_cache()

    return results


def plot_speed_comparison(
    results_df,
    output_dir="./results_speed",
    show_std=False,
    logy_time=True,
    ymin_time=None,
    timing_method="cuda_events",
):
    """Plot speed comparison in the same style as the first panel of combined plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    height = 400
    width = 400

    # Map method names to colours
    def _color_for_method(method):
        method_lower = str(method).lower()
        if method_lower == "vjp":
            color = HESSIAN_METHOD_TO_COLOUR.get("autograd")
        elif method_lower == "forward_pass":
            color = "#68c4af"  # Green-ish color
        else:
            color = HESSIAN_METHOD_TO_COLOUR.get(method_lower)
            if color is None:
                color = "#cfcfcf"
        return color

    # Filter by timing method
    filtered_df = results_df[results_df["timing_method"] == timing_method]

    # Aggregations for speed vs N
    avg_times = filtered_df.groupby(["n_atoms", "method"])["time"].mean().unstack()
    std_times = filtered_df.groupby(["n_atoms", "method"])["time"].std().unstack()

    fig = go.Figure()

    # Add traces for each method
    for method in avg_times.columns:
        color = _color_for_method(method)
        if str(method).lower() == "vjp":
            display_name = "VJP"
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
        title_text="Number of Atoms (N)",
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

    timing_suffix = "cuda" if timing_method == "cuda_events" else "benchmark"
    output_path = output_dir / f"speed_comparison_vjp_{timing_suffix}.png"
    fig.write_image(output_path, width=width, height=height, scale=2)
    print(f"Plot saved to \n {output_path}")


"""
uv run scriptsp/speed_test_vjp.py --n_trials 10
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speed test: Forward pass vs VJP on random samples"
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
        help="Number of samples per N atoms (default: 10)",
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
        "--output_dir",
        type=str,
        default="./results_speed",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--show_std",
        type=bool,
        default=False,
        help="Show standard deviation as error bars in the plot.",
    )

    args = parser.parse_args()
    torch.manual_seed(42)

    results = speed_test_random(
        checkpoint_path=args.ckpt_path,
        n_trials=args.n_trials,
        cutoff=args.cutoff,
        cutoff_hessian=args.cutoff_hessian,
    )

    # Print summary
    import pandas as pd

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("Speed Comparison: Forward Pass vs VJP")
    print("=" * 70)

    # Group by n_atoms, method, and timing_method
    avg_times = (
        results_df.groupby(["n_atoms", "method", "timing_method"])["time"]
        .mean()
        .unstack(level=[1, 2])
    )
    std_times = (
        results_df.groupby(["n_atoms", "method", "timing_method"])["time"]
        .std()
        .unstack(level=[1, 2])
    )
    avg_memory = results_df.groupby(["n_atoms", "method"])["memory"].mean().unstack()
    std_memory = results_df.groupby(["n_atoms", "method"])["memory"].std().unstack()

    # Print comparison for CUDA events
    print("\n" + "=" * 90)
    print("Speed Comparison: Forward Pass vs VJP (CUDA Events)")
    print("=" * 90)
    print(
        f"\n{'N (atoms)':<12} {'Forward Pass (ms)':<20} {'VJP (ms)':<20} {'Speedup':<12}"
    )
    print("-" * 90)
    for n_atoms in sorted(avg_times.index):
        fwd_time = avg_times.loc[n_atoms, ("forward_pass", "cuda_events")]
        vjp_time = avg_times.loc[n_atoms, ("vjp", "cuda_events")]
        speedup = vjp_time / fwd_time
        print(f"{n_atoms:<12} {fwd_time:<20.2f} {vjp_time:<20.2f} {speedup:<12.2f}x")

    print("-" * 90)
    fwd_mean = avg_times[("forward_pass", "cuda_events")].mean()
    vjp_mean = avg_times[("vjp", "cuda_events")].mean()
    print(
        f"{'Average':<12} {fwd_mean:<20.2f} {vjp_mean:<20.2f} {(vjp_mean / fwd_mean):<12.2f}x"
    )

    # Print comparison for Benchmark Timer
    print("\n" + "=" * 90)
    print("Speed Comparison: Forward Pass vs VJP (Benchmark Timer)")
    print("=" * 90)
    print(
        f"\n{'N (atoms)':<12} {'Forward Pass (ms)':<20} {'VJP (ms)':<20} {'Speedup':<12}"
    )
    print("-" * 90)
    for n_atoms in sorted(avg_times.index):
        fwd_time = avg_times.loc[n_atoms, ("forward_pass", "benchmark")]
        vjp_time = avg_times.loc[n_atoms, ("vjp", "benchmark")]
        speedup = vjp_time / fwd_time
        print(f"{n_atoms:<12} {fwd_time:<20.2f} {vjp_time:<20.2f} {speedup:<12.2f}x")

    print("-" * 90)
    fwd_mean = avg_times[("forward_pass", "benchmark")].mean()
    vjp_mean = avg_times[("vjp", "benchmark")].mean()
    print(
        f"{'Average':<12} {fwd_mean:<20.2f} {vjp_mean:<20.2f} {(vjp_mean / fwd_mean):<12.2f}x"
    )

    # Print timing method comparison
    print("\n" + "=" * 90)
    print("Timing Method Comparison: CUDA Events vs Benchmark Timer")
    print("=" * 90)
    print(
        f"\n{'N (atoms)':<12} {'Fwd: CUDA (ms)':<18} {'Fwd: Bench (ms)':<18} {'Ratio':<12} {'VJP: CUDA (ms)':<18} {'VJP: Bench (ms)':<18} {'Ratio':<12}"
    )
    print("-" * 90)
    for n_atoms in sorted(avg_times.index):
        fwd_cuda = avg_times.loc[n_atoms, ("forward_pass", "cuda_events")]
        fwd_bench = avg_times.loc[n_atoms, ("forward_pass", "benchmark")]
        fwd_ratio = fwd_bench / fwd_cuda
        vjp_cuda = avg_times.loc[n_atoms, ("vjp", "cuda_events")]
        vjp_bench = avg_times.loc[n_atoms, ("vjp", "benchmark")]
        vjp_ratio = vjp_bench / vjp_cuda
        print(
            f"{n_atoms:<12} {fwd_cuda:<18.2f} {fwd_bench:<18.2f} {fwd_ratio:<12.3f} {vjp_cuda:<18.2f} {vjp_bench:<18.2f} {vjp_ratio:<12.3f}"
        )

    print("-" * 90)
    fwd_cuda_mean = avg_times[("forward_pass", "cuda_events")].mean()
    fwd_bench_mean = avg_times[("forward_pass", "benchmark")].mean()
    vjp_cuda_mean = avg_times[("vjp", "cuda_events")].mean()
    vjp_bench_mean = avg_times[("vjp", "benchmark")].mean()
    print(
        f"{'Average':<12} {fwd_cuda_mean:<18.2f} {fwd_bench_mean:<18.2f} {(fwd_bench_mean / fwd_cuda_mean):<12.3f} {vjp_cuda_mean:<18.2f} {vjp_bench_mean:<18.2f} {(vjp_bench_mean / vjp_cuda_mean):<12.3f}"
    )

    print("\n" + "=" * 70)
    print("Memory Comparison: Forward Pass vs VJP")
    print("=" * 70)
    print(
        f"\n{'N (atoms)':<12} {'Forward Pass (MB)':<20} {'VJP (MB)':<20} {'Ratio':<12}"
    )
    print("-" * 70)
    for n_atoms in sorted(avg_memory.index):
        fwd_mem = avg_memory.loc[n_atoms, "forward_pass"]
        vjp_mem = avg_memory.loc[n_atoms, "vjp"]
        ratio = vjp_mem / fwd_mem
        print(f"{n_atoms:<12} {fwd_mem:<20.2f} {vjp_mem:<20.2f} {ratio:<12.2f}x")

    print("-" * 70)
    print(
        f"{'Average':<12} {avg_memory['forward_pass'].mean():<20.2f} {avg_memory['vjp'].mean():<20.2f} {(avg_memory['vjp'].mean() / avg_memory['forward_pass'].mean()):<12.2f}x"
    )

    # Plot results for both timing methods
    plot_speed_comparison(
        results_df,
        output_dir=args.output_dir,
        show_std=args.show_std,
        logy_time=True,
        timing_method="cuda_events",
    )
    plot_speed_comparison(
        results_df,
        output_dir=args.output_dir,
        show_std=args.show_std,
        logy_time=True,
        timing_method="benchmark",
    )

    print("\nDone!")
