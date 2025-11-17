import torch
from torch.utils.benchmark import Timer
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from hip.colours import (
    HESSIAN_METHOD_TO_COLOUR,
    ANNOTATION_BOLD_FONT_SIZE,
    AXES_FONT_SIZE,
    AXES_TITLE_FONT_SIZE,
    LEGEND_FONT_SIZE,
)

# https://plotly.com/python/templates/
PLOTLY_TEMPLATE = "plotly_white"


def create_random_matrices(n, device="cuda", seed=None):
    """Create two random N×N matrices."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    A = torch.rand(n, n, dtype=torch.float32, device=device)
    B = torch.rand(n, n, dtype=torch.float32, device=device)
    return A, B


def time_matmul_simple(A, B):
    """Times simple matrix multiplication and measures memory usage."""
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    with torch.no_grad():
        C = A @ B

    end_event.record()
    torch.cuda.synchronize()

    time_taken = start_event.elapsed_time(end_event)
    memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    return time_taken, memory_usage


def time_matmul_grad(A, B):
    """Times matrix multiplication with gradient computation using autograd and measures memory usage."""
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    A.requires_grad_()
    B.requires_grad_()
    C = A @ B
    loss = C.sum()
    # Compute gradients using autograd (like in the model script)
    grad_A, grad_B = torch.autograd.grad([loss], [A, B], create_graph=False)

    end_event.record()
    torch.cuda.synchronize()

    time_taken = start_event.elapsed_time(end_event)
    memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    return time_taken, memory_usage


def time_matmul_simple_benchmark(A, B):
    """Times simple matrix multiplication using torch.utils.benchmark.Timer."""
    timer = Timer(
        stmt="""
with torch.no_grad():
    C = A @ B
""",
        globals={"A": A, "B": B, "torch": torch},
        num_threads=1,
    )
    measurement = timer.blocked_autorange(min_run_time=0.1)
    time_taken_ms = measurement.mean * 1000  # Convert to ms
    return time_taken_ms


def time_matmul_grad_benchmark(A, B):
    """Times matrix multiplication with gradients using torch.utils.benchmark.Timer."""
    timer = Timer(
        stmt="""
A.requires_grad_()
B.requires_grad_()
C = A @ B
loss = C.sum()
grad_A, grad_B = torch.autograd.grad([loss], [A, B], create_graph=False)
""",
        globals={"A": A, "B": B, "torch": torch},
        num_threads=1,
    )
    measurement = timer.blocked_autorange(min_run_time=0.1)
    time_taken_ms = measurement.mean * 1000  # Convert to ms
    return time_taken_ms


def speed_test_matmul(
    n_trials=10,
    device="cuda",
):
    """Test speed of simple matrix multiplication vs matrix multiplication with gradients."""
    # Warm up - do a couple of operations to warm up
    # populate caches, jit, load cuda kernels, and what not
    print("Warming up...")
    for i in range(12):
        n_warmup = (i % 9) + 2  # Cycle through N=2 to N=10
        A, B = create_random_matrices(n_warmup, device, seed=i)
        time_matmul_simple(A, B)
        torch.cuda.empty_cache()
        A, B = create_random_matrices(n_warmup, device, seed=i)
        time_matmul_grad(A, B)
        torch.cuda.empty_cache()
    print("Warmed up\n")

    results = []

    # Test from N=2 to N=10
    for n in tqdm(range(2, 11), desc="Testing N (matrix size)"):
        for trial in range(n_trials):
            # Create random matrices
            A, B = create_random_matrices(n, device, seed=trial)

            # Time simple matmul (CUDA events)
            time_simple, mem_simple = time_matmul_simple(A, B)
            results.append(
                {
                    "n": n,
                    "method": "matmul_simple",
                    "timing_method": "cuda_events",
                    "time": time_simple,
                    "memory": mem_simple,
                    "trial": trial,
                }
            )
            torch.cuda.empty_cache()

            # Time simple matmul (benchmark Timer)
            A_bench, B_bench = create_random_matrices(n, device, seed=trial)
            time_simple_bench = time_matmul_simple_benchmark(A_bench, B_bench)
            results.append(
                {
                    "n": n,
                    "method": "matmul_simple",
                    "timing_method": "benchmark",
                    "time": time_simple_bench,
                    "memory": mem_simple,  # Use same memory as cuda_events
                    "trial": trial,
                }
            )
            torch.cuda.empty_cache()

            # Time matmul with gradients (CUDA events)
            A, B = create_random_matrices(n, device, seed=trial)
            time_grad, mem_grad = time_matmul_grad(A, B)
            results.append(
                {
                    "n": n,
                    "method": "matmul_grad",
                    "timing_method": "cuda_events",
                    "time": time_grad,
                    "memory": mem_grad,
                    "trial": trial,
                }
            )
            torch.cuda.empty_cache()

            # Time matmul with gradients (benchmark Timer)
            A_bench, B_bench = create_random_matrices(n, device, seed=trial)
            time_grad_bench = time_matmul_grad_benchmark(A_bench, B_bench)
            results.append(
                {
                    "n": n,
                    "method": "matmul_grad",
                    "timing_method": "benchmark",
                    "time": time_grad_bench,
                    "memory": mem_grad,  # Use same memory as cuda_events
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
        if method_lower == "matmul_grad":
            color = HESSIAN_METHOD_TO_COLOUR.get("autograd")
        elif method_lower == "matmul_simple":
            color = "#68c4af"  # Green-ish color
        else:
            color = HESSIAN_METHOD_TO_COLOUR.get(method_lower)
            if color is None:
                color = "#cfcfcf"
        return color

    # Filter by timing method
    filtered_df = results_df[results_df["timing_method"] == timing_method]

    # Aggregations for speed vs N
    avg_times = filtered_df.groupby(["n", "method"])["time"].mean().unstack()
    std_times = filtered_df.groupby(["n", "method"])["time"].std().unstack()

    fig = go.Figure()

    # Add traces for each method
    for method in avg_times.columns:
        color = _color_for_method(method)
        if str(method).lower() == "matmul_grad":
            display_name = "MatMul (Grad)"
        elif str(method).lower() == "matmul_simple":
            display_name = "MatMul (Simple)"
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
        title_text="Matrix Size (N×N)",
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
    output_path = output_dir / f"speed_comparison_matmul_{timing_suffix}.png"
    fig.write_image(output_path, width=width, height=height, scale=2)
    print(f"Plot saved to \n {output_path}")


"""
uv run scriptsp/speed_test_matmul.py --n_trials 10
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speed test: Simple MatMul vs MatMul with Gradients"
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of samples per N (default: 10)",
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

    results = speed_test_matmul(
        n_trials=args.n_trials,
    )

    # Print summary
    import pandas as pd

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("Speed Comparison: Simple MatMul vs MatMul with Gradients")
    print("=" * 70)

    # Group by n, method, and timing_method
    avg_times = (
        results_df.groupby(["n", "method", "timing_method"])["time"]
        .mean()
        .unstack(level=[1, 2])
    )
    std_times = (
        results_df.groupby(["n", "method", "timing_method"])["time"]
        .std()
        .unstack(level=[1, 2])
    )
    avg_memory = results_df.groupby(["n", "method"])["memory"].mean().unstack()
    std_memory = results_df.groupby(["n", "method"])["memory"].std().unstack()

    # Print comparison for CUDA events
    print("\n" + "=" * 90)
    print("Speed Comparison: Simple MatMul vs MatMul with Gradients (CUDA Events)")
    print("=" * 90)
    print(
        f"\n{'N (size)':<12} {'Simple MatMul (ms)':<20} {'MatMul Grad (ms)':<20} {'Speedup':<12}"
    )
    print("-" * 90)
    for n in sorted(avg_times.index):
        simple_time = avg_times.loc[n, ("matmul_simple", "cuda_events")]
        grad_time = avg_times.loc[n, ("matmul_grad", "cuda_events")]
        speedup = grad_time / simple_time
        print(f"{n:<12} {simple_time:<20.2f} {grad_time:<20.2f} {speedup:<12.2f}x")

    print("-" * 90)
    simple_mean = avg_times[("matmul_simple", "cuda_events")].mean()
    grad_mean = avg_times[("matmul_grad", "cuda_events")].mean()
    print(
        f"{'Average':<12} {simple_mean:<20.2f} {grad_mean:<20.2f} {(grad_mean / simple_mean):<12.2f}x"
    )

    # Print comparison for Benchmark Timer
    print("\n" + "=" * 90)
    print("Speed Comparison: Simple MatMul vs MatMul with Gradients (Benchmark Timer)")
    print("=" * 90)
    print(
        f"\n{'N (size)':<12} {'Simple MatMul (ms)':<20} {'MatMul Grad (ms)':<20} {'Speedup':<12}"
    )
    print("-" * 90)
    for n in sorted(avg_times.index):
        simple_time = avg_times.loc[n, ("matmul_simple", "benchmark")]
        grad_time = avg_times.loc[n, ("matmul_grad", "benchmark")]
        speedup = grad_time / simple_time
        print(f"{n:<12} {simple_time:<20.2f} {grad_time:<20.2f} {speedup:<12.2f}x")

    print("-" * 90)
    simple_mean = avg_times[("matmul_simple", "benchmark")].mean()
    grad_mean = avg_times[("matmul_grad", "benchmark")].mean()
    print(
        f"{'Average':<12} {simple_mean:<20.2f} {grad_mean:<20.2f} {(grad_mean / simple_mean):<12.2f}x"
    )

    # Print timing method comparison
    print("\n" + "=" * 90)
    print("Timing Method Comparison: CUDA Events vs Benchmark Timer")
    print("=" * 90)
    print(
        f"\n{'N (size)':<12} {'Simple: CUDA (ms)':<18} {'Simple: Bench (ms)':<18} {'Ratio':<12} {'Grad: CUDA (ms)':<18} {'Grad: Bench (ms)':<18} {'Ratio':<12}"
    )
    print("-" * 90)
    for n in sorted(avg_times.index):
        simple_cuda = avg_times.loc[n, ("matmul_simple", "cuda_events")]
        simple_bench = avg_times.loc[n, ("matmul_simple", "benchmark")]
        simple_ratio = simple_bench / simple_cuda
        grad_cuda = avg_times.loc[n, ("matmul_grad", "cuda_events")]
        grad_bench = avg_times.loc[n, ("matmul_grad", "benchmark")]
        grad_ratio = grad_bench / grad_cuda
        print(
            f"{n:<12} {simple_cuda:<18.2f} {simple_bench:<18.2f} {simple_ratio:<12.3f} {grad_cuda:<18.2f} {grad_bench:<18.2f} {grad_ratio:<12.3f}"
        )

    print("-" * 90)
    simple_cuda_mean = avg_times[("matmul_simple", "cuda_events")].mean()
    simple_bench_mean = avg_times[("matmul_simple", "benchmark")].mean()
    grad_cuda_mean = avg_times[("matmul_grad", "cuda_events")].mean()
    grad_bench_mean = avg_times[("matmul_grad", "benchmark")].mean()
    print(
        f"{'Average':<12} {simple_cuda_mean:<18.2f} {simple_bench_mean:<18.2f} {(simple_bench_mean / simple_cuda_mean):<12.3f} {grad_cuda_mean:<18.2f} {grad_bench_mean:<18.2f} {(grad_bench_mean / grad_cuda_mean):<12.3f}"
    )

    print("\n" + "=" * 70)
    print("Memory Comparison: Simple MatMul vs MatMul with Gradients")
    print("=" * 70)
    print(
        f"\n{'N (size)':<12} {'Simple MatMul (MB)':<20} {'MatMul Grad (MB)':<20} {'Ratio':<12}"
    )
    print("-" * 70)
    for n in sorted(avg_memory.index):
        simple_mem = avg_memory.loc[n, "matmul_simple"]
        grad_mem = avg_memory.loc[n, "matmul_grad"]
        ratio = grad_mem / simple_mem
        print(f"{n:<12} {simple_mem:<20.2f} {grad_mem:<20.2f} {ratio:<12.2f}x")

    print("-" * 70)
    print(
        f"{'Average':<12} {avg_memory['matmul_simple'].mean():<20.2f} {avg_memory['matmul_grad'].mean():<20.2f} {(avg_memory['matmul_grad'].mean() / avg_memory['matmul_simple'].mean()):<12.2f}x"
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
